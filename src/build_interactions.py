"""Tạo bảng sản phẩm và bảng tương tác (user x product) từ ``events``.

Hai collection được tạo hoàn toàn bằng aggregation pipeline với ``$out``:

* ``products``    : thông tin mỗi sản phẩm (danh mục, thương hiệu, giá trung
  bình, số lượt xem / thêm giỏ / mua).
* ``interactions``: mỗi dòng là một cặp (user_id, product_id) với điểm tương
  tác ``score`` = tổng điểm các sự kiện (view = 1, cart = 3, purchase = 5).
  Đây là "phản hồi ngầm" (implicit feedback) làm đầu vào cho mô hình ALS.

Tham số ``MIN_USER_INTERACTIONS`` (mặc định 2) là số sản phẩm tối thiểu mà một
người dùng phải tương tác để được đưa vào huấn luyện (xem ``train_model.py``).
Ở bước này chỉ thống kê trước xem ngưỡng đó sẽ loại bao nhiêu người dùng.

Cách dùng:
    python src/build_interactions.py
"""

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # chạy trực tiếp bằng python src/<tên>.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    COL_EVENTS,
    COL_INTERACTIONS,
    COL_PRODUCTS,
    EVENT_SCORES,
    MIN_USER_INTERACTIONS,
    thiet_lap_console,
)
from src.db import get_db_hoac_thoat


def _dem_loai(loai):
    """Biểu thức $sum đếm số sự kiện thuộc một loại."""
    return {"$sum": {"$cond": [{"$eq": ["$event_type", loai]}, 1, 0]}}


def tao_bang_san_pham(db):
    """Tạo collection ``products`` bằng aggregation ($group + $out).

    category_code và brand lấy bằng ``$max`` vì $max bỏ qua giá trị null:
    nếu sản phẩm có ít nhất một sự kiện ghi đủ thông tin thì ta lấy được nó.

    Returns:
        Số sản phẩm đã tạo.
    """
    list(db[COL_EVENTS].aggregate(
        [
            {
                "$group": {
                    "_id": "$product_id",
                    "category_code": {"$max": "$category_code"},
                    "brand": {"$max": "$brand"},
                    "avg_price": {"$avg": "$price"},
                    "views": _dem_loai("view"),
                    "carts": _dem_loai("cart"),
                    "purchases": _dem_loai("purchase"),
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "product_id": "$_id",
                    "category_code": 1,
                    "brand": 1,
                    "avg_price": {"$round": ["$avg_price", 2]},
                    "views": 1,
                    "carts": 1,
                    "purchases": 1,
                }
            },
            {"$out": COL_PRODUCTS},
        ],
        allowDiskUse=True,
    ))  # $out không trả về dòng nào; list() để chạy hết pipeline
    db[COL_PRODUCTS].create_index("product_id", unique=True)
    return db[COL_PRODUCTS].estimated_document_count()


def tao_bang_tuong_tac(db):
    """Tạo collection ``interactions`` bằng aggregation ($group + $out).

    Điểm tương tác của một cặp (user, product) là tổng điểm của mọi sự kiện
    giữa họ theo ``EVENT_SCORES`` (view = 1, cart = 3, purchase = 5).

    Returns:
        Số cặp (user, product) đã tạo.
    """
    diem_su_kien = {
        "$switch": {
            "branches": [
                {"case": {"$eq": ["$event_type", loai]}, "then": diem}
                for loai, diem in EVENT_SCORES.items()
            ],
            "default": 0,
        }
    }
    list(db[COL_EVENTS].aggregate(
        [
            {
                "$group": {
                    "_id": {
                        "user_id": "$user_id",
                        "product_id": "$product_id",
                    },
                    "score": {"$sum": diem_su_kien},
                    "views": _dem_loai("view"),
                    "carts": _dem_loai("cart"),
                    "purchases": _dem_loai("purchase"),
                    "last_time": {"$max": "$event_time"},
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "user_id": "$_id.user_id",
                    "product_id": "$_id.product_id",
                    "score": 1,
                    "views": 1,
                    "carts": 1,
                    "purchases": 1,
                    "last_time": 1,
                }
            },
            {"$out": COL_INTERACTIONS},
        ],
        allowDiskUse=True,
    ))
    bang = db[COL_INTERACTIONS]
    bang.create_index("user_id")
    bang.create_index("product_id")
    return bang.estimated_document_count()


def thong_ke_loc_nguoi_dung(db, min_user_interactions):
    """Thống kê số người dùng được giữ / bị loại theo ngưỡng tương tác.

    Một người dùng được giữ nếu có từ ``min_user_interactions`` sản phẩm
    khác nhau trở lên trong bảng ``interactions``.

    Returns:
        Dict gồm số người dùng giữ, loại và tổng.
    """
    ket_qua = db[COL_INTERACTIONS].aggregate(
        [
            {"$group": {"_id": "$user_id", "n": {"$sum": 1}}},
            {
                "$group": {
                    "_id": {"$gte": ["$n", min_user_interactions]},
                    "users": {"$sum": 1},
                }
            },
        ],
        allowDiskUse=True,
    )
    dem = {dong["_id"]: dong["users"] for dong in ket_qua}
    giu, loai = dem.get(True, 0), dem.get(False, 0)
    return {"giu": giu, "loai": loai, "tong": giu + loai}


def run(db=None, min_user_interactions=None):
    """Tạo ``products`` và ``interactions``; in thống kê.

    Args:
        db: Đối tượng database (mặc định tự kết nối).
        min_user_interactions: Ngưỡng tương tác tối thiểu của người dùng
            (mặc định ``MIN_USER_INTERACTIONS`` trong cấu hình).

    Raises:
        RuntimeError: Nếu collection ``events`` chưa có dữ liệu.
    """
    db = db if db is not None else get_db_hoac_thoat()
    nguong = min_user_interactions or MIN_USER_INTERACTIONS
    if db[COL_EVENTS].estimated_document_count() == 0:
        raise RuntimeError(
            f"Collection '{COL_EVENTS}' đang trống. "
            "Hãy chạy src/import_data.py trước."
        )

    bat_dau = time.perf_counter()
    so_san_pham = tao_bang_san_pham(db)
    print(f"  [OK] products     : {so_san_pham:,} sản phẩm "
          f"({time.perf_counter() - bat_dau:.1f} giây)")

    bat_dau = time.perf_counter()
    so_tuong_tac = tao_bang_tuong_tac(db)
    print(f"  [OK] interactions : {so_tuong_tac:,} cặp (user, product) "
          f"({time.perf_counter() - bat_dau:.1f} giây)")

    tk = thong_ke_loc_nguoi_dung(db, nguong)
    ti_le = tk["loai"] / tk["tong"] if tk["tong"] else 0
    print(f"  Với MIN_USER_INTERACTIONS = {nguong}: giữ {tk['giu']:,} / "
          f"{tk['tong']:,} người dùng khi huấn luyện "
          f"(loại {tk['loai']:,} = {ti_le:.1%}).")
    return {"products": so_san_pham, "interactions": so_tuong_tac, **tk}


def main(argv=None):
    """Điểm vào dòng lệnh."""
    thiet_lap_console()
    parser = argparse.ArgumentParser(
        description="Tạo bảng products và interactions từ events."
    )
    parser.add_argument(
        "--min-user-interactions", type=int, default=None,
        help="ngưỡng số sản phẩm tối thiểu của một người dùng khi huấn luyện "
             f"(mặc định: {MIN_USER_INTERACTIONS}); ở bước này chỉ dùng để "
             "thống kê số người dùng bị loại",
    )
    args = parser.parse_args(argv)
    print("Tạo bảng sản phẩm và bảng tương tác ...")
    try:
        run(min_user_interactions=args.min_user_interactions)
    except RuntimeError as loi:
        print(f"LỖI: {loi}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
