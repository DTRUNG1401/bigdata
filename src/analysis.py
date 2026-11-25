"""Phân tích dữ liệu hành vi bằng aggregation pipeline của MongoDB.

Mọi phép tính nặng đều chạy trong MongoDB (``allowDiskUse=True`` để không
bị giới hạn 100MB bộ nhớ mỗi stage). Kết quả được lưu thành các collection
``stats_*`` để dashboard Streamlit đọc trực tiếp:

    stats_overview, stats_funnel, stats_top_categories, stats_top_brands,
    stats_top_products, stats_by_hour, stats_by_weekday, stats_rfm,
    stats_rfm_summary

Cách dùng:
    python src/analysis.py
"""

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

if __package__ in (None, ""):  # chạy trực tiếp: python src/analysis.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import (
    COL_EVENTS,
    COL_STATS_BY_HOUR,
    COL_STATS_BY_WEEKDAY,
    COL_STATS_FUNNEL,
    COL_STATS_OVERVIEW,
    COL_STATS_RFM,
    COL_STATS_RFM_SUMMARY,
    COL_STATS_TOP_BRANDS,
    COL_STATS_TOP_CATEGORIES,
    COL_STATS_TOP_PRODUCTS,
    thiet_lap_console,
)
from src.db import get_db_hoac_thoat

SO_LUONG_TOP = 20
BATCH_GHI = 50_000

NHAN_THU = {
    1: "Thứ Hai",
    2: "Thứ Ba",
    3: "Thứ Tư",
    4: "Thứ Năm",
    5: "Thứ Sáu",
    6: "Thứ Bảy",
    7: "Chủ Nhật",
}

# Các nhóm khách hàng RFM, theo thứ tự hiển thị
NHOM_VIP = "Khách VIP"
NHOM_TRUNG_THANH = "Khách trung thành"
NHOM_TIEM_NANG = "Khách tiềm năng"
NHOM_CAN_CHAM_SOC = "Khách cần chăm sóc"
NHOM_SAP_ROI_BO = "Khách sắp rời bỏ"
THU_TU_NHOM = [
    NHOM_VIP,
    NHOM_TRUNG_THANH,
    NHOM_TIEM_NANG,
    NHOM_CAN_CHAM_SOC,
    NHOM_SAP_ROI_BO,
]


# --- Hàm hỗ trợ --------------------------------------------------------------
def chay_pipeline(collection, pipeline):
    """Chạy aggregation pipeline (cho phép dùng đĩa) và trả về list kết quả."""
    return list(collection.aggregate(pipeline, allowDiskUse=True))


def luu_ket_qua(db, ten_collection, tai_lieu, cac_index=()):
    """Ghi đè một collection kết quả bằng danh sách tài liệu mới.

    Args:
        db: Đối tượng database.
        ten_collection: Tên collection đích.
        tai_lieu: Danh sách dict cần ghi.
        cac_index: Các tuple (trường, unique) cần tạo index.
    """
    collection = db[ten_collection]
    collection.drop()
    for vi_tri in range(0, len(tai_lieu), BATCH_GHI):
        collection.insert_many(tai_lieu[vi_tri:vi_tri + BATCH_GHI])
    for truong, duy_nhat in cac_index:
        collection.create_index(truong, unique=duy_nhat)


def _dem_loai(loai):
    """Biểu thức $sum đếm số sự kiện thuộc một loại."""
    return {"$sum": {"$cond": [{"$eq": ["$event_type", loai]}, 1, 0]}}


def _doanh_thu():
    """Biểu thức $sum cộng giá của các sự kiện mua hàng."""
    return {
        "$sum": {
            "$cond": [
                {"$eq": ["$event_type", "purchase"]},
                {"$ifNull": ["$price", 0]},
                0,
            ]
        }
    }


def _dem_phan_biet(events, truong):
    """Đếm số giá trị khác nhau của một trường bằng $group + $count."""
    ket_qua = chay_pipeline(
        events, [{"$group": {"_id": f"${truong}"}}, {"$count": "n"}]
    )
    return ket_qua[0]["n"] if ket_qua else 0


# --- 1. Tổng quan ------------------------------------------------------------
def phan_tich_tong_quan(db):
    """Tính các chỉ số tổng quan và lưu vào ``stats_overview``."""
    events = db[COL_EVENTS]
    tong = chay_pipeline(
        events,
        [
            {
                "$group": {
                    "_id": None,
                    "total_events": {"$sum": 1},
                    "total_revenue": _doanh_thu(),
                    "first_event": {"$min": "$event_time"},
                    "last_event": {"$max": "$event_time"},
                }
            }
        ],
    )[0]
    tai_lieu = {
        "_id": "overview",
        "total_events": tong["total_events"],
        "total_users": _dem_phan_biet(events, "user_id"),
        "total_products": _dem_phan_biet(events, "product_id"),
        "total_sessions": _dem_phan_biet(events, "user_session"),
        "total_revenue": round(tong["total_revenue"], 2),
        "first_event": tong["first_event"],
        "last_event": tong["last_event"],
        "computed_at": datetime.now(timezone.utc),
    }
    luu_ket_qua(db, COL_STATS_OVERVIEW, [tai_lieu])
    return tai_lieu


# --- 2. Phễu chuyển đổi ------------------------------------------------------
def phan_tich_pheu(db):
    """Tính phễu view -> cart -> purchase và lưu vào ``stats_funnel``."""

    def ty_le(tu, den):
        return {
            "$cond": [
                {"$gt": [f"${tu}", 0]},
                {"$divide": [f"${den}", f"${tu}"]},
                0,
            ]
        }

    ket_qua = chay_pipeline(
        db[COL_EVENTS],
        [
            {
                "$group": {
                    "_id": None,
                    "views": _dem_loai("view"),
                    "carts": _dem_loai("cart"),
                    "purchases": _dem_loai("purchase"),
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "views": 1,
                    "carts": 1,
                    "purchases": 1,
                    "view_to_cart": ty_le("views", "carts"),
                    "cart_to_purchase": ty_le("carts", "purchases"),
                    "view_to_purchase": ty_le("views", "purchases"),
                }
            },
        ],
    )[0]
    ket_qua["_id"] = "funnel"
    luu_ket_qua(db, COL_STATS_FUNNEL, [ket_qua])
    return ket_qua


# --- 3. Bảng xếp hạng danh mục / thương hiệu / sản phẩm ----------------------
def phan_tich_xep_hang(db, truong, ten_collection, truong_them=()):
    """Top 20 theo doanh thu và theo lượt mua của một chiều phân tích.

    Chỉ xét sự kiện ``purchase``. Mỗi tài liệu kết quả có ``metric``
    ("revenue" hoặc "purchases"), ``rank`` (1..20), giá trị của ``truong``,
    ``revenue`` và ``purchases``.

    Args:
        db: Đối tượng database.
        truong: Trường nhóm (category_code, brand hoặc product_id).
        ten_collection: Collection đích.
        truong_them: Các trường mô tả thêm (lấy bằng $max, bỏ qua null).
    """
    loc = [{"$match": {"event_type": "purchase"}}]
    if truong != "product_id":  # danh mục/thương hiệu rỗng không có ý nghĩa
        loc.append({"$match": {truong: {"$ne": None}}})
    nhom = {
        "$group": {
            "_id": f"${truong}",
            "revenue": {"$sum": {"$ifNull": ["$price", 0]}},
            "purchases": {"$sum": 1},
            **{t: {"$max": f"${t}"} for t in truong_them},
        }
    }
    chon_cot = {
        "$project": {
            "_id": 0,
            truong: "$_id",
            "revenue": {"$round": ["$revenue", 2]},
            "purchases": 1,
            **{t: 1 for t in truong_them},
        }
    }

    tai_lieu = []
    for tieu_chi in ("revenue", "purchases"):
        pipeline = loc + [
            nhom,
            {"$sort": {tieu_chi: -1, "_id": 1}},
            {"$limit": SO_LUONG_TOP},
            chon_cot,
        ]
        ket_qua = chay_pipeline(db[COL_EVENTS], pipeline)
        for hang, dong in enumerate(ket_qua, 1):
            tai_lieu.append({"metric": tieu_chi, "rank": hang, **dong})
    luu_ket_qua(db, ten_collection, tai_lieu)
    return tai_lieu


# --- 4. Phân bố theo giờ và theo thứ -----------------------------------------
def _phan_tich_theo_thoi_gian(
    db, bieu_thuc_nhom, ten_collection, truong, day_du
):
    """Gom số sự kiện và doanh thu theo một đơn vị thời gian.

    Args:
        db: Đối tượng database.
        bieu_thuc_nhom: Biểu thức lấy giờ/thứ từ ``event_time``.
        ten_collection: Collection đích.
        truong: Tên trường khoá trong tài liệu kết quả ("hour"/"weekday").
        day_du: Danh sách mọi giá trị khoá có thể có (để bù giá trị 0).
    """
    ket_qua = chay_pipeline(
        db[COL_EVENTS],
        [
            {
                "$group": {
                    "_id": bieu_thuc_nhom,
                    "events": {"$sum": 1},
                    "views": _dem_loai("view"),
                    "carts": _dem_loai("cart"),
                    "purchases": _dem_loai("purchase"),
                    "revenue": _doanh_thu(),
                }
            },
            {"$sort": {"_id": 1}},
        ],
    )
    co_san = {dong["_id"]: dong for dong in ket_qua}
    tai_lieu = []
    for khoa in day_du:
        dong = co_san.get(khoa, {})
        muc = {
            truong: khoa,
            "events": dong.get("events", 0),
            "views": dong.get("views", 0),
            "carts": dong.get("carts", 0),
            "purchases": dong.get("purchases", 0),
            "revenue": round(dong.get("revenue", 0), 2),
        }
        if truong == "weekday":
            muc["label"] = NHAN_THU[khoa]
        tai_lieu.append(muc)
    luu_ket_qua(db, ten_collection, tai_lieu)
    return tai_lieu


def phan_tich_theo_gio(db):
    """Số sự kiện và doanh thu theo giờ trong ngày (0-23, giờ UTC)."""
    return _phan_tich_theo_thoi_gian(
        db, {"$hour": "$event_time"}, COL_STATS_BY_HOUR, "hour", range(24)
    )


def phan_tich_theo_thu(db):
    """Số sự kiện và doanh thu theo thứ trong tuần (1 = Thứ Hai ... 7 = CN)."""
    return _phan_tich_theo_thoi_gian(
        db,
        {"$isoDayOfWeek": "$event_time"},
        COL_STATS_BY_WEEKDAY,
        "weekday",
        range(1, 8),
    )


# --- 5. Phân nhóm khách hàng RFM ---------------------------------------------
def cham_diem_phan_vi(thu_tu, so_nhom=5):
    """Chấm điểm 1..so_nhom theo phân vị bằng ``pd.qcut``.

    Args:
        thu_tu: Series thứ hạng (1..n, không trùng); thứ hạng càng lớn càng
            tốt. Dùng thứ hạng thay cho giá trị gốc để ``qcut`` luôn chia
            được đủ số nhóm dù có nhiều giá trị trùng nhau.
        so_nhom: Số nhóm điểm (mặc định 5).

    Returns:
        Series số nguyên 1..so_nhom; 5 là tốt nhất.
    """
    n = len(thu_tu)
    if n < so_nhom:  # quá ít khách để qcut chia nhóm: chia đều theo hạng
        return np.ceil(thu_tu * so_nhom / n).astype(int)
    return pd.qcut(thu_tu, so_nhom, labels=range(1, so_nhom + 1)).astype(int)


def phan_nhom_rfm(df):
    """Thêm điểm R, F, M (1-5) và nhóm khách hàng vào bảng RFM.

    * Recency càng nhỏ (mới mua gần đây) càng tốt.
    * Frequency (số phiên có mua) và Monetary (tổng chi tiêu) càng lớn càng
      tốt. Frequency trùng nhau rất nhiều (đa số khách chỉ mua 1 phiên) nên
      khi xếp hạng, các khách cùng Frequency được phân biệt tiếp bằng
      Monetary.

    Quy tắc nhóm (xét lần lượt từ trên xuống):
        * Khách VIP           : R >= 4, F >= 4 và M >= 4
        * Khách trung thành   : R >= 3 và F >= 4
        * Khách tiềm năng     : R >= 4 (mới mua nhưng chưa thường xuyên)
        * Khách cần chăm sóc  : R = 3, hoặc R <= 2 nhưng từng mua nhiều
        * Khách sắp rời bỏ    : còn lại (lâu không mua và ít mua)
    """
    df = df.copy()
    n = len(df)

    def xep_hang(*khoa_sap_xep):
        """Thứ hạng 1..n theo các khoá (khoá cuối cùng là khoá chính)."""
        thu_tu = np.empty(n, dtype=np.int64)
        thu_tu[np.lexsort(khoa_sap_xep)] = np.arange(1, n + 1)
        return pd.Series(thu_tu, index=df.index)

    recency = df["recency_days"].to_numpy()
    tan_suat = df["frequency"].to_numpy()
    chi_tieu = df["monetary"].to_numpy()
    # lexsort: khoá cuối là khoá chính. Recency nhỏ -> hạng cao (đảo dấu).
    df["r_score"] = cham_diem_phan_vi(xep_hang(-recency))
    df["f_score"] = cham_diem_phan_vi(xep_hang(chi_tieu, tan_suat))
    df["m_score"] = cham_diem_phan_vi(xep_hang(chi_tieu))

    r, f, m = df["r_score"], df["f_score"], df["m_score"]
    df["segment"] = np.select(
        [
            (r >= 4) & (f >= 4) & (m >= 4),
            (r >= 3) & (f >= 4),
            r >= 4,
            (r >= 3) | (f >= 4),
        ],
        [NHOM_VIP, NHOM_TRUNG_THANH, NHOM_TIEM_NANG, NHOM_CAN_CHAM_SOC],
        default=NHOM_SAP_ROI_BO,
    )
    df["rfm_score"] = (
        r.astype(str) + f.astype(str) + m.astype(str)
    )
    return df


def phan_tich_rfm(db):
    """Tính RFM cho mọi user có mua hàng; lưu ``stats_rfm`` và tóm tắt.

    * Recency  : số ngày từ lần mua gần nhất đến ngày sau sự kiện cuối cùng.
    * Frequency: số phiên (user_session) có phát sinh mua hàng.
    * Monetary : tổng giá trị các đơn mua.
    """
    events = db[COL_EVENTS]
    ngay_cuoi = chay_pipeline(
        events,
        [{"$group": {"_id": None, "last": {"$max": "$event_time"}}}],
    )[0]["last"]
    moc_tinh = ngay_cuoi + timedelta(days=1)

    # Gom 2 tầng: (user, phiên) -> user, để Frequency = số phiên có mua
    ket_qua = chay_pipeline(
        events,
        [
            {"$match": {"event_type": "purchase"}},
            {
                "$group": {
                    "_id": {"user_id": "$user_id", "session": "$user_session"},
                    "revenue": {"$sum": {"$ifNull": ["$price", 0]}},
                    "last_purchase": {"$max": "$event_time"},
                }
            },
            {
                "$group": {
                    "_id": "$_id.user_id",
                    "frequency": {"$sum": 1},
                    "monetary": {"$sum": "$revenue"},
                    "last_purchase": {"$max": "$last_purchase"},
                }
            },
        ],
    )
    if not ket_qua:
        luu_ket_qua(db, COL_STATS_RFM, [])
        luu_ket_qua(db, COL_STATS_RFM_SUMMARY, [])
        return []

    # Sắp theo user_id để kết quả xếp hạng tái lập được: thứ tự $group trả
    # về thay đổi giữa các lần chạy, làm khách bằng điểm nhau bị xếp khác nhau
    df = pd.DataFrame(ket_qua).rename(columns={"_id": "user_id"})
    df = df.sort_values("user_id").reset_index(drop=True)
    df["recency_days"] = (
        (moc_tinh - pd.to_datetime(df["last_purchase"])).dt.total_seconds()
        / 86_400
    ).round(2)
    df["monetary"] = df["monetary"].round(2)
    df = phan_nhom_rfm(df)

    cot = [
        "user_id", "recency_days", "frequency", "monetary",
        "r_score", "f_score", "m_score", "rfm_score", "segment",
    ]
    df = df[cot].astype(
        {"user_id": "int64", "frequency": "int64", "recency_days": "float64",
         "monetary": "float64", "r_score": "int64", "f_score": "int64",
         "m_score": "int64"}
    )
    luu_ket_qua(
        db,
        COL_STATS_RFM,
        df.to_dict("records"),
        cac_index=[("user_id", True), ("segment", False)],
    )

    tom_tat = chay_pipeline(
        db[COL_STATS_RFM],
        [
            {
                "$group": {
                    "_id": "$segment",
                    "users": {"$sum": 1},
                    "revenue": {"$sum": "$monetary"},
                    "avg_recency_days": {"$avg": "$recency_days"},
                    "avg_frequency": {"$avg": "$frequency"},
                    "avg_monetary": {"$avg": "$monetary"},
                }
            },
            # Không khai báo window: mặc định tính trên toàn bộ các nhóm
            {
                "$setWindowFields": {
                    "output": {
                        "total_users": {"$sum": "$users"},
                        "total_revenue": {"$sum": "$revenue"},
                    }
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "segment": "$_id",
                    "order": {"$indexOfArray": [THU_TU_NHOM, "$_id"]},
                    "users": 1,
                    "revenue": {"$round": ["$revenue", 2]},
                    "pct_users": {
                        "$round": [
                            {"$multiply": [
                                {"$divide": ["$users", "$total_users"]}, 100]},
                            2,
                        ]
                    },
                    "pct_revenue": {
                        "$round": [
                            {"$multiply": [
                                {"$divide": ["$revenue", "$total_revenue"]},
                                100]},
                            2,
                        ]
                    },
                    "avg_recency_days": {"$round": ["$avg_recency_days", 1]},
                    "avg_frequency": {"$round": ["$avg_frequency", 2]},
                    "avg_monetary": {"$round": ["$avg_monetary", 2]},
                }
            },
            {"$sort": {"order": 1}},
            {"$project": {"order": 0}},
        ],
    )
    luu_ket_qua(db, COL_STATS_RFM_SUMMARY, tom_tat)
    return tom_tat


# --- Điều phối ---------------------------------------------------------------
def chay_tat_ca(db=None):
    """Chạy toàn bộ các phân tích theo thứ tự và in thời gian từng phần.

    Returns:
        Dict {tên phân tích: số giây}.

    Raises:
        RuntimeError: Nếu collection ``events`` chưa có dữ liệu.
    """
    db = db if db is not None else get_db_hoac_thoat()
    if db[COL_EVENTS].estimated_document_count() == 0:
        raise RuntimeError(
            f"Collection '{COL_EVENTS}' đang trống. "
            "Hãy chạy src/import_data.py trước."
        )

    cac_buoc = [
        ("Tổng quan", phan_tich_tong_quan),
        ("Phễu chuyển đổi", phan_tich_pheu),
        (
            "Top danh mục",
            lambda d: phan_tich_xep_hang(
                d, "category_code", COL_STATS_TOP_CATEGORIES
            ),
        ),
        (
            "Top thương hiệu",
            lambda d: phan_tich_xep_hang(d, "brand", COL_STATS_TOP_BRANDS),
        ),
        (
            "Top sản phẩm",
            lambda d: phan_tich_xep_hang(
                d,
                "product_id",
                COL_STATS_TOP_PRODUCTS,
                truong_them=("category_code", "brand"),
            ),
        ),
        ("Theo giờ trong ngày", phan_tich_theo_gio),
        ("Theo thứ trong tuần", phan_tich_theo_thu),
        ("Phân nhóm RFM", phan_tich_rfm),
    ]
    thoi_gian = {}
    for ten, ham in cac_buoc:
        bat_dau = time.perf_counter()
        ham(db)
        thoi_gian[ten] = time.perf_counter() - bat_dau
        print(f"  [OK] {ten:<22} {thoi_gian[ten]:7.1f} giây")
    return thoi_gian


def in_tom_tat(db):
    """In nhanh một số kết quả chính để kiểm tra bằng mắt."""
    tq = db[COL_STATS_OVERVIEW].find_one({"_id": "overview"})
    pheu = db[COL_STATS_FUNNEL].find_one({"_id": "funnel"})
    print("\n--- Tóm tắt ---")
    print(f"Sự kiện: {tq['total_events']:,} | User: {tq['total_users']:,} | "
          f"Sản phẩm: {tq['total_products']:,} | Phiên: "
          f"{tq['total_sessions']:,} | Doanh thu: {tq['total_revenue']:,.2f}")
    print(f"Phễu: view {pheu['views']:,} -> cart {pheu['carts']:,} -> "
          f"purchase {pheu['purchases']:,} "
          f"(view->cart {pheu['view_to_cart']:.2%}, "
          f"cart->purchase {pheu['cart_to_purchase']:.2%}, "
          f"view->purchase {pheu['view_to_purchase']:.2%})")
    for dong in db[COL_STATS_RFM_SUMMARY].find({}, {"_id": 0}):
        print(f"RFM | {dong['segment']:<20} {dong['users']:>7,} user "
              f"({dong['pct_users']:5.1f}%) | doanh thu "
              f"{dong['pct_revenue']:5.1f}%")


def main():
    """Điểm vào dòng lệnh."""
    thiet_lap_console()
    db = get_db_hoac_thoat()
    print("Phân tích dữ liệu bằng aggregation pipeline ...")
    try:
        chay_tat_ca(db)
    except RuntimeError as loi:
        print(f"LỖI: {loi}", file=sys.stderr)
        return 1
    in_tom_tat(db)
    return 0


if __name__ == "__main__":
    sys.exit(main())
