"""Nạp dữ liệu từ file CSV vào MongoDB (collection ``events``).

Quy trình:
  1. Đọc CSV theo từng chunk bằng pandas (không đọc cả file vào RAM).
  2. Lấy mẫu theo người dùng: chỉ giữ dòng có
     ``user_id % 100 < SAMPLE_PERCENT`` để mỗi người dùng được giữ trọn vẹn
     lịch sử hành vi.
  3. Làm sạch: bỏ dòng thiếu user_id/product_id, đổi event_time sang datetime,
     đổi NaN thành null, bỏ cột category_id.
  4. ``insert_many(ordered=False)`` theo lô nhỏ, có thanh tiến trình tqdm.
  5. Tạo index: user_id, product_id, event_type, event_time.

Cách dùng:
    python src/import_data.py --drop
    python src/import_data.py --csv data/raw/sample_events.csv --drop
"""

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # chạy trực tiếp: python src/import_data.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from pymongo import ASCENDING
from pymongo.errors import BulkWriteError
from tqdm import tqdm

from src.config import (
    CHUNK_SIZE,
    COL_EVENTS,
    CSV_PATH,
    SAMPLE_PERCENT,
    thiet_lap_console,
)
from src.db import get_db_hoac_thoat

# Các cột được đọc từ CSV (category_id bị bỏ ngay từ lúc đọc để tiết kiệm RAM)
COT_GIU_LAI = [
    "event_time",
    "event_type",
    "product_id",
    "category_code",
    "brand",
    "price",
    "user_id",
    "user_session",
]

# Đọc id dưới dạng float64 để chịu được ô trống; sau đó mới ép về int64
KIEU_DU_LIEU = {
    "product_id": "float64",
    "user_id": "float64",
    "price": "float64",
}

DINH_DANG_THOI_GIAN = "%Y-%m-%d %H:%M:%S UTC"

# Số tài liệu trong mỗi lần insert_many (giới hạn RAM cho danh sách dict)
KICH_THUOC_LO = 50_000

# Các trường được đánh index
TRUONG_INDEX = ["user_id", "product_id", "event_type", "event_time"]


def kiem_tra_file_csv(csv_path):
    """Kiểm tra file CSV tồn tại và đủ các cột cần thiết.

    Raises:
        FileNotFoundError: Nếu không thấy file.
        ValueError: Nếu thiếu cột bắt buộc.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Không tìm thấy file CSV: {csv_path}\n"
            "  - Tải dữ liệu thật từ Kaggle (xem README), hoặc\n"
            "  - Sinh dữ liệu mẫu: python scripts/generate_sample.py rồi chạy "
            "với --csv data/raw/sample_events.csv"
        )
    cot_trong_file = set(pd.read_csv(csv_path, nrows=0).columns)
    thieu = [c for c in COT_GIU_LAI if c not in cot_trong_file]
    if thieu:
        raise ValueError(
            f"File CSV thiếu các cột bắt buộc: {', '.join(thieu)}. "
            "Cần đúng schema của bộ dữ liệu REES46."
        )


def lam_sach_chunk(chunk, sample_percent):
    """Làm sạch và lấy mẫu một chunk dữ liệu.

    Args:
        chunk: DataFrame đọc thô từ CSV.
        sample_percent: Giữ dòng có ``user_id % 100 < sample_percent``.

    Returns:
        Tuple (DataFrame đã làm sạch, dict số dòng bị loại theo từng lý do:
        ``thieu_truong``, ``lay_mau``, ``sai_thoi_gian``).
    """
    so_dong = len(chunk)

    # Bỏ dòng thiếu user_id / product_id / event_type
    chunk = chunk.dropna(subset=["user_id", "product_id", "event_type"])
    thieu_truong = so_dong - len(chunk)
    chunk = chunk.assign(
        user_id=chunk["user_id"].astype("int64"),
        product_id=chunk["product_id"].astype("int64"),
    )

    # Lấy mẫu theo người dùng để giữ nguyên lịch sử của từng user
    lay_mau = 0
    if sample_percent < 100:
        so_truoc = len(chunk)
        chunk = chunk[(chunk["user_id"] % 100) < sample_percent]
        lay_mau = so_truoc - len(chunk)

    # Chuyển event_time sang datetime; giá trị sai định dạng thành NaT rồi bỏ
    thoi_gian = pd.to_datetime(
        chunk["event_time"], format=DINH_DANG_THOI_GIAN, errors="coerce"
    )
    chunk = chunk.assign(event_time=thoi_gian)
    so_truoc = len(chunk)
    chunk = chunk.dropna(subset=["event_time"])
    sai_thoi_gian = so_truoc - len(chunk)

    return chunk, {
        "thieu_truong": thieu_truong,
        "lay_mau": lay_mau,
        "sai_thoi_gian": sai_thoi_gian,
    }


def _cot_thanh_danh_sach(serie):
    """Chuyển một cột pandas thành list Python, thay NaN bằng None (null)."""
    return serie.astype(object).where(serie.notna(), None).tolist()


def chuyen_thanh_tai_lieu(chunk):
    """Chuyển DataFrame đã làm sạch thành danh sách tài liệu MongoDB.

    Dựng từ các cột (zip) nhanh hơn nhiều so với ``DataFrame.to_dict``.
    """
    thoi_gian = (
        chunk["event_time"].to_numpy(dtype="datetime64[us]").astype(object)
    )
    cac_cot = zip(
        thoi_gian.tolist(),
        chunk["event_type"].tolist(),
        chunk["product_id"].tolist(),
        _cot_thanh_danh_sach(chunk["category_code"]),
        _cot_thanh_danh_sach(chunk["brand"]),
        _cot_thanh_danh_sach(chunk["price"]),
        chunk["user_id"].tolist(),
        chunk["user_session"].tolist(),
    )
    return [dict(zip(COT_GIU_LAI, hang)) for hang in cac_cot]


def chen_lo(collection, tai_lieu):
    """Chèn một lô tài liệu bằng ``insert_many(ordered=False)``.

    Returns:
        Tuple (số tài liệu chèn thành công, số tài liệu bị lỗi).
    """
    try:
        ket_qua = collection.insert_many(tai_lieu, ordered=False)
        return len(ket_qua.inserted_ids), 0
    except BulkWriteError as loi:
        chi_tiet = loi.details
        so_loi = len(chi_tiet.get("writeErrors", []))
        return chi_tiet.get("nInserted", 0), so_loi


def tao_index(collection):
    """Tạo các index phục vụ truy vấn và aggregation."""
    for truong in TRUONG_INDEX:
        collection.create_index([(truong, ASCENDING)])


def run(
    csv_path=None,
    sample_percent=None,
    chunk_size=None,
    drop=False,
    append=False,
    db=None,
):
    """Nạp dữ liệu CSV vào collection ``events``.

    Args:
        csv_path: Đường dẫn file CSV (mặc định ``CSV_PATH`` trong cấu hình).
        sample_percent: Phần trăm người dùng giữ lại (mặc định theo cấu hình).
        chunk_size: Số dòng đọc mỗi lượt (mặc định theo cấu hình).
        drop: Xoá collection ``events`` cũ trước khi nạp.
        append: Cho phép nạp thêm vào collection đã có dữ liệu.
        db: Đối tượng database (mặc định tự kết nối).

    Returns:
        Dict thống kê: số dòng đọc, giữ lại, chèn thành công, lỗi, thời gian.

    Raises:
        FileNotFoundError, ValueError: Nếu file CSV không hợp lệ.
        RuntimeError: Nếu collection đã có dữ liệu mà không chọn drop/append.
    """
    csv_path = Path(csv_path) if csv_path else CSV_PATH
    if sample_percent is None:
        sample_percent = SAMPLE_PERCENT
    chunk_size = chunk_size or CHUNK_SIZE
    if not 1 <= sample_percent <= 100:
        raise ValueError("sample_percent phải nằm trong khoảng 1..100")

    kiem_tra_file_csv(csv_path)
    db = db if db is not None else get_db_hoac_thoat()
    collection = db[COL_EVENTS]

    if drop:
        collection.drop()
        print(f"Đã xoá collection '{COL_EVENTS}' cũ.")
    elif collection.estimated_document_count() > 0 and not append:
        raise RuntimeError(
            f"Collection '{COL_EVENTS}' đã có dữ liệu. Nạp lại sẽ bị trùng "
            "lặp sự kiện và làm sai kết quả phân tích.\n"
            "  - Dùng --drop để xoá dữ liệu cũ rồi nạp lại từ đầu, hoặc\n"
            "  - Dùng --append nếu cố ý nạp thêm file khác, hoặc\n"
            "  - (với run_pipeline.py) dùng --skip-import để giữ dữ liệu "
            "đã nạp và chỉ chạy các bước sau."
        )

    print(f"Nạp file : {csv_path}")
    print(f"Lấy mẫu  : {sample_percent}% người dùng "
          f"(user_id % 100 < {sample_percent}) | chunk = {chunk_size:,} dòng")

    bat_dau = time.perf_counter()
    doc = giu_lai = da_chen = loi_chen = 0
    bi_loai = {"thieu_truong": 0, "lay_mau": 0, "sai_thoi_gian": 0}

    kich_thuoc_file = csv_path.stat().st_size
    with open(csv_path, "rb") as tep, tqdm(
        total=kich_thuoc_file,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Nạp CSV",
    ) as thanh:
        doc_chunk = pd.read_csv(
            tep,
            usecols=COT_GIU_LAI,
            dtype=KIEU_DU_LIEU,
            chunksize=chunk_size,
        )
        vi_tri_cu = 0
        for chunk in doc_chunk:
            doc += len(chunk)
            chunk, so_bi_loai = lam_sach_chunk(chunk, sample_percent)
            for ly_do, so_dong in so_bi_loai.items():
                bi_loai[ly_do] += so_dong
            giu_lai += len(chunk)

            for vi_tri in range(0, len(chunk), KICH_THUOC_LO):
                lo = chunk.iloc[vi_tri:vi_tri + KICH_THUOC_LO]
                ok, loi = chen_lo(collection, chuyen_thanh_tai_lieu(lo))
                da_chen += ok
                loi_chen += loi

            vi_tri_moi = tep.tell()
            thanh.update(vi_tri_moi - vi_tri_cu)
            vi_tri_cu = vi_tri_moi
            thanh.set_postfix(doc=f"{doc:,}", chen=f"{da_chen:,}")
        thanh.update(kich_thuoc_file - vi_tri_cu)

    print("Đang tạo index: " + ", ".join(TRUONG_INDEX) + " ...")
    tao_index(collection)

    thong_ke = {
        "so_dong_doc": doc,
        "so_dong_giu_lai": giu_lai,
        "so_dong_bi_loai": bi_loai,
        "so_dong_da_chen": da_chen,
        "so_dong_loi_chen": loi_chen,
        "so_giay": time.perf_counter() - bat_dau,
    }
    print(f"Đã đọc {doc:,} dòng. Loại bỏ: "
          f"thiếu user_id/product_id/event_type {bi_loai['thieu_truong']:,}, "
          f"sai định dạng thời gian {bi_loai['sai_thoi_gian']:,}, "
          f"ngoài mẫu {sample_percent}% người dùng {bi_loai['lay_mau']:,}.")
    print(f"Giữ lại {giu_lai:,} dòng; chèn thành công {da_chen:,} tài liệu "
          f"(lỗi chèn: {loi_chen:,}).")
    print(f"Hoàn tất trong {thong_ke['so_giay']:.1f} giây. "
          f"Tổng số tài liệu trong '{COL_EVENTS}': "
          f"{collection.estimated_document_count():,}")
    return thong_ke


def main(argv=None):
    """Điểm vào dòng lệnh."""
    thiet_lap_console()
    parser = argparse.ArgumentParser(
        description="Nạp file CSV hành vi người dùng vào MongoDB."
    )
    parser.add_argument("--csv", type=Path, default=None,
                        help=f"đường dẫn file CSV (mặc định: {CSV_PATH})")
    parser.add_argument("--sample-percent", type=int, default=None,
                        help="phần trăm người dùng giữ lại, 1-100 "
                             f"(mặc định: {SAMPLE_PERCENT})")
    parser.add_argument("--chunk-size", type=int, default=None,
                        help=f"số dòng mỗi chunk (mặc định: {CHUNK_SIZE:,})")
    parser.add_argument("--drop", action="store_true",
                        help="xoá collection events cũ trước khi nạp")
    parser.add_argument("--append", action="store_true",
                        help="cho phép nạp thêm vào collection đã có dữ liệu")
    args = parser.parse_args(argv)

    try:
        run(args.csv, args.sample_percent, args.chunk_size,
            args.drop, args.append)
    except (FileNotFoundError, ValueError, RuntimeError) as loi:
        print(f"LỖI: {loi}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
