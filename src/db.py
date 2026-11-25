"""Kết nối MongoDB.

Cung cấp hàm :func:`get_db` trả về đối tượng database đã được kiểm tra kết
nối, và báo lỗi tiếng Việt rõ ràng nếu MongoDB chưa chạy.
"""

import sys
from pathlib import Path

if __package__ in (None, ""):  # chạy trực tiếp: python src/db.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from src.config import DB_NAME, MONGO_URI

# Thời gian tối đa (ms) chờ tìm thấy máy chủ MongoDB
THOI_GIAN_CHO_MS = 5000


class MongoConnectionError(RuntimeError):
    """Không kết nối được tới MongoDB."""


def get_client(uri=None, timeout_ms=THOI_GIAN_CHO_MS):
    """Tạo MongoClient và kiểm tra kết nối bằng lệnh ping.

    Args:
        uri: Chuỗi kết nối. Mặc định lấy ``MONGO_URI`` trong cấu hình.
        timeout_ms: Thời gian chờ chọn máy chủ (mili giây).

    Returns:
        Đối tượng ``MongoClient`` đã kết nối thành công.

    Raises:
        MongoConnectionError: Nếu không kết nối được MongoDB.
    """
    uri = uri or MONGO_URI
    client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
    try:
        client.admin.command("ping")
    except PyMongoError as loi:
        client.close()
        chi_tiet = str(loi)
        if len(chi_tiet) > 150:
            chi_tiet = chi_tiet[:150] + "..."
        raise MongoConnectionError(
            f"Không kết nối được MongoDB tại '{uri}'.\n"
            "  - Kiểm tra Docker Desktop đã chạy chưa.\n"
            "  - Khởi động MongoDB bằng lệnh: docker compose up -d\n"
            "  - Kiểm tra lại MONGO_URI trong file .env.\n"
            f"Chi tiết lỗi: {loi.__class__.__name__}: {chi_tiet}"
        ) from None
    return client


def get_db(uri=None, db_name=None):
    """Trả về database của dự án sau khi đã kiểm tra kết nối.

    Args:
        uri: Chuỗi kết nối. Mặc định lấy ``MONGO_URI`` trong cấu hình.
        db_name: Tên database. Mặc định lấy ``DB_NAME`` trong cấu hình.

    Raises:
        MongoConnectionError: Nếu không kết nối được MongoDB.
    """
    return get_client(uri)[db_name or DB_NAME]


def get_db_hoac_thoat():
    """Giống :func:`get_db` nhưng in lỗi và thoát chương trình nếu thất bại.

    Dùng cho các script chạy dòng lệnh để người dùng thấy thông báo gọn,
    không bị dội cả traceback.
    """
    try:
        return get_db()
    except MongoConnectionError as loi:
        print(f"LỖI: {loi}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    # Kiểm tra nhanh: python src/db.py
    ket_noi = get_db_hoac_thoat()
    print(f"Kết nối thành công tới database '{ket_noi.name}'.")
    cac_collection = ket_noi.list_collection_names()
    print("Các collection hiện có:", cac_collection or "(chưa có)")
