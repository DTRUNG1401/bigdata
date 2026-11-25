"""Cấu hình chung của dự án.

Các tham số được đọc từ biến môi trường hoặc từ file ``.env`` ở thư mục gốc
dự án (xem ``.env.example``). Biến môi trường thật của hệ thống luôn được
ưu tiên hơn giá trị trong file ``.env``.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Thư mục gốc dự án (thư mục chứa src/, app/, docs/ ...)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _doc_so_nguyen(ten, mac_dinh, toi_thieu=None, toi_da=None):
    """Đọc biến môi trường kiểu số nguyên và kiểm tra khoảng giá trị.

    Args:
        ten: Tên biến môi trường.
        mac_dinh: Giá trị dùng khi biến không được đặt.
        toi_thieu: Giá trị nhỏ nhất cho phép (None = không giới hạn).
        toi_da: Giá trị lớn nhất cho phép (None = không giới hạn).

    Raises:
        ValueError: Nếu giá trị không phải số nguyên hoặc nằm ngoài khoảng.
    """
    gia_tri = os.getenv(ten, "").strip()
    if not gia_tri:
        return mac_dinh
    try:
        so = int(gia_tri)
    except ValueError:
        raise ValueError(
            f"Biến {ten}={gia_tri!r} không phải số nguyên. "
            "Hãy kiểm tra file .env."
        ) from None
    if (toi_thieu is not None and so < toi_thieu) or (
        toi_da is not None and so > toi_da
    ):
        raise ValueError(
            f"Biến {ten}={so} nằm ngoài khoảng cho phép "
            f"[{toi_thieu}, {toi_da}]. Hãy kiểm tra file .env."
        )
    return so


def _duong_dan(gia_tri):
    """Chuyển đường dẫn tương đối thành tuyệt đối (tính từ thư mục gốc)."""
    path = Path(gia_tri).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


# --- Kết nối MongoDB ---------------------------------------------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "ecommerce")

# --- Nạp dữ liệu ---------------------------------------------------------------
CSV_PATH = _duong_dan(os.getenv("CSV_PATH", "data/raw/2019-Oct.csv"))
SAMPLE_PERCENT = _doc_so_nguyen("SAMPLE_PERCENT", 100, 1, 100)
CHUNK_SIZE = _doc_so_nguyen("CHUNK_SIZE", 500_000, 1)

# --- Huấn luyện ----------------------------------------------------------------
MIN_USER_INTERACTIONS = _doc_so_nguyen("MIN_USER_INTERACTIONS", 2, 1)

# Điểm tương tác theo loại sự kiện (dùng cho bảng interactions)
EVENT_SCORES = {"view": 1, "cart": 3, "purchase": 5}

# Thư mục lưu mô hình đã huấn luyện (đã được .gitignore)
MODEL_DIR = PROJECT_ROOT / "models"

# --- Tên các collection --------------------------------------------------------
COL_EVENTS = "events"
COL_PRODUCTS = "products"
COL_INTERACTIONS = "interactions"
COL_RECOMMENDATIONS = "recommendations"
COL_SIMILAR_ITEMS = "similar_items"
COL_POPULAR_PRODUCTS = "popular_products"
COL_MODEL_METRICS = "model_metrics"

COL_STATS_OVERVIEW = "stats_overview"
COL_STATS_FUNNEL = "stats_funnel"
COL_STATS_TOP_CATEGORIES = "stats_top_categories"
COL_STATS_TOP_BRANDS = "stats_top_brands"
COL_STATS_TOP_PRODUCTS = "stats_top_products"
COL_STATS_BY_HOUR = "stats_by_hour"
COL_STATS_BY_WEEKDAY = "stats_by_weekday"
COL_STATS_RFM = "stats_rfm"
COL_STATS_RFM_SUMMARY = "stats_rfm_summary"


def thiet_lap_console():
    """Ép đầu ra console dùng UTF-8 để hiển thị đúng tiếng Việt.

    Cần thiết trên Windows khi đầu ra bị chuyển hướng ra file hoặc console
    dùng bảng mã cũ (cp1252).
    """
    for luong in (sys.stdout, sys.stderr):
        if hasattr(luong, "reconfigure"):
            luong.reconfigure(encoding="utf-8", errors="replace")
