"""Chạy toàn bộ pipeline từ CSV đến mô hình gợi ý.

Thứ tự: nạp dữ liệu -> phân tích -> tạo bảng tương tác -> huấn luyện mô hình.
In thời gian của từng bước. Dừng ngay và báo bước bị lỗi nếu có sự cố.

Cách dùng:
    python src/run_pipeline.py --drop
    python src/run_pipeline.py --csv data/raw/sample_events.csv --drop
    python src/run_pipeline.py --skip-import      # chỉ chạy lại các bước sau
"""

import argparse
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # chạy trực tiếp: python src/run_pipeline.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import analysis, build_interactions, import_data, train_model
from src.config import (
    CHUNK_SIZE,
    CSV_PATH,
    MIN_USER_INTERACTIONS,
    SAMPLE_PERCENT,
    thiet_lap_console,
)
from src.db import get_db_hoac_thoat


def tao_parser():
    """Tạo bộ phân tích tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Chạy toàn bộ pipeline: nạp dữ liệu -> phân tích -> "
                    "bảng tương tác -> huấn luyện mô hình."
    )
    nap = parser.add_argument_group("nạp dữ liệu")
    nap.add_argument("--csv", type=Path, default=None,
                     help=f"đường dẫn file CSV (mặc định: {CSV_PATH})")
    nap.add_argument("--sample-percent", type=int, default=None,
                     help="phần trăm người dùng giữ lại, 1-100 "
                          f"(mặc định: {SAMPLE_PERCENT})")
    nap.add_argument("--chunk-size", type=int, default=None,
                     help=f"số dòng mỗi chunk (mặc định: {CHUNK_SIZE:,})")
    nap.add_argument("--drop", action="store_true",
                     help="xoá collection events cũ trước khi nạp")
    nap.add_argument("--append", action="store_true",
                     help="cho phép nạp thêm vào collection đã có dữ liệu")
    nap.add_argument("--skip-import", action="store_true",
                     help="bỏ qua bước nạp, dùng dữ liệu events đã có")

    mo_hinh = parser.add_argument_group("mô hình ALS")
    mo_hinh.add_argument("--factors", type=int,
                         default=train_model.FACTORS_MAC_DINH)
    mo_hinh.add_argument("--regularization", type=float,
                         default=train_model.REGULARIZATION_MAC_DINH)
    mo_hinh.add_argument("--alpha", type=float,
                         default=train_model.ALPHA_MAC_DINH)
    mo_hinh.add_argument("--iterations", type=int,
                         default=train_model.ITERATIONS_MAC_DINH)
    mo_hinh.add_argument("--min-user-interactions", type=int, default=None,
                         help="số sản phẩm tối thiểu của một người dùng "
                              f"(mặc định: {MIN_USER_INTERACTIONS})")
    mo_hinh.add_argument("--seed", type=int,
                         default=train_model.SEED_MAC_DINH)
    mo_hinh.add_argument("--no-save-model", action="store_true",
                         help="không lưu file mô hình vào thư mục models/")
    return parser


def main(argv=None):
    """Điểm vào dòng lệnh. Trả về 0 nếu thành công, 1 nếu có lỗi."""
    thiet_lap_console()
    args = tao_parser().parse_args(argv)
    db = get_db_hoac_thoat()

    def buoc_nap():
        import_data.run(
            args.csv, args.sample_percent, args.chunk_size,
            drop=args.drop, append=args.append, db=db,
        )

    def buoc_phan_tich():
        analysis.chay_tat_ca(db)
        analysis.in_tom_tat(db)

    def buoc_tuong_tac():
        build_interactions.run(db, args.min_user_interactions)

    def buoc_huan_luyen():
        train_model.run(
            db,
            factors=args.factors,
            regularization=args.regularization,
            alpha=args.alpha,
            iterations=args.iterations,
            min_user_interactions=args.min_user_interactions,
            seed=args.seed,
            luu_file_mo_hinh=not args.no_save_model,
        )

    cac_buoc = []
    if not args.skip_import:
        cac_buoc.append(("Nạp dữ liệu vào MongoDB", buoc_nap))
    cac_buoc += [
        ("Phân tích dữ liệu (aggregation)", buoc_phan_tich),
        ("Tạo bảng sản phẩm và tương tác", buoc_tuong_tac),
        ("Huấn luyện và đánh giá mô hình ALS", buoc_huan_luyen),
    ]

    thoi_gian = []
    bat_dau_chung = time.perf_counter()
    for so_thu_tu, (ten, ham) in enumerate(cac_buoc, start=1):
        print(f"\n{'=' * 70}\nBƯỚC {so_thu_tu}/{len(cac_buoc)}: {ten}\n"
              f"{'=' * 70}")
        bat_dau = time.perf_counter()
        try:
            ham()
        except (FileNotFoundError, ValueError, RuntimeError) as loi:
            print(f"\nLỖI ở bước '{ten}': {loi}", file=sys.stderr)
            print("Pipeline đã dừng.", file=sys.stderr)
            return 1
        thoi_gian.append((ten, time.perf_counter() - bat_dau))
        print(f"-> Xong bước {so_thu_tu} trong {thoi_gian[-1][1]:.1f} giây.")

    print(f"\n{'=' * 70}\nTHỜI GIAN TỪNG BƯỚC\n{'=' * 70}")
    for ten, giay in thoi_gian:
        print(f"  {ten:<40}{giay:>10.1f} giây")
    print(f"  {'Tổng cộng':<40}"
          f"{time.perf_counter() - bat_dau_chung:>10.1f} giây")
    print("\nHoàn tất. Chạy dashboard: streamlit run app/app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
