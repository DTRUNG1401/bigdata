"""Huấn luyện mô hình gợi ý ALS (implicit feedback) và sinh gợi ý.

Các bước:
  1. Đọc ``interactions``, loại người dùng có ít hơn ``MIN_USER_INTERACTIONS``
     sản phẩm, mã hoá user_id/product_id thành chỉ số liên tục rồi dựng ma
     trận thưa CSR (user x item).
  2. Đánh giá: chia 80/20 bằng ``implicit.evaluation.train_test_split``, huấn
     luyện ALS trên 80%, đo precision/map/ndcg/auc @K và so sánh với baseline
     "gợi ý sản phẩm phổ biến nhất" trên cùng tập test. Lưu vào
     ``model_metrics``.
  3. Huấn luyện lại ALS trên toàn bộ dữ liệu.
  4. Sinh top-10 gợi ý cho mọi người dùng (``recommendations``), top-10 sản
     phẩm tương tự (``similar_items``) và top-50 sản phẩm phổ biến cho người
     dùng mới (``popular_products``, giải quyết cold-start).

Cách dùng:
    python src/train_model.py
    python src/train_model.py --factors 128 --iterations 30 --alpha 2.0
"""

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):  # chạy trực tiếp: python src/train_model.py
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from implicit.evaluation import ranking_metrics_at_k, train_test_split
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from src.config import (
    COL_INTERACTIONS,
    COL_MODEL_METRICS,
    COL_POPULAR_PRODUCTS,
    COL_PRODUCTS,
    COL_RECOMMENDATIONS,
    COL_SIMILAR_ITEMS,
    MIN_USER_INTERACTIONS,
    MODEL_DIR,
    thiet_lap_console,
)
from src.db import get_db_hoac_thoat

# --- Hằng số -----------------------------------------------------------------
K_DANH_GIA = 10                 # K trong precision@K, map@K, ndcg@K, auc@K
TY_LE_TRAIN = 0.8               # chia 80/20 khi đánh giá
SO_GOI_Y = 10                   # số sản phẩm gợi ý cho mỗi người dùng
SO_TUONG_TU = 10                # số sản phẩm tương tự cho mỗi sản phẩm
TUONG_TAC_TOI_THIEU_SP = 5      # sản phẩm cần >= 5 tương tác mới có "tương tự"
SO_SAN_PHAM_PHO_BIEN = 50       # kích thước bảng popular_products
BATCH_NGUOI_DUNG = 50_000       # số user mỗi lượt sinh gợi ý
BATCH_SAN_PHAM = 10_000         # số sản phẩm mỗi lượt tìm sản phẩm tương tự
KHOI_DOC = 500_000              # số tài liệu mỗi khối khi đọc từ MongoDB
DIEM_HOP_LE_TOI_THIEU = -1e30   # implicit đệm mục bị lọc bằng điểm rất nhỏ

# Tham số mặc định của ALS
FACTORS_MAC_DINH = 64
REGULARIZATION_MAC_DINH = 0.05
ALPHA_MAC_DINH = 1.0
ITERATIONS_MAC_DINH = 20
SEED_MAC_DINH = 42


# --- Đọc dữ liệu -------------------------------------------------------------
def doc_ma_tran_tuong_tac(db, min_user_interactions):
    """Đọc ``interactions`` và dựng ma trận thưa CSR (user x item).

    Args:
        db: Đối tượng database.
        min_user_interactions: Chỉ giữ người dùng có từ ngần này sản phẩm
            khác nhau trở lên.

    Returns:
        Tuple (ma_tran, user_ids, item_ids, thong_ke): ``user_ids[i]`` là
        user_id gốc của hàng i, ``item_ids[j]`` là product_id gốc của cột j.

    Raises:
        RuntimeError: Nếu không có dữ liệu để huấn luyện.
    """
    collection = db[COL_INTERACTIONS]
    tong = collection.estimated_document_count()
    if tong == 0:
        raise RuntimeError(
            f"Collection '{COL_INTERACTIONS}' đang trống. "
            "Hãy chạy src/build_interactions.py trước."
        )

    cac_khoi = []
    khoi = []

    def dong_khoi():
        df = pd.DataFrame(khoi)
        cac_khoi.append(
            (
                df["user_id"].to_numpy(np.int64),
                df["product_id"].to_numpy(np.int64),
                df["score"].to_numpy(np.float32),
            )
        )
        khoi.clear()

    con_tro = collection.find(
        {}, {"_id": 0, "user_id": 1, "product_id": 1, "score": 1},
        batch_size=100_000,
    )
    for tai_lieu in tqdm(con_tro, total=tong, desc="Đọc interactions",
                         unit="dòng"):
        khoi.append(tai_lieu)
        if len(khoi) >= KHOI_DOC:
            dong_khoi()
    if khoi:
        dong_khoi()

    user_goc = np.concatenate([k[0] for k in cac_khoi])
    item_goc = np.concatenate([k[1] for k in cac_khoi])
    diem = np.concatenate([k[2] for k in cac_khoi])
    so_user_truoc = len(np.unique(user_goc))

    # Loại người dùng quá ít tương tác (mỗi cặp (user, product) là duy nhất)
    _, ma_user = np.unique(user_goc, return_inverse=True)
    giu = np.bincount(ma_user)[ma_user] >= min_user_interactions
    user_goc, item_goc, diem = user_goc[giu], item_goc[giu], diem[giu]
    if len(user_goc) == 0:
        raise RuntimeError(
            f"Không còn dữ liệu sau khi loại người dùng có dưới "
            f"{min_user_interactions} tương tác. Hãy giảm ngưỡng "
            "MIN_USER_INTERACTIONS."
        )

    # np.unique sắp xếp nên chỉ số ổn định giữa các lần chạy
    user_ids, hang = np.unique(user_goc, return_inverse=True)
    item_ids, cot = np.unique(item_goc, return_inverse=True)
    ma_tran = csr_matrix(
        (diem, (hang, cot)), shape=(len(user_ids), len(item_ids)),
        dtype=np.float32,
    )
    thong_ke = {
        "n_users_before_filter": int(so_user_truoc),
        "n_users": int(len(user_ids)),
        "n_items": int(len(item_ids)),
        "n_interactions": int(ma_tran.nnz),
        "density": float(ma_tran.nnz / (len(user_ids) * len(item_ids))),
        "min_user_interactions": int(min_user_interactions),
    }
    return ma_tran, user_ids, item_ids, thong_ke


def doc_thong_tin_san_pham(db, item_ids):
    """Lấy category_code, brand, giá trung bình của từng cột trong ma trận.

    Returns:
        Tuple 3 list (category_code, brand, price) cùng thứ tự với
        ``item_ids``; thiếu dữ liệu thì là None.
    """
    san_pham = pd.DataFrame(
        db[COL_PRODUCTS].find(
            {}, {"_id": 0, "product_id": 1, "category_code": 1, "brand": 1,
                 "avg_price": 1}
        )
    ).set_index("product_id")
    san_pham = san_pham.reindex(item_ids)

    def thanh_list(cot):
        return san_pham[cot].astype(object).where(
            san_pham[cot].notna(), None
        ).tolist()

    return (
        thanh_list("category_code"),
        thanh_list("brand"),
        thanh_list("avg_price"),
    )


# --- Mô hình và đánh giá -----------------------------------------------------
class MoHinhPhoBien:
    """Baseline: luôn gợi ý các sản phẩm phổ biến nhất chưa được user xem.

    Độ phổ biến của sản phẩm là tổng điểm tương tác trên tập huấn luyện. Lớp
    có phương thức ``recommend`` cùng giao diện với mô hình của implicit để
    dùng chung hàm ``ranking_metrics_at_k``, đảm bảo so sánh công bằng.
    """

    def __init__(self, train_user_items):
        """Tính độ phổ biến từ ma trận huấn luyện (user x item)."""
        self.do_pho_bien = np.asarray(train_user_items.sum(axis=0)).ravel()
        self.thu_tu = np.argsort(-self.do_pho_bien, kind="stable")

    def recommend(self, userid, user_items, N=10,
                  filter_already_liked_items=True, filter_items=None,
                  recalculate_user=False, items=None):
        """Trả về (ids, scores) giống ``implicit`` cho một lô người dùng.

        ``user_items`` là ma trận CSR chỉ gồm các hàng của lô người dùng.
        """
        user_items = csr_matrix(user_items)
        so_user = user_items.shape[0]
        ids = np.full((so_user, N), -1, dtype=np.int32)
        diem = np.full((so_user, N), -np.inf, dtype=np.float32)
        for h in range(so_user):
            da_co = user_items.indices[
                user_items.indptr[h]:user_items.indptr[h + 1]
            ]
            # Lấy dư len(da_co) ứng viên để sau khi bỏ sản phẩm đã xem vẫn đủ N
            ung_vien = self.thu_tu[: N + len(da_co)]
            if filter_already_liked_items:
                ung_vien = ung_vien[~np.isin(ung_vien, da_co)]
            ung_vien = ung_vien[:N]
            ids[h, : len(ung_vien)] = ung_vien
            diem[h, : len(ung_vien)] = self.do_pho_bien[ung_vien]
        if np.isscalar(userid):
            return ids[0], diem[0]
        return ids, diem


def huan_luyen(tham_so, seed, ma_tran):
    """Tạo và huấn luyện mô hình ALS của implicit.

    BLAS chạy 1 luồng theo khuyến nghị của implicit (ALS tự song song hoá
    bằng OpenMP); implicit kiểm tra cấu hình này ngay khi khởi tạo mô hình
    nên việc tạo và huấn luyện đều nằm trong khối giới hạn luồng.

    Args:
        tham_so: Dict factors, regularization, alpha, iterations.
        seed: Hạt giống ngẫu nhiên.
        ma_tran: Ma trận thưa CSR (user x item).

    Returns:
        Tuple (mô hình đã huấn luyện, số giây huấn luyện).
    """
    bat_dau = time.perf_counter()
    with threadpool_limits(limits=1, user_api="blas"):
        mo_hinh = AlternatingLeastSquares(random_state=seed, **tham_so)
        mo_hinh.fit(ma_tran, show_progress=True)
    return mo_hinh, time.perf_counter() - bat_dau


def lam_tron(ket_qua):
    """Làm tròn các độ đo về 6 chữ số thập phân để lưu gọn."""
    return {ten: round(float(gia_tri), 6) for ten, gia_tri in ket_qua.items()}


def danh_gia(ma_tran, tham_so, seed):
    """Đánh giá ALS so với baseline phổ biến nhất trên cùng tập test.

    Chia ngẫu nhiên 80/20 các tương tác, huấn luyện ALS trên 80% rồi đo
    precision, map, ndcg, auc @K trên 20% còn lại.

    Returns:
        Dict gồm kết quả của từng mô hình và thời gian.
    """
    train, test = train_test_split(
        ma_tran, train_percentage=TY_LE_TRAIN, random_state=seed
    )
    print(f"  Chia dữ liệu: train {train.nnz:,} | test {test.nnz:,} tương tác")

    mo_hinh, giay_train = huan_luyen(tham_so, seed, train)

    bat_dau = time.perf_counter()
    kq_als = ranking_metrics_at_k(
        mo_hinh, train, test, K=K_DANH_GIA, show_progress=True
    )
    kq_pho_bien = ranking_metrics_at_k(
        MoHinhPhoBien(train), train, test, K=K_DANH_GIA, show_progress=True
    )
    giay_danh_gia = time.perf_counter() - bat_dau
    return {
        "als": lam_tron(kq_als),
        "popularity": lam_tron(kq_pho_bien),
        "train_seconds": round(giay_train, 2),
        "eval_seconds": round(giay_danh_gia, 2),
        "n_train": int(train.nnz),
        "n_test": int(test.nnz),
    }


# --- Sinh gợi ý --------------------------------------------------------------
def _muc_san_pham(ma_cot, diem, item_ids, thong_tin):
    """Dựng dict mô tả một sản phẩm được gợi ý (kèm điểm)."""
    danh_muc, thuong_hieu, gia = thong_tin
    return {
        "product_id": item_ids[ma_cot],
        "score": round(float(diem), 4),
        "category_code": danh_muc[ma_cot],
        "brand": thuong_hieu[ma_cot],
        "price": gia[ma_cot],
    }


def sinh_goi_y(db, mo_hinh, ma_tran, user_ids, item_ids, thong_tin):
    """Sinh top-10 gợi ý cho mọi người dùng, lưu vào ``recommendations``.

    Chạy theo lô 50.000 người dùng và loại sản phẩm người dùng đã tương tác.
    Mỗi tài liệu có dạng::

        {user_id, recommendations: [{product_id, score, category_code,
                                     brand, price}, ...]}

    Returns:
        Số người dùng đã có gợi ý.
    """
    collection = db[COL_RECOMMENDATIONS]
    collection.drop()
    ds_user = user_ids.tolist()
    ds_item = item_ids.tolist()
    da_ghi = 0
    for dau in tqdm(range(0, len(ds_user), BATCH_NGUOI_DUNG),
                    desc="Sinh gợi ý", unit="lô"):
        cuoi = min(dau + BATCH_NGUOI_DUNG, len(ds_user))
        ids, diem = mo_hinh.recommend(
            np.arange(dau, cuoi), ma_tran[dau:cuoi], N=SO_GOI_Y,
            filter_already_liked_items=True,
        )
        tai_lieu = []
        for h in range(cuoi - dau):
            goi_y = [
                _muc_san_pham(int(ids[h, j]), diem[h, j], ds_item, thong_tin)
                for j in range(SO_GOI_Y)
                if diem[h, j] > DIEM_HOP_LE_TOI_THIEU
            ]
            tai_lieu.append(
                {"user_id": ds_user[dau + h], "recommendations": goi_y}
            )
        collection.insert_many(tai_lieu, ordered=False)
        da_ghi += len(tai_lieu)
    collection.create_index("user_id", unique=True)
    return da_ghi


def sinh_san_pham_tuong_tu(db, mo_hinh, ma_tran, item_ids, thong_tin):
    """Sinh top-10 sản phẩm tương tự, lưu vào ``similar_items``.

    Chỉ làm cho sản phẩm có từ ``TUONG_TAC_TOI_THIEU_SP`` tương tác trở lên
    (ít tương tác thì vector nhân tố không đáng tin). Mỗi tài liệu::

        {product_id, similar: [{product_id, score, category_code, brand,
                                price}, ...]}

    Returns:
        Số sản phẩm đã có danh sách tương tự.
    """
    collection = db[COL_SIMILAR_ITEMS]
    collection.drop()
    ds_item = item_ids.tolist()
    so_tuong_tac = np.bincount(ma_tran.indices, minlength=len(ds_item))
    du_dieu_kien = np.flatnonzero(so_tuong_tac >= TUONG_TAC_TOI_THIEU_SP)
    so_ung_vien = min(SO_TUONG_TU + 1, len(ds_item))  # +1: bao gồm chính nó

    da_ghi = 0
    for dau in tqdm(range(0, len(du_dieu_kien), BATCH_SAN_PHAM),
                    desc="Sản phẩm tương tự", unit="lô"):
        lo = du_dieu_kien[dau:dau + BATCH_SAN_PHAM]
        ids, diem = mo_hinh.similar_items(lo, N=so_ung_vien)
        tai_lieu = []
        for h, goc in enumerate(lo.tolist()):
            tuong_tu = [
                _muc_san_pham(int(ids[h, j]), diem[h, j], ds_item, thong_tin)
                for j in range(so_ung_vien)
                if int(ids[h, j]) != goc  # bỏ chính sản phẩm đó
            ][:SO_TUONG_TU]
            tai_lieu.append({"product_id": ds_item[goc], "similar": tuong_tu})
        if tai_lieu:
            collection.insert_many(tai_lieu, ordered=False)
            da_ghi += len(tai_lieu)
    collection.create_index("product_id", unique=True)
    return da_ghi


def luu_san_pham_pho_bien(db):
    """Lưu top-50 sản phẩm theo tổng điểm vào ``popular_products``.

    Dùng cho người dùng mới chưa có lịch sử (cold-start). Tính trên toàn bộ
    bảng ``interactions`` bằng aggregation pipeline.

    Returns:
        Số sản phẩm đã lưu.
    """
    ket_qua = db[COL_INTERACTIONS].aggregate(
        [
            {
                "$group": {
                    "_id": "$product_id",
                    "score": {"$sum": "$score"},
                    "purchases": {"$sum": "$purchases"},
                }
            },
            {"$sort": {"score": -1, "_id": 1}},
            {"$limit": SO_SAN_PHAM_PHO_BIEN},
            {
                "$lookup": {
                    "from": COL_PRODUCTS,
                    "localField": "_id",
                    "foreignField": "product_id",
                    "as": "thong_tin",
                }
            },
            {
                "$unwind": {
                    "path": "$thong_tin",
                    "preserveNullAndEmptyArrays": True,
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "product_id": "$_id",
                    "score": 1,
                    "purchases": 1,
                    "category_code": "$thong_tin.category_code",
                    "brand": "$thong_tin.brand",
                    "price": "$thong_tin.avg_price",
                }
            },
            {"$sort": {"score": -1, "product_id": 1}},
        ],
        allowDiskUse=True,
    )
    tai_lieu = [
        {"rank": hang, **dong} for hang, dong in enumerate(ket_qua, start=1)
    ]
    for dong in tai_lieu:  # trường vắng mặt -> null cho đồng nhất schema
        for truong in ("category_code", "brand", "price"):
            dong.setdefault(truong, None)
    collection = db[COL_POPULAR_PRODUCTS]
    collection.drop()
    if tai_lieu:
        collection.insert_many(tai_lieu)
    collection.create_index("rank", unique=True)
    return len(tai_lieu)


def luu_mo_hinh(mo_hinh, user_ids, item_ids):
    """Lưu mô hình và bảng ánh xạ id vào ``models/`` (đã .gitignore)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    duong_dan = MODEL_DIR / "als_model.npz"
    mo_hinh.save(str(duong_dan))
    np.savez_compressed(
        MODEL_DIR / "id_map.npz", user_ids=user_ids, item_ids=item_ids
    )
    return duong_dan


# --- Điều phối ---------------------------------------------------------------
def kiem_tra_tham_so(factors, regularization, alpha, iterations):
    """Kiểm tra tham số ALS hợp lệ.

    Raises:
        ValueError: Nếu có tham số ngoài miền cho phép.
    """
    if factors < 1 or iterations < 1:
        raise ValueError("factors và iterations phải >= 1")
    if regularization < 0:
        raise ValueError("regularization phải >= 0")
    if alpha <= 0:
        raise ValueError("alpha phải > 0")


def run(
    db=None,
    factors=FACTORS_MAC_DINH,
    regularization=REGULARIZATION_MAC_DINH,
    alpha=ALPHA_MAC_DINH,
    iterations=ITERATIONS_MAC_DINH,
    min_user_interactions=None,
    seed=SEED_MAC_DINH,
    luu_file_mo_hinh=True,
):
    """Chạy toàn bộ quy trình huấn luyện, đánh giá và sinh gợi ý.

    Returns:
        Tài liệu kết quả đã lưu vào ``model_metrics``.

    Raises:
        ValueError: Nếu tham số không hợp lệ.
        RuntimeError: Nếu chưa có dữ liệu ``interactions``.
    """
    kiem_tra_tham_so(factors, regularization, alpha, iterations)
    db = db if db is not None else get_db_hoac_thoat()
    nguong = min_user_interactions or MIN_USER_INTERACTIONS
    tham_so = {
        "factors": factors,
        "regularization": regularization,
        "alpha": alpha,
        "iterations": iterations,
    }

    print("[1/6] Đọc dữ liệu và dựng ma trận thưa CSR ...")
    ma_tran, user_ids, item_ids, thong_ke = doc_ma_tran_tuong_tac(db, nguong)
    print(f"  Ma trận {ma_tran.shape[0]:,} user x {ma_tran.shape[1]:,} "
          f"sản phẩm, {ma_tran.nnz:,} tương tác "
          f"(mật độ {thong_ke['density']:.4%}); "
          f"giữ {thong_ke['n_users']:,}/{thong_ke['n_users_before_filter']:,}"
          f" người dùng.")

    print(f"[2/6] Đánh giá (train/test {TY_LE_TRAIN:.0%}/"
          f"{1 - TY_LE_TRAIN:.0%}, K={K_DANH_GIA}) ...")
    ket_qua = danh_gia(ma_tran, tham_so, seed)

    print("[3/6] Huấn luyện lại ALS trên toàn bộ dữ liệu ...")
    mo_hinh, giay_train_cuoi = huan_luyen(tham_so, seed, ma_tran)

    tai_lieu_do = {
        "created_at": datetime.now(timezone.utc),
        "k": K_DANH_GIA,
        "model": "implicit.als.AlternatingLeastSquares",
        "params": {
            **tham_so,
            "random_state": seed,
            "train_percentage": TY_LE_TRAIN,
            "min_user_interactions": nguong,
        },
        "dataset": {
            **thong_ke,
            "n_train": ket_qua["n_train"],
            "n_test": ket_qua["n_test"],
        },
        "results": {
            "als": ket_qua["als"],
            "popularity": ket_qua["popularity"],
        },
        "train_seconds": {
            "evaluation_fit": ket_qua["train_seconds"],
            "final_fit": round(giay_train_cuoi, 2),
        },
        "eval_seconds": ket_qua["eval_seconds"],
    }
    db[COL_MODEL_METRICS].insert_one(tai_lieu_do)

    thong_tin = doc_thong_tin_san_pham(db, item_ids)
    print("[4/6] Sinh top-10 gợi ý cho mọi người dùng ...")
    so_user = sinh_goi_y(db, mo_hinh, ma_tran, user_ids, item_ids, thong_tin)
    print("[5/6] Sinh top-10 sản phẩm tương tự ...")
    so_sp = sinh_san_pham_tuong_tu(db, mo_hinh, ma_tran, item_ids, thong_tin)
    print("[6/6] Lưu sản phẩm phổ biến cho người dùng mới (cold-start) ...")
    so_pho_bien = luu_san_pham_pho_bien(db)
    if luu_file_mo_hinh:
        print(f"  Đã lưu mô hình: {luu_mo_hinh(mo_hinh, user_ids, item_ids)}")

    print(f"\nĐã lưu: {so_user:,} user có gợi ý | {so_sp:,} sản phẩm có "
          f"danh sách tương tự | {so_pho_bien} sản phẩm phổ biến.")
    in_bang_so_sanh(tai_lieu_do)
    return tai_lieu_do


def in_bang_so_sanh(tai_lieu_do):
    """In bảng so sánh ALS với baseline phổ biến nhất."""
    als = tai_lieu_do["results"]["als"]
    pho_bien = tai_lieu_do["results"]["popularity"]
    print(f"\nĐộ đo @{tai_lieu_do['k']} trên tập test:")
    print(f"  {'Độ đo':<10}{'ALS':>12}{'Phổ biến nhất':>16}{'Tăng':>10}")
    for ten in ("precision", "map", "ndcg", "auc"):
        a, b = als[ten], pho_bien[ten]
        tang = f"x{a / b:.1f}" if b > 0 else "n/a"
        print(f"  {ten:<10}{a:>12.4f}{b:>16.4f}{tang:>10}")
    giay = tai_lieu_do["train_seconds"]
    print(f"  Thời gian huấn luyện: {giay['evaluation_fit']}s (đánh giá), "
          f"{giay['final_fit']}s (toàn bộ dữ liệu)")


def main(argv=None):
    """Điểm vào dòng lệnh."""
    thiet_lap_console()
    parser = argparse.ArgumentParser(
        description="Huấn luyện ALS, đánh giá và sinh gợi ý vào MongoDB."
    )
    parser.add_argument("--factors", type=int, default=FACTORS_MAC_DINH,
                        help="số nhân tố ẩn (mặc định %(default)s)")
    parser.add_argument("--regularization", type=float,
                        default=REGULARIZATION_MAC_DINH,
                        help="hệ số chính quy hoá (mặc định %(default)s)")
    parser.add_argument("--alpha", type=float, default=ALPHA_MAC_DINH,
                        help="trọng số độ tin cậy (mặc định %(default)s)")
    parser.add_argument("--iterations", type=int, default=ITERATIONS_MAC_DINH,
                        help="số vòng lặp ALS (mặc định %(default)s)")
    parser.add_argument("--min-user-interactions", type=int, default=None,
                        help="số sản phẩm tối thiểu của một người dùng "
                             f"(mặc định: {MIN_USER_INTERACTIONS})")
    parser.add_argument("--seed", type=int, default=SEED_MAC_DINH,
                        help="hạt giống ngẫu nhiên (mặc định %(default)s)")
    parser.add_argument("--no-save-model", action="store_true",
                        help="không lưu file mô hình vào thư mục models/")
    args = parser.parse_args(argv)

    try:
        run(
            factors=args.factors,
            regularization=args.regularization,
            alpha=args.alpha,
            iterations=args.iterations,
            min_user_interactions=args.min_user_interactions,
            seed=args.seed,
            luu_file_mo_hinh=not args.no_save_model,
        )
    except (ValueError, RuntimeError) as loi:
        print(f"LỖI: {loi}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
