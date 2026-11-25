"""Sinh file CSV giả lập nhỏ để chạy thử khi chưa có dữ liệu thật.

File sinh ra có cùng schema với bộ dữ liệu REES46 "eCommerce behavior data
from multi category store":

    event_time, event_type, product_id, category_id, category_code,
    brand, price, user_id, user_session

Dữ liệu giả lập vẫn có cấu trúc để mô hình gợi ý học được:
  * mỗi người dùng có 2-3 danh mục ưa thích;
  * độ phổ biến của sản phẩm trong danh mục theo phân phối Zipf;
  * hành vi theo phiên: xem -> (thêm giỏ) -> (mua), giờ cao điểm buổi chiều;
  * số lượng view > cart > purchase như dữ liệu thực tế.

Lưu ý: tỷ lệ chuyển đổi trong dữ liệu mẫu cao hơn dữ liệu thật (thật khoảng
96% view, 2% cart, 2% purchase) để file nhỏ vẫn có đủ lượt mua cho huấn luyện.

Cách dùng:
    python scripts/generate_sample.py
    python scripts/generate_sample.py --rows 500000 --users 10000 --seed 7
"""

import argparse
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

GOC_DU_AN = Path(__file__).resolve().parent.parent
FILE_MAC_DINH = GOC_DU_AN / "data" / "raw" / "sample_events.csv"

# (category_code, giá trung vị) - định dạng giống bộ dữ liệu REES46
DANH_MUC = [
    ("electronics.smartphone", 320),
    ("electronics.audio.headphone", 60),
    ("electronics.video.tv", 380),
    ("electronics.clocks", 90),
    ("electronics.tablet", 250),
    ("computers.notebook", 650),
    ("computers.desktop", 480),
    ("computers.peripherals.mouse", 18),
    ("computers.peripherals.keyboard", 35),
    ("computers.components.videocards", 320),
    ("computers.components.cpu", 210),
    ("appliances.kitchen.refrigerators", 420),
    ("appliances.kitchen.washer", 340),
    ("appliances.kitchen.microwave", 80),
    ("appliances.kitchen.kettle", 25),
    ("appliances.kitchen.hood", 120),
    ("appliances.environment.vacuum", 110),
    ("appliances.environment.air_conditioner", 450),
    ("appliances.personal.hair_cutter", 22),
    ("furniture.living_room.sofa", 380),
    ("furniture.bedroom.bed", 260),
    ("furniture.kitchen.table", 140),
    ("apparel.shoes", 55),
    ("apparel.shoes.sneakers", 70),
    ("apparel.jeans", 40),
    ("apparel.tshirt", 15),
    ("accessories.bag", 45),
    ("accessories.umbrella", 12),
    ("auto.accessories.player", 70),
    ("auto.accessories.radar", 55),
    ("sport.bicycle", 230),
    ("sport.tennis", 38),
    ("kids.toys", 20),
    ("kids.carriage", 160),
    ("kids.fmcg.diapers", 18),
    ("construction.tools.drill", 65),
    ("medicine.tools.tonometer", 30),
    ("country_yard.lawn_mower", 150),
    ("stationery.cartrige", 24),
]

THUONG_HIEU = [
    "samsung", "apple", "xiaomi", "huawei", "lg", "sony", "philips", "bosch",
    "asus", "acer", "lenovo", "hp", "dell", "msi", "gigabyte", "intel", "amd",
    "nvidia", "logitech", "razer", "jbl", "sennheiser", "bose", "canon",
    "nikon", "panasonic", "sharp", "toshiba", "hitachi", "tefal", "braun",
    "dyson", "karcher", "makita", "stihl", "bork", "redmond", "vitek",
    "polaris", "indesit", "whirlpool", "electrolux", "beko", "haier",
    "hisense", "tcl", "nike", "adidas", "puma", "reebok", "levis", "zara",
    "fossil", "casio", "garmin", "lego", "fisher-price", "huggies", "pampers",
    "oppo",
]

# Tỷ lệ thiếu dữ liệu ở cấp sản phẩm (dữ liệu thật cũng bị thiếu như vậy)
TY_LE_THIEU_DANH_MUC = 0.12
TY_LE_THIEU_THUONG_HIEU = 0.12

# Hành vi trong phiên
XAC_SUAT_THEM_GIO = 0.075       # sau khi xem
XAC_SUAT_MUA_SAU_GIO = 0.55     # sau khi thêm giỏ
XAC_SUAT_MUA_TRUC_TIEP = 0.010  # mua ngay không qua giỏ
XAC_SUAT_DANH_MUC_UA_THICH = 0.80

SO_NGAY = 31  # tháng 10 năm 2019
GIAY_MOT_NGAY = 86_400
TRONG_SO_GIO = np.array(
    [2, 1, 1, 1, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 13, 12, 10, 8, 6,
     5, 4, 3],
    dtype=float,
)


def tao_san_pham(rng, so_san_pham):
    """Tạo bảng sản phẩm (product_id, danh mục, thương hiệu, giá, độ hot).

    Mỗi sản phẩm có thuộc tính cố định; một phần sản phẩm bị thiếu
    category_code hoặc brand giống dữ liệu thật.
    """
    so_dm = len(DANH_MUC)
    ma_sp = np.unique(rng.integers(1_000_000, 100_000_000, so_san_pham * 2))
    ma_sp = rng.permutation(ma_sp)[:so_san_pham]

    # Danh mục lớn nhỏ khác nhau; đảm bảo mỗi danh mục có ít nhất 5 sản phẩm
    trong_so_dm = 1.0 / np.arange(1, so_dm + 1) ** 0.6
    trong_so_dm = rng.permutation(trong_so_dm / trong_so_dm.sum())
    chi_so_dm = rng.choice(so_dm, size=so_san_pham, p=trong_so_dm)
    toi_thieu = np.repeat(np.arange(so_dm), 5)
    chi_so_dm[: len(toi_thieu)] = toi_thieu

    ma_dm = rng.integers(
        2_000_000_000_000_000_000, 2_300_000_000_000_000_000, so_dm
    )
    gia_trung_vi = np.array([gia for _, gia in DANH_MUC], dtype=float)
    gia = gia_trung_vi[chi_so_dm] * rng.lognormal(0.0, 0.45, so_san_pham)
    gia = np.round(np.clip(gia, 0.5, None), 2)

    # Mỗi danh mục chỉ có một nhóm thương hiệu riêng (6-14 thương hiệu)
    nhom_th = [
        rng.choice(len(THUONG_HIEU), size=rng.integers(6, 15), replace=False)
        for _ in range(so_dm)
    ]
    chi_so_th = np.array([rng.choice(nhom_th[c]) for c in chi_so_dm])

    ten_dm = np.array([ma for ma, _ in DANH_MUC], dtype=object)[chi_so_dm]
    ten_th = np.array(THUONG_HIEU, dtype=object)[chi_so_th]
    ten_dm[rng.random(so_san_pham) < TY_LE_THIEU_DANH_MUC] = None
    ten_th[rng.random(so_san_pham) < TY_LE_THIEU_THUONG_HIEU] = None

    # Độ hot theo Zipf trong từng danh mục
    do_hot = np.zeros(so_san_pham)
    for c in range(so_dm):
        vi_tri = np.flatnonzero(chi_so_dm == c)
        hang = rng.permutation(len(vi_tri)) + 1
        do_hot[vi_tri] = 1.0 / (hang + 5) ** 0.9

    return pd.DataFrame(
        {
            "product_id": ma_sp,
            "category_id": ma_dm[chi_so_dm],
            "category_code": ten_dm,
            "brand": ten_th,
            "price": gia,
            "danh_muc": chi_so_dm,
            "do_hot": do_hot,
        }
    )


def tao_nguoi_dung(rng, so_nguoi_dung):
    """Tạo người dùng: mã, danh mục ưa thích, trọng số và mức độ hoạt động."""
    so_dm = len(DANH_MUC)
    ma = np.unique(rng.integers(100_000_000, 600_000_000, so_nguoi_dung * 2))
    ma = rng.permutation(ma)[:so_nguoi_dung]

    ua_thich = np.empty((so_nguoi_dung, 3), dtype=int)
    for i in range(so_nguoi_dung):
        ua_thich[i] = rng.choice(so_dm, size=3, replace=False)
    # Trọng số cộng dồn của 3 danh mục ưa thích (chuẩn hoá về tổng = 1)
    trong_so = rng.dirichlet([3.0, 1.5, 1.0], size=so_nguoi_dung)
    hoat_dong = rng.lognormal(0.0, 0.9, so_nguoi_dung)
    hoat_dong = hoat_dong / hoat_dong.sum()
    return ma, ua_thich, np.cumsum(trong_so, axis=1), hoat_dong


def tao_cac_phien(rng, sp, nguoi_dung, so_phien, phien_dau):
    """Sinh các sự kiện của ``so_phien`` phiên mua sắm.

    Args:
        rng: Bộ sinh số ngẫu nhiên numpy.
        sp: Bảng sản phẩm từ :func:`tao_san_pham`.
        nguoi_dung: Bộ giá trị từ :func:`tao_nguoi_dung`.
        so_phien: Số phiên cần sinh.
        phien_dau: Nếu True, phiên i (i < số người dùng) thuộc người dùng i
            để mọi người dùng đều xuất hiện ít nhất một lần.

    Returns:
        DataFrame các sự kiện, theo thứ tự phiên (chưa sắp xếp theo giờ).
    """
    ma_nd, ua_thich, cong_don, hoat_dong = nguoi_dung
    so_nd = len(ma_nd)
    so_dm = len(DANH_MUC)

    # --- Phiên thuộc về người dùng nào -----------------------------------
    nd_phien = rng.choice(so_nd, size=so_phien, p=hoat_dong)
    if phien_dau:
        so_dau = min(so_nd, so_phien)
        nd_phien[:so_dau] = rng.permutation(so_nd)[:so_dau]

    # --- Thời điểm bắt đầu phiên ----------------------------------------
    ngay = rng.integers(0, SO_NGAY, so_phien)
    gio = rng.choice(24, size=so_phien, p=TRONG_SO_GIO / TRONG_SO_GIO.sum())
    bat_dau = (
        ngay * GIAY_MOT_NGAY + gio * 3600 + rng.integers(0, 3600, so_phien)
    )

    # --- Các lượt xem trong phiên ----------------------------------------
    so_xem = 1 + rng.poisson(3.0, so_phien)
    phien_cua_xem = np.repeat(np.arange(so_phien), so_xem)
    tong_xem = len(phien_cua_xem)
    nd_cua_xem = nd_phien[phien_cua_xem]

    # Chọn danh mục: 80% từ danh mục ưa thích, còn lại ngẫu nhiên toàn sàn
    u = rng.random(tong_xem)
    vi_tri_ua_thich = (u[:, None] > cong_don[nd_cua_xem]).sum(axis=1)
    vi_tri_ua_thich = np.minimum(vi_tri_ua_thich, 2)
    dm_ua_thich = ua_thich[nd_cua_xem, vi_tri_ua_thich]
    dm_ngau_nhien = rng.integers(0, so_dm, tong_xem)
    dung_ua_thich = rng.random(tong_xem) < XAC_SUAT_DANH_MUC_UA_THICH
    dm_xem = np.where(dung_ua_thich, dm_ua_thich, dm_ngau_nhien)

    # Chọn sản phẩm trong danh mục theo độ hot
    dm_cua_sp = sp["danh_muc"].to_numpy()
    do_hot = sp["do_hot"].to_numpy()
    chi_so_sp = np.empty(tong_xem, dtype=int)
    for c in range(so_dm):
        vi_tri_sp = np.flatnonzero(dm_cua_sp == c)
        vi_tri_xem = np.flatnonzero(dm_xem == c)
        if len(vi_tri_xem) == 0:
            continue
        p = do_hot[vi_tri_sp]
        chi_so_sp[vi_tri_xem] = rng.choice(
            vi_tri_sp, size=len(vi_tri_xem), p=p / p.sum()
        )

    # Thời điểm xem: cộng dồn khoảng cách ngẫu nhiên (~60 giây) trong phiên
    khoang_cach = rng.exponential(60.0, tong_xem).astype(int) + 1
    dau_phien = np.concatenate(([0], np.cumsum(so_xem)[:-1]))
    truoc_do = np.cumsum(khoang_cach) - khoang_cach  # tổng khoảng cách trước
    giay_xem = (
        bat_dau[phien_cua_xem]
        + truoc_do
        - np.repeat(truoc_do[dau_phien], so_xem)
    )

    # --- Thêm giỏ và mua ----------------------------------------------------
    them_gio = rng.random(tong_xem) < XAC_SUAT_THEM_GIO
    mua_sau_gio = them_gio & (rng.random(tong_xem) < XAC_SUAT_MUA_SAU_GIO)
    mua_truc_tiep = ~them_gio & (rng.random(tong_xem) < XAC_SUAT_MUA_TRUC_TIEP)

    giay_gio = giay_xem + rng.integers(10, 300, tong_xem)
    giay_mua = np.where(
        them_gio, giay_gio, giay_xem
    ) + rng.integers(30, 600, tong_xem)

    def tao_nhom(loai, mat_na, giay):
        """Tạo mảng sự kiện cho các lượt xem thoả mặt nạ."""
        return pd.DataFrame(
            {
                "event_type": loai,
                "sp": chi_so_sp[mat_na],
                "nd": nd_cua_xem[mat_na],
                "phien": phien_cua_xem[mat_na],
                "giay": giay[mat_na],
            }
        )

    tat_ca = np.ones(tong_xem, dtype=bool)
    su_kien = pd.concat(
        [
            tao_nhom("view", tat_ca, giay_xem),
            tao_nhom("cart", them_gio, giay_gio),
            tao_nhom("purchase", mua_sau_gio | mua_truc_tiep, giay_mua),
        ],
        ignore_index=True,
    )
    # Giữ thứ tự phiên để có thể cắt đuôi khi thừa dòng
    su_kien = su_kien.sort_values(["phien", "giay"], kind="stable")

    ma_phien = np.array(
        [
            str(uuid.UUID(bytes=rng.bytes(16), version=4))
            for _ in range(so_phien)
        ]
    )
    san_pham = sp.iloc[su_kien["sp"].to_numpy()]
    return pd.DataFrame(
        {
            "giay": su_kien["giay"].to_numpy(),
            "event_type": su_kien["event_type"].to_numpy(),
            "product_id": san_pham["product_id"].to_numpy(),
            "category_id": san_pham["category_id"].to_numpy(),
            "category_code": san_pham["category_code"].to_numpy(),
            "brand": san_pham["brand"].to_numpy(),
            "price": san_pham["price"].to_numpy(),
            "user_id": ma_nd[su_kien["nd"].to_numpy()],
            "user_session": ma_phien[su_kien["phien"].to_numpy()],
        }
    )


def sinh_du_lieu(so_dong, so_nguoi_dung, so_san_pham, seed):
    """Sinh toàn bộ bảng sự kiện với đúng ``so_dong`` dòng.

    Returns:
        DataFrame đã sắp xếp theo thời gian, cột đúng thứ tự của file REES46.
    """
    rng = np.random.default_rng(seed)
    sp = tao_san_pham(rng, so_san_pham)
    nguoi_dung = tao_nguoi_dung(rng, so_nguoi_dung)

    cac_phan = []
    da_co = 0
    lan_dau = True
    while da_co < so_dong:
        # Mỗi phiên trung bình ~4.5 sự kiện; sinh dư 10% rồi cắt bớt
        so_phien = max(int((so_dong - da_co) / 4.5 * 1.1), so_nguoi_dung // 10)
        if lan_dau:
            so_phien = max(so_phien, so_nguoi_dung)
        phan = tao_cac_phien(rng, sp, nguoi_dung, so_phien, lan_dau)
        cac_phan.append(phan)
        da_co += len(phan)
        lan_dau = False
    du_lieu = pd.concat(cac_phan, ignore_index=True).iloc[:so_dong].copy()

    du_lieu = du_lieu.sort_values("giay", kind="stable", ignore_index=True)
    giay_toi_da = SO_NGAY * GIAY_MOT_NGAY - 1
    giay = np.clip(du_lieu.pop("giay").to_numpy(), 0, giay_toi_da)
    moc = np.datetime64("2019-10-01T00:00:00")
    thoi_gian = pd.Series(moc + giay.astype("timedelta64[s]"))
    du_lieu.insert(
        0, "event_time", thoi_gian.dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    )
    return du_lieu


def main():
    """Điểm vào dòng lệnh."""
    for luong in (sys.stdout, sys.stderr):
        if hasattr(luong, "reconfigure"):
            luong.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Sinh file CSV giả lập cùng schema với bộ dữ liệu REES46."
    )
    parser.add_argument("--rows", type=int, default=200_000,
                        help="số dòng cần sinh (mặc định 200000)")
    parser.add_argument("--users", type=int, default=5_000,
                        help="số người dùng (mặc định 5000)")
    parser.add_argument("--products", type=int, default=2_000,
                        help="số sản phẩm (mặc định 2000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="hạt giống ngẫu nhiên để tái lập kết quả")
    parser.add_argument("--output", type=Path, default=FILE_MAC_DINH,
                        help="đường dẫn file CSV đầu ra")
    args = parser.parse_args()

    toi_thieu_sp = len(DANH_MUC) * 5
    if args.users < 1 or args.products < toi_thieu_sp:
        parser.error(f"--users phải >= 1 và --products phải >= {toi_thieu_sp}")
    if args.rows < args.users * 2:
        parser.error("--rows nên lớn hơn ít nhất 2 lần --users")

    bat_dau = time.perf_counter()
    du_lieu = sinh_du_lieu(args.rows, args.users, args.products, args.seed)

    dau_ra = args.output
    dau_ra.parent.mkdir(parents=True, exist_ok=True)
    du_lieu.to_csv(dau_ra, index=False)

    dem = du_lieu["event_type"].value_counts()
    tong = len(du_lieu)
    print(f"Đã ghi {tong:,} dòng vào: {dau_ra}")
    print(f"  Dung lượng        : {dau_ra.stat().st_size / 1e6:.1f} MB")
    print(f"  Người dùng        : {du_lieu['user_id'].nunique():,}")
    print(f"  Sản phẩm          : {du_lieu['product_id'].nunique():,}")
    print(f"  Danh mục          : {du_lieu['category_code'].nunique():,}")
    print(f"  Thương hiệu       : {du_lieu['brand'].nunique():,}")
    print(f"  Phiên             : {du_lieu['user_session'].nunique():,}")
    for loai in ("view", "cart", "purchase"):
        print(f"  {loai:<9}         : {dem.get(loai, 0):>8,} "
              f"({dem.get(loai, 0) / tong:.1%})")
    print(f"  Thời gian chạy    : {time.perf_counter() - bat_dau:.1f} giây")


if __name__ == "__main__":
    main()
