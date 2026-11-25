"""Dashboard Streamlit: phân tích thương mại điện tử và hệ thống gợi ý.

Bốn trang trên sidebar:
  1. Tổng quan & phân tích   (đọc các collection ``stats_*``)
  2. Gợi ý cho người dùng    (``recommendations``, ``popular_products``)
  3. Sản phẩm tương tự       (``similar_items``, ``products``)
  4. Đánh giá mô hình        (``model_metrics``)

Chạy:  streamlit run app/app.py
"""

import sys
from pathlib import Path

if str(Path(__file__).resolve().parent.parent) not in sys.path:
    # Để import được gói src khi chạy bằng "streamlit run app/app.py"
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    COL_EVENTS,
    COL_MODEL_METRICS,
    COL_POPULAR_PRODUCTS,
    COL_PRODUCTS,
    COL_RECOMMENDATIONS,
    COL_SIMILAR_ITEMS,
    COL_STATS_BY_HOUR,
    COL_STATS_BY_WEEKDAY,
    COL_STATS_FUNNEL,
    COL_STATS_OVERVIEW,
    COL_STATS_RFM_SUMMARY,
    COL_STATS_TOP_BRANDS,
    COL_STATS_TOP_CATEGORIES,
    COL_STATS_TOP_PRODUCTS,
    DB_NAME,
    MIN_USER_INTERACTIONS,
)
from src.db import MongoConnectionError, get_client

st.set_page_config(
    page_title="Phân tích thương mại điện tử & gợi ý sản phẩm",
    page_icon="🛒",
    layout="wide",
)

# --- Bảng màu ----------------------------------------------------------------
# Bảng màu dữ liệu đã kiểm tra bằng validate_palette (sáng và tối); phễu dùng
# dải ordinal một tông xanh; baseline dùng xám trung tính. Màu chữ, đường lưới
# và trục do theme của Streamlit đảm nhiệm nên luôn khớp giao diện sáng/tối.
BANG_MAU = {
    "light": {
        "chinh": "#257de0",      # slot 1 - số lượng
        "doanh_thu": "#f47230",  # slot 2 - doanh thu
        "nhom": ["#257de0", "#f47230", "#00b894", "#f2a600", "#ee7fac"],
        "pheu": ["#84b9f3", "#338ced", "#1757ac"],
        "xam": "#a3a3a3",
    },
    "dark": {
        "chinh": "#338ced",
        "doanh_thu": "#e76624",
        "nhom": ["#338ced", "#e76624", "#00a888", "#d28e05", "#e24d8b"],
        "pheu": ["#84b9f3", "#338ced", "#1757ac"],
        "xam": "#7a7a7a",
    },
}

# Màu gắn cố định với từng nhóm khách hàng (màu theo thực thể, không theo hạng)
NHOM_RFM = [
    "Khách VIP",
    "Khách trung thành",
    "Khách tiềm năng",
    "Khách cần chăm sóc",
    "Khách sắp rời bỏ",
]

NHAN_SU_KIEN = {"view": "Xem", "cart": "Thêm giỏ", "purchase": "Mua"}
CHU_SO_DAU_THUONG = "system-ui, -apple-system, 'Segoe UI', sans-serif"
THOI_GIAN_CACHE = 300  # giây


def bang_mau():
    """Chọn bảng màu dữ liệu sáng/tối theo giao diện của Streamlit.

    Ưu tiên theme ép bằng cấu hình (``theme.base``); nếu không có thì theo
    ``st.context.theme`` (phản ánh ưu tiên màu của trình duyệt/hệ điều hành).
    """
    che_do = st.get_option("theme.base")
    if che_do not in ("light", "dark"):
        try:
            che_do = st.context.theme.type
        except Exception:  # bản Streamlit cũ chưa có st.context.theme
            che_do = "light"
    return BANG_MAU["dark" if che_do == "dark" else "light"]


def dinh_dang_so(gia_tri, so_le=0):
    """Định dạng số theo kiểu Việt Nam: dấu chấm hàng nghìn, phẩy thập phân."""
    if gia_tri is None or pd.isna(gia_tri):
        return "—"
    chuoi = f"{gia_tri:,.{so_le}f}"
    return chuoi.replace(",", "\0").replace(".", ",").replace("\0", ".")


# --- Truy cập MongoDB (có cache) ---------------------------------------------
@st.cache_resource(show_spinner=False)
def ket_noi_mongo():
    """Tạo kết nối MongoDB dùng chung cho mọi phiên làm việc."""
    return get_client()


def lay_db():
    """Trả về database của dự án."""
    return ket_noi_mongo()[DB_NAME]


@st.cache_data(ttl=THOI_GIAN_CACHE, show_spinner=False)
def truy_van(ten_collection, bo_loc=None, sap_xep=None, gioi_han=0):
    """Đọc tài liệu từ một collection (không lấy ``_id``).

    Args:
        ten_collection: Tên collection.
        bo_loc: Điều kiện lọc (dict) hoặc None.
        sap_xep: Danh sách (trường, chiều) hoặc None.
        gioi_han: Số tài liệu tối đa (0 = không giới hạn).

    Returns:
        List các dict.
    """
    con_tro = lay_db()[ten_collection].find(bo_loc or {}, {"_id": 0})
    if sap_xep:
        con_tro = con_tro.sort(sap_xep)
    if gioi_han:
        con_tro = con_tro.limit(gioi_han)
    return list(con_tro)


def lay_mot(ten_collection, bo_loc=None, sap_xep=None):
    """Đọc một tài liệu (hoặc None nếu không có)."""
    ket_qua = truy_van(ten_collection, bo_loc, sap_xep, 1)
    return ket_qua[0] if ket_qua else None


@st.cache_data(ttl=THOI_GIAN_CACHE, show_spinner=False)
def mau_ngau_nhien(ten_collection, truong, so_luong=30):
    """Lấy ngẫu nhiên vài giá trị của một trường để người dùng chọn nhanh."""
    ket_qua = lay_db()[ten_collection].aggregate(
        [{"$sample": {"size": so_luong}}, {"$project": {"_id": 0, truong: 1}}]
    )
    return sorted(dong[truong] for dong in ket_qua)


@st.cache_data(ttl=THOI_GIAN_CACHE, show_spinner=False)
def dem_su_kien_user(user_id):
    """Đếm số sự kiện của một người dùng (dùng index user_id)."""
    return lay_db()[COL_EVENTS].count_documents({"user_id": user_id})


def hien_chua_co_du_lieu(lenh):
    """Thông báo khi collection cần thiết chưa có dữ liệu."""
    st.warning(
        "Chưa có dữ liệu cho mục này. Hãy chạy pipeline trước:\n\n"
        f"`{lenh}`"
    )


# --- Định dạng biểu đồ -------------------------------------------------------
def dinh_dang_hinh(hinh, mau, chieu_cao=360, chu_thich=False):
    """Áp dụng kiểu chung; màu chữ, lưới và trục lấy từ theme Streamlit."""
    hinh.update_layout(
        height=chieu_cao,
        margin=dict(l=8, r=16, t=24 if not chu_thich else 48, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=CHU_SO_DAU_THUONG, size=13),
        separators=",.",
        showlegend=chu_thich,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0),
        hoverlabel=dict(font=dict(family=CHU_SO_DAU_THUONG)),
    )
    # automargin: tự nới lề để nhãn trục không bị cắt
    hinh.update_xaxes(zeroline=False, automargin=True)
    hinh.update_yaxes(zeroline=False, automargin=True)
    return hinh


def ve_cot_ngang(du_lieu, truong_nhan, truong_gia_tri, ten_gia_tri, mau_cot,
                 mau, chieu_cao=None):
    """Biểu đồ cột ngang xếp hạng (hạng 1 ở trên cùng), một màu duy nhất."""
    hinh = go.Figure(
        go.Bar(
            x=du_lieu[truong_gia_tri],
            y=du_lieu[truong_nhan].astype(str),
            orientation="h",
            marker=dict(color=mau_cot, cornerradius=4),
            hovertemplate=(
                "<b>%{y}</b><br>" + ten_gia_tri + ": %{x:,.0f}<extra></extra>"
            ),
        )
    )
    hinh.update_layout(bargap=0.4)
    hinh.update_yaxes(autorange="reversed", automargin=True, showgrid=False)
    hinh.update_xaxes(title=None)
    chieu_cao = chieu_cao or max(260, 26 * len(du_lieu) + 40)
    return dinh_dang_hinh(hinh, mau, chieu_cao)


def ve_duong(du_lieu, truong_x, truong_y, ten_y, mau_duong, mau, nhan_x,
             hau_to_x=""):
    """Biểu đồ đường (đường 2px, điểm 8px), đánh dấu điểm cao nhất."""
    hinh = go.Figure(
        go.Scatter(
            x=du_lieu[truong_x],
            y=du_lieu[truong_y],
            mode="lines+markers",
            line=dict(color=mau_duong, width=2),
            marker=dict(size=8, color=mau_duong),
            hovertemplate=ten_y + ": %{y:,.0f}<extra></extra>",
        )
    )
    diem_cao = du_lieu.loc[du_lieu[truong_y].idxmax()]
    hinh.add_annotation(
        x=diem_cao[truong_x], y=diem_cao[truong_y], yshift=16,
        text=f"Cao nhất: {int(diem_cao[truong_x])}{hau_to_x}",
        showarrow=False, font=dict(size=12),
    )
    hinh.update_layout(hovermode="x unified")
    hinh.update_xaxes(title=nhan_x, ticksuffix=hau_to_x, showgrid=False,
                      tickmode="linear", dtick=2)
    hinh.update_yaxes(title=None, rangemode="tozero")
    return dinh_dang_hinh(hinh, mau, 320)


def ve_cot_doc(du_lieu, truong_x, truong_y, ten_y, mau_cot, mau):
    """Biểu đồ cột đứng một màu (dùng cho phân bố theo thứ trong tuần)."""
    hinh = go.Figure(
        go.Bar(
            x=du_lieu[truong_x],
            y=du_lieu[truong_y],
            marker=dict(color=mau_cot, cornerradius=4),
            hovertemplate="<b>%{x}</b><br>" + ten_y
            + ": %{y:,.0f}<extra></extra>",
        )
    )
    hinh.update_layout(bargap=0.45)
    hinh.update_xaxes(title=None, showgrid=False)
    hinh.update_yaxes(title=None)
    return dinh_dang_hinh(hinh, mau, 320)


def hien_bang(du_lieu, **cau_hinh):
    """Hiển thị DataFrame dạng bảng gọn, không có cột chỉ số."""
    st.dataframe(du_lieu, hide_index=True, **cau_hinh)


# --- Bảng sản phẩm dùng chung ------------------------------------------------
def bang_san_pham(danh_sach, co_diem=True):
    """Chuyển danh sách sản phẩm (dict) thành DataFrame hiển thị tiếng Việt.

    Args:
        danh_sach: List dict có product_id, category_code, brand, price và
            (tuỳ chọn) score.
        co_diem: Có thêm cột điểm gợi ý hay không.
    """
    hang = []
    for thu_tu, sp in enumerate(danh_sach, start=1):
        dong = {
            "Hạng": thu_tu,
            "Mã sản phẩm": str(sp["product_id"]),
            "Danh mục": sp.get("category_code") or "(không rõ)",
            "Thương hiệu": sp.get("brand") or "(không rõ)",
            "Giá": sp.get("price"),
        }
        if co_diem:
            dong["Điểm"] = sp.get("score")
        hang.append(dong)
    return pd.DataFrame(hang)


CAU_HINH_BANG_SP = {
    "column_config": {
        "Giá": st.column_config.NumberColumn("Giá", format="%.2f"),
        "Điểm": st.column_config.NumberColumn("Điểm", format="%.4f"),
    }
}


# =============================================================================
# Trang 1: Tổng quan & phân tích
# =============================================================================
def trang_tong_quan():
    """Thẻ số liệu, phễu, top danh mục/thương hiệu, thời gian, RFM."""
    mau = bang_mau()
    st.title("Tổng quan & phân tích")
    tong_quan = lay_mot(COL_STATS_OVERVIEW)
    if tong_quan is None:
        hien_chua_co_du_lieu("python src/run_pipeline.py --drop")
        return

    cot = st.columns(5)
    cot[0].metric("Tổng sự kiện", dinh_dang_so(tong_quan["total_events"]))
    cot[1].metric("Người dùng", dinh_dang_so(tong_quan["total_users"]))
    cot[2].metric("Sản phẩm", dinh_dang_so(tong_quan["total_products"]))
    cot[3].metric("Phiên truy cập", dinh_dang_so(tong_quan["total_sessions"]))
    cot[4].metric("Doanh thu", dinh_dang_so(tong_quan["total_revenue"]))
    st.caption(
        f"Dữ liệu từ {tong_quan['first_event']:%d/%m/%Y %H:%M} đến "
        f"{tong_quan['last_event']:%d/%m/%Y %H:%M} (giờ UTC). Doanh thu tính "
        "theo đơn vị giá của dữ liệu gốc."
    )

    _muc_pheu(mau)
    _muc_xep_hang(mau)
    _muc_thoi_gian(mau)
    _muc_rfm(mau)


def _muc_pheu(mau):
    """Phễu chuyển đổi view -> cart -> purchase."""
    st.header("Phễu chuyển đổi")
    pheu = lay_mot(COL_STATS_FUNNEL)
    if pheu is None:
        hien_chua_co_du_lieu("python src/analysis.py")
        return
    cac_buoc = ["Xem (view)", "Thêm giỏ (cart)", "Mua (purchase)"]
    gia_tri = [pheu["views"], pheu["carts"], pheu["purchases"]]
    ti_le_dau = [g / gia_tri[0] if gia_tri[0] else 0 for g in gia_tri]
    nhan = [
        f"{dinh_dang_so(g)} ({dinh_dang_so(t * 100, 1)}% lượt xem)"
        for g, t in zip(gia_tri, ti_le_dau)
    ]
    hinh = go.Figure(
        go.Bar(
            x=gia_tri, y=cac_buoc, orientation="h", text=nhan,
            textposition="outside", cliponaxis=False,
            textfont=dict(size=13),
            marker=dict(color=mau["pheu"], cornerradius=4),
            hovertemplate="<b>%{y}</b><br>%{x:,.0f} sự kiện<extra></extra>",
        )
    )
    hinh.update_layout(bargap=0.45)
    hinh.update_yaxes(autorange="reversed", showgrid=False)
    hinh.update_xaxes(title=None, range=[0, max(gia_tri) * 1.6])
    dinh_dang_hinh(hinh, mau, 280)

    trai, phai = st.columns([3, 1])
    with trai:
        st.plotly_chart(hinh)
    with phai:
        st.metric("Xem → Thêm giỏ",
                  f"{dinh_dang_so(pheu['view_to_cart'] * 100, 2)}%")
        st.metric("Thêm giỏ → Mua",
                  f"{dinh_dang_so(pheu['cart_to_purchase'] * 100, 2)}%")
        st.metric("Xem → Mua",
                  f"{dinh_dang_so(pheu['view_to_purchase'] * 100, 2)}%")
    st.caption(
        "Đếm theo sự kiện: lượt mua có thể không đi qua bước thêm giỏ trong "
        "cùng phiên nên tỷ lệ giỏ → mua có thể rất cao."
    )
    with st.expander("Xem dạng bảng"):
        hien_bang(pd.DataFrame({"Bước": cac_buoc, "Số sự kiện": gia_tri}))


def _muc_xep_hang(mau):
    """Top danh mục, thương hiệu và sản phẩm theo doanh thu / lượt mua."""
    st.header("Top danh mục, thương hiệu và sản phẩm")
    tieu_chi_nhan = st.radio(
        "Xếp hạng theo", ["Doanh thu", "Lượt mua"], horizontal=True
    )
    tieu_chi = "revenue" if tieu_chi_nhan == "Doanh thu" else "purchases"
    ten_gia_tri = tieu_chi_nhan
    bo_loc = {"metric": tieu_chi}

    danh_muc = pd.DataFrame(
        truy_van(COL_STATS_TOP_CATEGORIES, bo_loc, [("rank", 1)])
    )
    thuong_hieu = pd.DataFrame(
        truy_van(COL_STATS_TOP_BRANDS, bo_loc, [("rank", 1)])
    )
    if danh_muc.empty or thuong_hieu.empty:
        hien_chua_co_du_lieu("python src/analysis.py")
        return

    trai, phai = st.columns(2)
    with trai:
        st.subheader("Top 20 danh mục")
        st.plotly_chart(
            ve_cot_ngang(danh_muc, "category_code", tieu_chi, ten_gia_tri,
                         mau["chinh"], mau),
        )
    with phai:
        st.subheader("Top 20 thương hiệu")
        st.plotly_chart(
            ve_cot_ngang(thuong_hieu, "brand", tieu_chi, ten_gia_tri,
                         mau["chinh"], mau),
        )

    with st.expander("Xem dạng bảng (danh mục, thương hiệu)"):
        t1, t2 = st.columns(2)
        with t1:
            hien_bang(danh_muc.rename(columns={
                "rank": "Hạng", "category_code": "Danh mục",
                "revenue": "Doanh thu", "purchases": "Lượt mua"
            }).drop(columns="metric"))
        with t2:
            hien_bang(thuong_hieu.rename(columns={
                "rank": "Hạng", "brand": "Thương hiệu",
                "revenue": "Doanh thu", "purchases": "Lượt mua"
            }).drop(columns="metric"))

    st.subheader("Top 20 sản phẩm")
    san_pham = pd.DataFrame(
        truy_van(COL_STATS_TOP_PRODUCTS, bo_loc, [("rank", 1)])
    )
    if not san_pham.empty:
        san_pham["product_id"] = san_pham["product_id"].astype(str)
        san_pham["category_code"] = san_pham["category_code"].fillna(
            "(không rõ)")
        san_pham["brand"] = san_pham["brand"].fillna("(không rõ)")
        cot_hien_thi = ["rank", "product_id", "category_code", "brand",
                        "revenue", "purchases"]
        hien_bang(
            san_pham[cot_hien_thi].rename(columns={
                "rank": "Hạng", "product_id": "Mã sản phẩm",
                "category_code": "Danh mục", "brand": "Thương hiệu",
                "revenue": "Doanh thu", "purchases": "Lượt mua",
            }),
            height=35 * (len(san_pham) + 1) + 3,
            column_config={
                "Doanh thu": st.column_config.NumberColumn(format="%.2f")
            },
        )


def _muc_thoi_gian(mau):
    """Số sự kiện và doanh thu theo giờ trong ngày và theo thứ trong tuần."""
    st.header("Hành vi theo thời gian")
    theo_gio = pd.DataFrame(truy_van(COL_STATS_BY_HOUR, None, [("hour", 1)]))
    theo_thu = pd.DataFrame(
        truy_van(COL_STATS_BY_WEEKDAY, None, [("weekday", 1)])
    )
    if theo_gio.empty or theo_thu.empty:
        hien_chua_co_du_lieu("python src/analysis.py")
        return

    tab_gio, tab_thu = st.tabs(["Theo giờ trong ngày", "Theo thứ trong tuần"])
    with tab_gio:
        st.caption("Giờ tính theo UTC, đúng như trong dữ liệu gốc.")
        trai, phai = st.columns(2)
        with trai:
            st.subheader("Số sự kiện")
            st.plotly_chart(
                ve_duong(theo_gio, "hour", "events", "Sự kiện", mau["chinh"],
                         mau, "Giờ trong ngày (UTC)", "h"),
            )
        with phai:
            st.subheader("Doanh thu")
            st.plotly_chart(
                ve_duong(theo_gio, "hour", "revenue", "Doanh thu",
                         mau["doanh_thu"], mau, "Giờ trong ngày (UTC)", "h"),
            )
    with tab_thu:
        trai, phai = st.columns(2)
        with trai:
            st.subheader("Số sự kiện")
            st.plotly_chart(
                ve_cot_doc(theo_thu, "label", "events", "Sự kiện",
                           mau["chinh"], mau),
            )
        with phai:
            st.subheader("Doanh thu")
            st.plotly_chart(
                ve_cot_doc(theo_thu, "label", "revenue", "Doanh thu",
                           mau["doanh_thu"], mau),
            )
    with st.expander("Xem dạng bảng (theo giờ, theo thứ)"):
        t1, t2 = st.columns(2)
        cot_hien_thi = {"events": "Sự kiện", "views": "Xem",
                        "carts": "Thêm giỏ", "purchases": "Mua",
                        "revenue": "Doanh thu"}
        with t1:
            hien_bang(theo_gio.rename(
                columns={"hour": "Giờ (UTC)", **cot_hien_thi}))
        with t2:
            hien_bang(theo_thu.drop(columns="weekday").rename(
                columns={"label": "Thứ", **cot_hien_thi}))


def _ve_cot_nhom(tom_tat, truong, nhan_truong, mau, ham_nhan):
    """Biểu đồ cột ngang theo nhóm RFM, mỗi nhóm một màu cố định."""
    thu_tu = tom_tat.set_index("segment").reindex(NHOM_RFM).dropna(
        subset=[truong])
    hinh = go.Figure(
        go.Bar(
            x=thu_tu[truong],
            y=thu_tu.index,
            orientation="h",
            marker=dict(
                color=[mau["nhom"][NHOM_RFM.index(n)] for n in thu_tu.index],
                cornerradius=4,
            ),
            text=[ham_nhan(dong) for _, dong in thu_tu.iterrows()],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(size=13),
            hovertemplate="<b>%{y}</b><br>" + nhan_truong
            + ": %{x:,.0f}<extra></extra>",
        )
    )
    hinh.update_layout(bargap=0.45)
    hinh.update_yaxes(autorange="reversed", automargin=True, showgrid=False)
    hinh.update_xaxes(title=None, range=[0, thu_tu[truong].max() * 1.85])
    return dinh_dang_hinh(hinh, mau, 300)


def _muc_rfm(mau):
    """Phân nhóm khách hàng RFM."""
    st.header("Phân nhóm khách hàng (RFM)")
    tom_tat = pd.DataFrame(truy_van(COL_STATS_RFM_SUMMARY))
    if tom_tat.empty:
        hien_chua_co_du_lieu("python src/analysis.py")
        return
    st.caption(
        "Mỗi khách có mua hàng được chấm 1–5 điểm cho Recency (mua gần đây), "
        "Frequency (số phiên có mua) và Monetary (tổng chi tiêu) theo phân vị."
    )
    trai, phai = st.columns(2)
    with trai:
        st.subheader("Số khách hàng theo nhóm")
        st.plotly_chart(
            _ve_cot_nhom(
                tom_tat, "users", "Khách hàng", mau,
                lambda d: f"{dinh_dang_so(d['users'])} "
                          f"({dinh_dang_so(d['pct_users'], 1)}%)",
            ),
        )
    with phai:
        st.subheader("Doanh thu theo nhóm")
        st.plotly_chart(
            _ve_cot_nhom(
                tom_tat, "revenue", "Doanh thu", mau,
                lambda d: f"{dinh_dang_so(d['revenue'])} "
                          f"({dinh_dang_so(d['pct_revenue'], 1)}%)",
            ),
        )

    bang = tom_tat.set_index("segment").reindex(NHOM_RFM).reset_index()
    cot_bang = ["segment", "users", "pct_users", "revenue", "pct_revenue",
                "avg_recency_days", "avg_frequency", "avg_monetary"]
    hien_bang(
        bang[cot_bang].rename(columns={
            "segment": "Nhóm khách hàng", "users": "Số khách",
            "pct_users": "% khách", "revenue": "Doanh thu",
            "pct_revenue": "% doanh thu",
            "avg_recency_days": "Recency TB (ngày)",
            "avg_frequency": "Frequency TB (phiên)",
            "avg_monetary": "Monetary TB",
        }),
        column_config={
            "Số khách": st.column_config.NumberColumn(format="%d"),
            "% khách": st.column_config.NumberColumn(format="%.1f"),
            "Doanh thu": st.column_config.NumberColumn(format="%.2f"),
            "% doanh thu": st.column_config.NumberColumn(format="%.1f"),
            "Recency TB (ngày)": st.column_config.NumberColumn(format="%.1f"),
            "Frequency TB (phiên)": st.column_config.NumberColumn(
                format="%.2f"),
            "Monetary TB": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    with st.expander("Cách phân nhóm khách hàng"):
        st.markdown(
            "Quy tắc xét lần lượt từ trên xuống (R, F, M là điểm 1–5):\n"
            "- **Khách VIP**: R ≥ 4, F ≥ 4 và M ≥ 4\n"
            "- **Khách trung thành**: R ≥ 3 và F ≥ 4\n"
            "- **Khách tiềm năng**: R ≥ 4 (mới mua nhưng chưa thường xuyên)\n"
            "- **Khách cần chăm sóc**: R = 3, hoặc R ≤ 2 nhưng từng mua "
            "nhiều\n"
            "- **Khách sắp rời bỏ**: còn lại (lâu không mua và ít mua)"
        )


# =============================================================================
# Trang 2: Gợi ý cho người dùng
# =============================================================================
def lay_tuong_tac_gan_day(user_id):
    """Lấy 20 sự kiện gần nhất của một người dùng."""
    su_kien = truy_van(
        COL_EVENTS, {"user_id": user_id}, [("event_time", -1)], 20
    )
    if not su_kien:
        return pd.DataFrame()
    df = pd.DataFrame(su_kien)
    return pd.DataFrame({
        "Thời gian (UTC)": df["event_time"].dt.strftime("%d/%m/%Y %H:%M:%S"),
        "Hành động": df["event_type"].map(NHAN_SU_KIEN).fillna(
            df["event_type"]),
        "Mã sản phẩm": df["product_id"].astype(str),
        "Danh mục": df["category_code"].fillna("(không rõ)"),
        "Thương hiệu": df["brand"].fillna("(không rõ)"),
        "Giá": df["price"],
    })


def _nguong_toi_thieu_user():
    """Ngưỡng tương tác tối thiểu đã dùng ở lần huấn luyện gần nhất."""
    lan_cuoi = lay_mot(COL_MODEL_METRICS, None, [("created_at", -1)])
    if lan_cuoi:
        return lan_cuoi["params"].get(
            "min_user_interactions", MIN_USER_INTERACTIONS
        )
    return MIN_USER_INTERACTIONS


def trang_goi_y():
    """Xem lịch sử và gợi ý cho một người dùng; cold-start nếu chưa có."""
    st.title("Gợi ý cho người dùng")
    if lay_mot(COL_RECOMMENDATIONS) is None:
        hien_chua_co_du_lieu("python src/run_pipeline.py --drop")
        return

    trai, phai = st.columns(2)
    with trai:
        nhap = st.text_input(
            "Nhập user_id", placeholder="ví dụ: 512345678",
        )
    with phai:
        chon = st.selectbox(
            "…hoặc chọn nhanh một người dùng có gợi ý",
            options=mau_ngau_nhien(COL_RECOMMENDATIONS, "user_id"),
            index=None, placeholder="Chọn user_id",
        )

    if nhap.strip():
        try:
            user_id = int(nhap.strip())
        except ValueError:
            st.error("user_id phải là một số nguyên, ví dụ 512345678.")
            return
    elif chon is not None:
        user_id = chon
    else:
        st.info("Hãy nhập user_id hoặc chọn một người dùng ở trên.")
        return

    so_su_kien = dem_su_kien_user(user_id)
    st.subheader(f"Người dùng {user_id}")
    if so_su_kien:
        st.caption(
            f"Tổng số sự kiện trong hệ thống: {dinh_dang_so(so_su_kien)}"
        )

    st.markdown("#### 20 tương tác gần nhất")
    lich_su = lay_tuong_tac_gan_day(user_id)
    if lich_su.empty:
        st.write("Người dùng này chưa có tương tác nào trong hệ thống.")
    else:
        hien_bang(
            lich_su,
            column_config={
                "Giá": st.column_config.NumberColumn("Giá", format="%.2f")
            },
        )

    st.markdown("#### 10 sản phẩm được gợi ý")
    goi_y = lay_mot(COL_RECOMMENDATIONS, {"user_id": user_id})
    if goi_y and goi_y["recommendations"]:
        hien_bang(bang_san_pham(goi_y["recommendations"]),
                  **CAU_HINH_BANG_SP)
        st.caption(
            "Điểm là mức độ phù hợp do mô hình ALS tính; các sản phẩm người "
            "dùng đã tương tác đã được loại khỏi danh sách."
        )
        return

    if so_su_kien:
        st.warning(
            f"**Cold-start:** người dùng này có {so_su_kien} sự kiện nhưng "
            f"chưa đủ {_nguong_toi_thieu_user()} sản phẩm tương tác để được "
            "huấn luyện, nên hiển thị các sản phẩm phổ biến nhất."
        )
    else:
        st.warning(
            "**Cold-start:** không tìm thấy người dùng này trong hệ thống, "
            "nên hiển thị các sản phẩm phổ biến nhất."
        )
    pho_bien = truy_van(COL_POPULAR_PRODUCTS, None, [("rank", 1)], 10)
    if pho_bien:
        hien_bang(bang_san_pham(pho_bien, co_diem=False),
                  **CAU_HINH_BANG_SP)
    else:
        hien_chua_co_du_lieu("python src/train_model.py")


# =============================================================================
# Trang 3: Sản phẩm tương tự
# =============================================================================
def trang_san_pham_tuong_tu():
    """Hiện thông tin một sản phẩm và 10 sản phẩm tương tự."""
    st.title("Sản phẩm tương tự")
    if lay_mot(COL_SIMILAR_ITEMS) is None:
        hien_chua_co_du_lieu("python src/run_pipeline.py --drop")
        return

    pho_bien = truy_van(COL_POPULAR_PRODUCTS, None, [("rank", 1)], 30)
    nhan_chon = {
        f"{sp['product_id']} — {sp.get('category_code') or '(không rõ)'}":
            sp["product_id"]
        for sp in pho_bien
    }
    trai, phai = st.columns(2)
    with trai:
        nhap = st.text_input("Nhập product_id", placeholder="ví dụ: 1004767")
    with phai:
        chon = st.selectbox(
            "…hoặc chọn nhanh một sản phẩm phổ biến",
            options=list(nhan_chon), index=None, placeholder="Chọn sản phẩm",
        )

    if nhap.strip():
        try:
            product_id = int(nhap.strip())
        except ValueError:
            st.error("product_id phải là một số nguyên, ví dụ 1004767.")
            return
    elif chon is not None:
        product_id = nhan_chon[chon]
    else:
        st.info("Hãy nhập product_id hoặc chọn một sản phẩm ở trên.")
        return

    thong_tin = lay_mot(COL_PRODUCTS, {"product_id": product_id})
    if thong_tin is None:
        st.error(f"Không tìm thấy sản phẩm {product_id} trong hệ thống.")
        return

    st.subheader(f"Sản phẩm {product_id}")
    cot = st.columns(3)
    cot[0].metric("Danh mục", thong_tin.get("category_code") or "(không rõ)")
    cot[1].metric("Thương hiệu", thong_tin.get("brand") or "(không rõ)")
    cot[2].metric(
        "Giá trung bình", dinh_dang_so(thong_tin.get("avg_price"), 2)
    )
    cot = st.columns(3)
    cot[0].metric("Lượt xem", dinh_dang_so(thong_tin.get("views", 0)))
    cot[1].metric("Lượt thêm giỏ", dinh_dang_so(thong_tin.get("carts", 0)))
    cot[2].metric("Lượt mua", dinh_dang_so(thong_tin.get("purchases", 0)))

    st.markdown("#### 10 sản phẩm tương tự")
    tuong_tu = lay_mot(COL_SIMILAR_ITEMS, {"product_id": product_id})
    if tuong_tu is None:
        st.info(
            "Sản phẩm này chưa có danh sách tương tự vì có quá ít tương tác "
            "(mô hình chỉ tính cho sản phẩm có từ 5 tương tác trở lên)."
        )
        return
    hien_bang(bang_san_pham(tuong_tu["similar"]), **CAU_HINH_BANG_SP)
    st.caption(
        "Điểm là độ tương tự cosine giữa các vector nhân tố ẩn của mô hình "
        "ALS (càng gần 1 càng giống)."
    )


# =============================================================================
# Trang 4: Đánh giá mô hình
# =============================================================================
TEN_DO_DO = {
    "precision": "Precision",
    "map": "MAP",
    "ndcg": "NDCG",
    "auc": "AUC",
}


def trang_danh_gia_mo_hinh():
    """So sánh ALS với baseline phổ biến nhất từ ``model_metrics``."""
    mau = bang_mau()
    st.title("Đánh giá mô hình")
    cac_lan = truy_van(COL_MODEL_METRICS, None, [("created_at", -1)])
    if not cac_lan:
        hien_chua_co_du_lieu("python src/train_model.py")
        return
    moi_nhat = cac_lan[0]
    k = moi_nhat["k"]
    als = moi_nhat["results"]["als"]
    co_so = moi_nhat["results"]["popularity"]

    st.caption(
        f"Lần huấn luyện gần nhất: {moi_nhat['created_at']:%d/%m/%Y %H:%M} "
        f"(UTC). Đo trên tập test (20% tương tác được giữ lại), K = {k}."
    )

    ten_do = [f"{TEN_DO_DO[ten]}@{k}" for ten in TEN_DO_DO]
    hinh = go.Figure([
        go.Bar(
            name="ALS", x=ten_do, y=[als[t] for t in TEN_DO_DO],
            marker=dict(color=mau["chinh"], cornerradius=4),
            text=[f"{als[t]:.3f}" for t in TEN_DO_DO],
            textposition="outside", cliponaxis=False,
            textfont=dict(size=13),
            hovertemplate="<b>ALS</b> — %{x}: %{y:.4f}<extra></extra>",
        ),
        go.Bar(
            name="Baseline: sản phẩm phổ biến nhất", x=ten_do,
            y=[co_so[t] for t in TEN_DO_DO],
            marker=dict(color=mau["xam"], cornerradius=4),
            text=[f"{co_so[t]:.3f}" for t in TEN_DO_DO],
            textposition="outside", cliponaxis=False,
            textfont=dict(size=13),
            hovertemplate="<b>Phổ biến nhất</b> — %{x}: %{y:.4f}"
                          "<extra></extra>",
        ),
    ])
    hinh.update_layout(barmode="group", bargap=0.4, bargroupgap=0.06)
    hinh.update_xaxes(showgrid=False)
    hinh.update_yaxes(title=None, range=[0, max(als.values()) * 1.2])
    dinh_dang_hinh(hinh, mau, 400, chu_thich=True)
    st.plotly_chart(hinh)

    bang = pd.DataFrame({
        "Độ đo": ten_do,
        "ALS": [als[t] for t in TEN_DO_DO],
        "Baseline phổ biến nhất": [co_so[t] for t in TEN_DO_DO],
        "ALS / Baseline (lần)": [
            als[t] / co_so[t] if co_so[t] else None for t in TEN_DO_DO
        ],
    })
    hien_bang(
        bang,
        column_config={
            "ALS": st.column_config.NumberColumn(format="%.4f"),
            "Baseline phổ biến nhất": st.column_config.NumberColumn(
                format="%.4f"),
            "ALS / Baseline (lần)": st.column_config.NumberColumn(
                format="%.1f"),
        },
    )
    st.caption(
        "Các độ đo tính bằng implicit.evaluation.ranking_metrics_at_k. "
        "Precision của thư viện này chia cho min(K, số sản phẩm test của "
        "người dùng) nên cao hơn precision@K = số trúng / K thông thường; "
        "ALS và baseline được đo cùng một cách trên cùng tập test."
    )

    st.subheader("Tham số và dữ liệu huấn luyện")
    tham_so = moi_nhat["params"]
    du_lieu = moi_nhat["dataset"]
    giay = moi_nhat["train_seconds"]
    cot = st.columns(4)
    cot[0].metric("Số nhân tố (factors)", tham_so["factors"])
    cot[1].metric("Chính quy hoá", tham_so["regularization"])
    cot[2].metric("Alpha", tham_so["alpha"])
    cot[3].metric("Số vòng lặp", tham_so["iterations"])
    cot = st.columns(4)
    cot[0].metric("Người dùng", dinh_dang_so(du_lieu["n_users"]))
    cot[1].metric("Sản phẩm", dinh_dang_so(du_lieu["n_items"]))
    cot[2].metric("Tương tác", dinh_dang_so(du_lieu["n_interactions"]))
    cot[3].metric("Mật độ ma trận", f"{du_lieu['density'] * 100:.2f}%".replace(
        ".", ","))
    cot = st.columns(4)
    cot[0].metric("Huấn luyện (đánh giá)",
                  f"{dinh_dang_so(giay['evaluation_fit'], 1)} s")
    cot[1].metric("Huấn luyện (toàn bộ)",
                  f"{dinh_dang_so(giay['final_fit'], 1)} s")
    cot[2].metric("Thời gian đánh giá",
                  f"{dinh_dang_so(moi_nhat['eval_seconds'], 1)} s")
    cot[3].metric("Người dùng bị loại",
                  dinh_dang_so(du_lieu["n_users_before_filter"]
                               - du_lieu["n_users"]))

    with st.expander(f"Lịch sử các lần huấn luyện ({len(cac_lan)})"):
        lich_su = pd.DataFrame([
            {
                "Thời gian (UTC)": lan["created_at"].strftime(
                    "%d/%m/%Y %H:%M:%S"),
                "factors": lan["params"]["factors"],
                "regularization": lan["params"]["regularization"],
                "alpha": lan["params"]["alpha"],
                "iterations": lan["params"]["iterations"],
                "min user": lan["params"].get("min_user_interactions"),
                f"Precision@{lan['k']} ALS": lan["results"]["als"][
                    "precision"],
                f"Precision@{lan['k']} baseline": lan["results"][
                    "popularity"]["precision"],
                f"NDCG@{lan['k']} ALS": lan["results"]["als"]["ndcg"],
            }
            for lan in cac_lan
        ])
        hien_bang(lich_su)


# =============================================================================
# Điều hướng
# =============================================================================
TRANG = {
    "Tổng quan & phân tích": trang_tong_quan,
    "Gợi ý cho người dùng": trang_goi_y,
    "Sản phẩm tương tự": trang_san_pham_tuong_tu,
    "Đánh giá mô hình": trang_danh_gia_mo_hinh,
}


def main():
    """Dựng sidebar, kiểm tra kết nối MongoDB và hiển thị trang được chọn."""
    with st.sidebar:
        st.title("🛒 E-commerce")
        st.caption("Phân tích dữ liệu thương mại điện tử & hệ thống gợi ý "
                   "sản phẩm")
        ten_trang = st.radio("Trang", list(TRANG))
        if st.button("Làm mới dữ liệu"):
            st.cache_data.clear()
            st.rerun()

    try:
        ket_noi_mongo()
    except MongoConnectionError as loi:
        st.error(f"Không kết nối được MongoDB.\n\n```\n{loi}\n```")
        st.stop()

    with st.sidebar:
        st.caption(f"Database: `{DB_NAME}`")
    TRANG[ten_trang]()


main()
