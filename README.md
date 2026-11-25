# Phân tích dữ liệu thương mại điện tử và hệ thống gợi ý sản phẩm

Đồ án môn **Công nghệ khai thác Big Data**.

Hệ thống nạp dữ liệu hành vi mua sắm (xem, thêm giỏ, mua) vào **MongoDB**,
phân tích bằng **aggregation pipeline**, huấn luyện mô hình gợi ý
**ALS (implicit feedback)** và trình bày kết quả trên dashboard **Streamlit**.
Không dùng Spark/Hadoop.

## Mục lục

1. [Tính năng](#1-tính-năng)
2. [Công nghệ](#2-công-nghệ)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Cài đặt trên Windows](#4-cài-đặt-trên-windows)
5. [Chuẩn bị dữ liệu](#5-chuẩn-bị-dữ-liệu)
6. [Chạy pipeline](#6-chạy-pipeline)
7. [Chạy dashboard](#7-chạy-dashboard)
8. [Cấu hình và SAMPLE_PERCENT](#8-cấu-hình-và-sample_percent)
9. [Ảnh chụp màn hình](#9-ảnh-chụp-màn-hình)
10. [Xử lý sự cố](#10-xử-lý-sự-cố)
11. [Tài liệu](#11-tài-liệu)

## 1. Tính năng

- **Nạp dữ liệu** CSV hàng chục triệu dòng vào MongoDB theo từng chunk, lấy mẫu
  theo người dùng, làm sạch và tạo index.
- **Phân tích** bằng aggregation pipeline: tổng quan, phễu chuyển đổi, top
  danh mục/thương hiệu/sản phẩm, hành vi theo giờ và thứ, phân nhóm khách hàng
  **RFM**.
- **Gợi ý sản phẩm** bằng ALS: top-10 cho mọi người dùng, top-10 sản phẩm tương
  tự, danh sách sản phẩm phổ biến cho người dùng mới (cold-start).
- **Đánh giá** Precision/MAP/NDCG/AUC @10 và so sánh với baseline "sản phẩm phổ
  biến nhất" trên cùng tập test.
- **Dashboard** 4 trang: tổng quan và phân tích, gợi ý cho người dùng, sản phẩm
  tương tự, đánh giá mô hình.

## 2. Công nghệ

| Thành phần | Công nghệ |
|---|---|
| Ngôn ngữ | Python 3.11+ |
| Cơ sở dữ liệu | MongoDB 8.0 (chạy bằng Docker) |
| Xử lý dữ liệu | pymongo, pandas, numpy, scipy |
| Mô hình gợi ý | implicit (thuật toán ALS) |
| Dashboard | Streamlit, Plotly |
| Khác | python-dotenv, tqdm |

## 3. Cấu trúc thư mục

```
├── docker-compose.yml        # MongoDB 8.0 + volume lưu dữ liệu
├── requirements.txt
├── .env.example              # mẫu cấu hình (sao chép thành .env)
├── README.md
├── src/
│   ├── config.py             # đọc cấu hình từ .env
│   ├── db.py                 # get_db(): kết nối MongoDB, báo lỗi rõ ràng
│   ├── import_data.py        # nạp CSV vào collection events
│   ├── analysis.py           # phân tích bằng aggregation -> stats_*
│   ├── build_interactions.py # tạo products và interactions
│   ├── train_model.py        # ALS: đánh giá, gợi ý, sản phẩm tương tự
│   └── run_pipeline.py       # chạy tuần tự các bước trên
├── app/
│   └── app.py                # dashboard Streamlit
├── scripts/
│   └── generate_sample.py    # sinh CSV giả lập để chạy thử
└── docs/
    ├── kien-truc.md          # kiến trúc, schema các collection
    ├── dan-y-bao-cao.md      # dàn ý báo cáo đồ án
    └── images/               # nơi đặt ảnh chụp màn hình
```

Các thư mục `data/` (file CSV), `models/` (mô hình đã huấn luyện) và file `.env`
được tạo trên máy bạn và **không** được đưa lên GitHub.

## 4. Cài đặt trên Windows

### Bước 1. Cài Python

1. Tải Python **3.11 hoặc 3.12** tại <https://www.python.org/downloads/windows/>.
2. Khi cài, **tick vào "Add python.exe to PATH"**.
3. Mở PowerShell hoặc Command Prompt và kiểm tra:

```bash
python --version
```

> Nên dùng Python 3.11 hoặc 3.12: các thư viện tính toán (đặc biệt
> `implicit`) cung cấp sẵn gói cài đặt cho các phiên bản này trên Windows. Với
> phiên bản quá mới, `pip` có thể phải biên dịch từ mã nguồn và dễ lỗi.

### Bước 2. Cài Docker Desktop

1. Tải và cài Docker Desktop tại <https://www.docker.com/products/docker-desktop/>
   (bật tuỳ chọn dùng **WSL 2** khi được hỏi, khởi động lại máy nếu cần).
2. Mở Docker Desktop và đợi đến khi báo **Engine running**.
3. Kiểm tra:

```bash
docker --version
docker compose version
```

### Bước 3. Lấy mã nguồn

```bash
git clone https://github.com/DTRUNG1401/bigdata.git
cd bigdata
```

### Bước 4. Tạo môi trường ảo và cài thư viện

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> Nếu PowerShell báo không cho chạy script `activate`, chạy lệnh sau rồi kích
> hoạt lại (chỉ có hiệu lực trong cửa sổ hiện tại):
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

Sau khi kích hoạt, đầu dòng lệnh có `(.venv)`. Mỗi lần mở cửa sổ mới cần chạy
lại `.venv\Scripts\activate`.

### Bước 5. Tạo file cấu hình

```bash
copy .env.example .env
```

Mặc định đã chạy được ngay; chỉ cần mở `.env` để chỉnh khi muốn đổi
(xem [mục 8](#8-cấu-hình-và-sample_percent)).

### Bước 6. Khởi động MongoDB

```bash
docker compose up -d
docker compose ps
```

Cột `STATUS` hiện `healthy` là MongoDB đã sẵn sàng (lần đầu cần tải image
`mongo:8.0`). Dữ liệu được lưu trong volume Docker nên không mất khi tắt máy.
Kiểm tra kết nối từ Python:

```bash
python src/db.py
```

> Trên Linux/macOS: dùng `python3 -m venv .venv`, `source .venv/bin/activate`
> và `cp .env.example .env`; các lệnh còn lại giữ nguyên.

## 5. Chuẩn bị dữ liệu

Có hai cách. Nên chạy thử bằng **dữ liệu mẫu** trước để chắc chắn mọi thứ hoạt
động, rồi mới tải dữ liệu thật.

### Cách A. Dữ liệu mẫu giả lập (chạy được ngay, không cần tài khoản)

```bash
python scripts/generate_sample.py
```

Sinh file `data/raw/sample_events.csv` (~200.000 dòng, 5.000 người dùng, 2.000
sản phẩm, cùng schema với dữ liệu thật) trong vài giây. Tuỳ chọn:
`--rows`, `--users`, `--products`, `--seed`, `--output`.

> Dữ liệu mẫu có tỷ lệ chuyển đổi cao hơn thật và sở thích người dùng rất rõ
> ràng, nên các chỉ số mô hình trên dữ liệu mẫu **cao hơn nhiều** so với dữ liệu
> thật. Chỉ dùng để kiểm tra chương trình chạy đúng, **không** dùng số liệu này
> trong báo cáo.

### Cách B. Dữ liệu thật: bộ REES46 trên Kaggle

Bộ dữ liệu **"eCommerce behavior data from multi category store"**
(slug `mkechinov/ecommerce-behavior-data-from-multi-category-store`), dùng file
`2019-Oct.csv` (khoảng 42 triệu dòng, dung lượng khoảng 5-6 GB sau khi giải
nén). Các cột: `event_time`, `event_type` (`view`/`cart`/`purchase`),
`product_id`, `category_id`, `category_code`, `brand`, `price`, `user_id`,
`user_session`.

**Tải thủ công (đơn giản nhất):**

1. Đăng nhập Kaggle, mở
   <https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store>.
2. Tải file `2019-Oct.csv` (hoặc tải cả bộ rồi giải nén).
3. Tạo thư mục `data\raw` trong thư mục dự án (nếu chưa có) và đặt file vào:
   `data\raw\2019-Oct.csv`.

**Hoặc dùng Kaggle CLI:**

1. Trên Kaggle: *Settings -> API -> Create New Token* để tải `kaggle.json`.
2. Đặt file vào `%USERPROFILE%\.kaggle\kaggle.json`.
3. Chạy trong thư mục dự án (đã kích hoạt `.venv`):

```bash
pip install kaggle
kaggle datasets download -d mkechinov/ecommerce-behavior-data-from-multi-category-store -f 2019-Oct.csv -p data/raw --unzip
```

Nếu lệnh chỉ tải về `2019-Oct.csv.zip`, hãy giải nén thủ công vào `data\raw\`.

> Đảm bảo ổ đĩa còn trống tối thiểu **15 GB** (file CSV, dữ liệu MongoDB và
> cache của Docker).

## 6. Chạy pipeline

Pipeline gồm 4 bước chạy tuần tự: **nạp dữ liệu -> phân tích -> tạo bảng tương
tác -> huấn luyện mô hình**, in thời gian từng bước.

**Với dữ liệu mẫu:**

```bash
python src/run_pipeline.py --csv data/raw/sample_events.csv --drop
```

**Với dữ liệu thật** (mặc định đọc `data/raw/2019-Oct.csv` theo `CSV_PATH` trong
`.env`):

```bash
python src/run_pipeline.py --drop
```

`--drop` xoá collection `events` cũ trước khi nạp. Nếu `events` đã có dữ liệu mà
không có `--drop`, chương trình sẽ dừng và báo lỗi để tránh nạp trùng làm sai kết
quả.

Các tuỳ chọn thường dùng:

| Tuỳ chọn | Ý nghĩa |
|---|---|
| `--csv ĐƯỜNG_DẪN` | file CSV cần nạp (mặc định theo `CSV_PATH`) |
| `--sample-percent N` | chỉ nạp N% người dùng (ghi đè `SAMPLE_PERCENT`) |
| `--drop` | xoá `events` cũ trước khi nạp |
| `--append` | nạp thêm vào `events` đã có (ví dụ thêm file tháng khác) |
| `--skip-import` | bỏ qua bước nạp, chỉ chạy lại phân tích và huấn luyện |
| `--factors`, `--regularization`, `--alpha`, `--iterations` | tham số ALS (mặc định 64, 0.05, 1.0, 20) |
| `--min-user-interactions N` | số sản phẩm tối thiểu của người dùng khi huấn luyện (mặc định 2) |
| `--seed N` | hạt giống ngẫu nhiên (mặc định 42) |
| `--no-save-model` | không lưu file mô hình vào `models/` |

Ví dụ thử một bộ tham số khác mà không phải nạp lại dữ liệu:

```bash
python src/run_pipeline.py --skip-import --factors 128 --iterations 30
```

Mỗi lần huấn luyện được lưu thành một dòng trong `model_metrics`, nên có thể so
sánh các bộ tham số ngay trên dashboard.

**Chạy riêng từng bước** (khi cần chạy lại một phần):

```bash
python src/import_data.py --drop      # 1. nạp CSV vào MongoDB
python src/analysis.py                # 2. phân tích bằng aggregation
python src/build_interactions.py      # 3. tạo products, interactions
python src/train_model.py             # 4. huấn luyện ALS, sinh gợi ý
```

### Kết quả chạy thử trên dữ liệu mẫu

Pipeline đầy đủ trên file mẫu mặc định chạy khoảng 10 giây. Các chỉ số @10 trên
tập test (ALS so với baseline "sản phẩm phổ biến nhất"):

| Độ đo | ALS | Phổ biến nhất |
|---|---|---|
| Precision | 0,2673 | 0,0200 |
| MAP | 0,1215 | 0,0058 |
| NDCG | 0,2133 | 0,0147 |
| AUC | 0,6114 | 0,5065 |

Đây là số liệu của **dữ liệu giả lập** (xem lưu ý ở mục 5), chỉ để minh hoạ
pipeline chạy đúng. Kết quả trên dữ liệu thật sẽ khác.

## 7. Chạy dashboard

Sau khi pipeline chạy xong:

```bash
streamlit run app/app.py
```

Trình duyệt tự mở <http://localhost:8501>. Bốn trang trên sidebar:

1. **Tổng quan & phân tích**: thẻ số liệu, phễu chuyển đổi, top danh mục và
   thương hiệu, hành vi theo giờ/thứ, phân nhóm khách hàng RFM.
2. **Gợi ý cho người dùng**: nhập hoặc chọn `user_id` để xem 20 tương tác gần
   nhất và 10 sản phẩm được gợi ý. Người dùng chưa có trong hệ thống được
   hiển thị sản phẩm phổ biến kèm ghi chú *cold-start*.
3. **Sản phẩm tương tự**: nhập `product_id` để xem thông tin sản phẩm và 10 sản
   phẩm tương tự.
4. **Đánh giá mô hình**: so sánh ALS với baseline, tham số và thời gian huấn
   luyện, lịch sử các lần huấn luyện.

Bấm **Làm mới dữ liệu** ở sidebar sau khi chạy lại pipeline để xoá cache.
Dashboard tự theo giao diện sáng/tối của Streamlit.

Chạy không mở trình duyệt (ví dụ trên máy chủ): thêm
`--server.headless true`.

## 8. Cấu hình và SAMPLE_PERCENT

Các tham số đặt trong file `.env` (sao chép từ `.env.example`). Biến môi trường
của hệ thống, nếu có, được ưu tiên hơn `.env`.

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017` | địa chỉ MongoDB |
| `DB_NAME` | `ecommerce` | tên database |
| `CSV_PATH` | `data/raw/2019-Oct.csv` | file CSV cần nạp (tính từ thư mục dự án) |
| `SAMPLE_PERCENT` | `100` | phần trăm người dùng được nạp (1-100) |
| `CHUNK_SIZE` | `500000` | số dòng CSV đọc mỗi lượt |
| `MIN_USER_INTERACTIONS` | `2` | số sản phẩm tối thiểu của người dùng khi huấn luyện |

### SAMPLE_PERCENT là gì?

Chỉ giữ các dòng có `user_id % 100 < SAMPLE_PERCENT`. Ví dụ `SAMPLE_PERCENT=10`
giữ khoảng 10% **người dùng** (những người có `user_id % 100` từ 0 đến 9) và với
mỗi người được giữ thì giữ **toàn bộ** lịch sử của họ. Lấy mẫu theo người dùng
(thay vì theo dòng) để ma trận người dùng x sản phẩm không bị thưa đi một cách
giả tạo và mô hình gợi ý vẫn có ý nghĩa.

Dữ liệu thật tháng 10/2019 rất lớn (khoảng 42 triệu sự kiện, hàng triệu người
dùng). Ma trận huấn luyện phải nằm vừa trong RAM, nên chọn theo máy:

| RAM của máy | `SAMPLE_PERCENT` gợi ý |
|---|---|
| 8 GB | **10-20** |
| 16 GB | 30-50 |
| 32 GB trở lên | 100 (lấy hết) |

Đây là gợi ý ban đầu: hãy bắt đầu với giá trị thấp, chạy thành công rồi tăng
dần. Nếu bị thiếu RAM khi nạp, giảm `CHUNK_SIZE` (ví dụ `100000`); nếu thiếu RAM
khi huấn luyện, giảm `SAMPLE_PERCENT`. Máy 8 GB cũng nên bỏ comment dòng
`command: ["--wiredTigerCacheSizeGB", "2"]` trong `docker-compose.yml` để giới
hạn bộ nhớ đệm của MongoDB, rồi chạy lại `docker compose up -d`.

## 9. Ảnh chụp màn hình

Chèn ảnh chụp dashboard (lưu vào thư mục `docs/images/`) vào các chỗ dưới đây.

> 📷 *Chèn ảnh chụp màn hình trang **Tổng quan & phân tích** tại đây*
> (`docs/images/tong-quan.png`)

> 📷 *Chèn ảnh chụp màn hình trang **Gợi ý cho người dùng** tại đây*
> (`docs/images/goi-y-nguoi-dung.png`)

> 📷 *Chèn ảnh chụp màn hình trang **Sản phẩm tương tự** tại đây*
> (`docs/images/san-pham-tuong-tu.png`)

> 📷 *Chèn ảnh chụp màn hình trang **Đánh giá mô hình** tại đây*
> (`docs/images/danh-gia-mo-hinh.png`)

Sau khi có ảnh, thay mỗi khối trên bằng dòng
`![Mô tả](docs/images/ten-anh.png)`.

## 10. Xử lý sự cố

| Triệu chứng | Cách xử lý |
|---|---|
| `Không kết nối được MongoDB` | Mở Docker Desktop, chạy `docker compose up -d`, kiểm tra `docker compose ps` đã `healthy`, kiểm tra `MONGO_URI` trong `.env`. |
| Cổng 27017 đã bị chiếm (đã cài MongoDB trên máy) | Tắt MongoDB đang chạy, hoặc đổi cổng trong `docker-compose.yml` (ví dụ `"127.0.0.1:27018:27017"`) và `MONGO_URI` trong `.env`. |
| `Không tìm thấy file CSV` | Kiểm tra `CSV_PATH` trong `.env` hoặc truyền `--csv`; tên file phân biệt đúng chính tả. |
| `Collection 'events' đã có dữ liệu` | Thêm `--drop` để nạp lại từ đầu, hoặc `--skip-import` để dùng dữ liệu đã nạp. |
| Hết RAM khi nạp hoặc huấn luyện | Giảm `CHUNK_SIZE` và `SAMPLE_PERCENT` (xem mục 8). |
| `pip install` lỗi ở `implicit` | Dùng Python 3.11 hoặc 3.12 (mục 4, bước 1) rồi tạo lại `.venv`. |
| Dashboard báo chưa có dữ liệu | Chạy `python src/run_pipeline.py --drop` rồi bấm **Làm mới dữ liệu**. |
| Muốn xoá sạch dữ liệu MongoDB | `docker compose down -v` (xoá cả volume), sau đó `docker compose up -d`. |

## 11. Tài liệu

- [`docs/kien-truc.md`](docs/kien-truc.md): sơ đồ luồng dữ liệu, mô tả từng
  collection kèm schema mẫu, lý do chọn MongoDB (schema linh hoạt, aggregation
  pipeline, index, sharding).
- [`docs/dan-y-bao-cao.md`](docs/dan-y-bao-cao.md): dàn ý báo cáo đồ án, ghi rõ
  chỗ chèn hình và bảng lấy từ dashboard.
- Dữ liệu thuộc REES46, sử dụng theo điều khoản của bộ dữ liệu trên Kaggle.
