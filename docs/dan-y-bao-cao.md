# Dàn ý báo cáo đồ án

**Đề tài:** Phân tích dữ liệu thương mại điện tử và hệ thống gợi ý sản phẩm
(E-commerce Recommendation System)
**Môn học:** Công nghệ khai thác Big Data
**Công nghệ:** Python 3.11+, MongoDB 8.0, thuật toán ALS (thư viện `implicit`),
Streamlit, Plotly

> **Cách đọc dàn ý.** Mục nào cần minh hoạ có dòng bắt đầu bằng
> `[CHÈN HÌNH]` hoặc `[CHÈN BẢNG]`, kèm nguồn lấy: trang nào của dashboard
> (`streamlit run app/app.py`), mục nào bấm "Xem dạng bảng", hoặc lệnh nào để
> xuất. **Chụp ảnh và lấy số liệu từ lần chạy trên dữ liệu thật
> `2019-Oct.csv`**, không dùng dữ liệu mẫu giả lập (xem chương 5).
> Gợi ý độ dài: 35-50 trang.

---

## Lời mở đầu

- Lời cảm ơn, mục lục, danh mục hình, danh mục bảng, danh mục từ viết tắt (RFM,
  ALS, CF, NDCG, MAP, CSR, ...).

## Chương 1. Tổng quan đề tài

### 1.1. Đặt vấn đề
- Thương mại điện tử sinh ra lượng lớn dữ liệu hành vi (xem, thêm giỏ, mua);
  cần phân tích để hiểu khách hàng và gợi ý sản phẩm phù hợp.
- Bài toán quá tải thông tin: người dùng khó tìm sản phẩm trong hàng trăm nghìn
  sản phẩm.

### 1.2. Mục tiêu và phạm vi
- Mục tiêu 1: lưu trữ và phân tích dữ liệu hành vi quy mô lớn bằng MongoDB.
- Mục tiêu 2: xây dựng hệ thống gợi ý sản phẩm bằng lọc cộng tác (ALS).
- Mục tiêu 3: trực quan hoá bằng dashboard.
- Phạm vi: dữ liệu tháng 10/2019 của bộ REES46; **không** dùng Spark/Hadoop.

### 1.3. Đối tượng và phương pháp nghiên cứu
- Dữ liệu, công cụ, quy trình thực nghiệm; đánh giá offline bằng tách train/test.

### 1.4. Bố cục báo cáo
- Tóm tắt nội dung từng chương.

## Chương 2. Cơ sở lý thuyết

### 2.1. Big Data
- Đặc trưng 5V (Volume, Velocity, Variety, Veracity, Value); liên hệ với dữ
  liệu hành vi trong đề tài (hàng chục triệu sự kiện mỗi tháng).
- Vì sao dữ liệu này đủ "lớn" để cần công cụ chuyên dụng và chiến lược lấy mẫu.

### 2.2. NoSQL và MongoDB
- So sánh CSDL quan hệ và NoSQL; các loại NoSQL (tài liệu, khoá-giá trị, cột,
  đồ thị).
- Mô hình tài liệu của MongoDB: BSON, collection, document; schema linh hoạt.
- Index trong MongoDB và ảnh hưởng đến truy vấn.
- Lý do chọn MongoDB cho đề tài (tham chiếu `docs/kien-truc.md`, mục 4).
- `[CHÈN HÌNH]` Sơ đồ so sánh mô hình bảng và mô hình tài liệu (tự vẽ).

### 2.3. Aggregation pipeline
- Khái niệm pipeline gồm các stage: `$match`, `$group`, `$sort`, `$limit`,
  `$project`, `$lookup`, `$setWindowFields`, `$out`.
- `allowDiskUse` và giới hạn bộ nhớ của stage.
- So sánh với `GROUP BY` của SQL.
- `[CHÈN HÌNH]` Sơ đồ một pipeline mẫu (ví dụ pipeline tính RFM hai tầng
  `$group`), vẽ các stage nối tiếp nhau.

### 2.4. Sharding
- Khái niệm mở rộng ngang; thành phần: shard, `mongos`, config server.
- Khoá shard: băm (hashed) và theo khoảng (ranged); chọn khoá phù hợp.
- Đề xuất cho dữ liệu của đề tài: `events` theo `{ user_id: "hashed" }`.
- Nêu rõ đồ án chạy trên một nút; sharding là hướng mở rộng.

### 2.5. Hệ thống gợi ý và lọc cộng tác
- Các hướng tiếp cận: dựa trên nội dung, lọc cộng tác, kết hợp.
- Lọc cộng tác dựa trên bộ nhớ (user-based, item-based) và dựa trên mô hình
  (phân rã ma trận).
- Phản hồi tường minh (rating) và **phản hồi ngầm** (lượt xem, giỏ, mua).
- Vấn đề dữ liệu thưa và cold-start.

### 2.6. ALS cho implicit feedback
- Phân rã ma trận: R ≈ X·Yᵀ với X là vector nhân tố người dùng, Y là vector
  nhân tố sản phẩm.
- Mô hình của Hu, Koren, Volinsky (2008): sở thích p_ui (có/không tương tác)
  và độ tin cậy c_ui tăng theo cường độ tương tác; hàm mục tiêu có chính quy
  hoá λ.
- Thuật toán luân phiên: cố định Y giải X, rồi ngược lại; mỗi bước là bài toán
  bình phương tối thiểu có nghiệm đóng.
- Ý nghĩa các tham số dùng trong đề tài: `factors`, `regularization`,
  `alpha`, `iterations`.
- Vì sao chọn ALS: song song hoá tốt, xử lý được ma trận thưa lớn.
- `[CHÈN HÌNH]` Sơ đồ phân rã ma trận user x item thành hai ma trận nhân tố
  (tự vẽ).

### 2.7. Các độ đo đánh giá xếp hạng
- **Precision@K**: tỷ lệ sản phẩm gợi ý trong top-K có trong tập test.
- **MAP@K**: trung bình của Average Precision trên các người dùng.
- **NDCG@K**: chiết khấu theo vị trí; sản phẩm đúng ở thứ hạng cao được điểm
  cao hơn.
- **AUC**: xác suất sản phẩm đúng được xếp cao hơn sản phẩm sai.
- Công thức cho từng độ đo và ví dụ tính tay với K nhỏ.
- Lưu ý: precision trong `implicit.evaluation.ranking_metrics_at_k` chia cho
  `min(K, số sản phẩm test của người dùng)`, khác công thức "số trúng / K"
  (xem `docs/kien-truc.md`, mục 3). Nêu rõ trong báo cáo để người đọc không
  nhầm.
- Baseline "sản phẩm phổ biến nhất" và vai trò của nó.

### 2.8. Phân tích RFM
- Recency, Frequency, Monetary; chấm điểm theo phân vị; các nhóm khách hàng và
  ý nghĩa kinh doanh.

## Chương 3. Dữ liệu và phân tích

### 3.1. Mô tả bộ dữ liệu
- Nguồn: REES46, "eCommerce behavior data from multi category store" (Kaggle),
  file `2019-Oct.csv`.
- Các cột và ý nghĩa: `event_time`, `event_type`, `product_id`, `category_id`,
  `category_code`, `brand`, `price`, `user_id`, `user_session`.
- `[CHÈN BẢNG]` Bảng mô tả các cột (kiểu dữ liệu, ví dụ, tỷ lệ thiếu).
- `[CHÈN BẢNG]` Vài dòng dữ liệu mẫu (xuất từ MongoDB:
  `db.events.find().limit(5)`).

### 3.2. Nạp dữ liệu vào MongoDB
- Đọc theo chunk bằng pandas, vì sao không đọc cả file vào RAM.
- Lấy mẫu theo người dùng (`SAMPLE_PERCENT`): giải thích vì sao giữ trọn lịch
  sử từng người dùng.
- Làm sạch: bỏ dòng thiếu `user_id`/`product_id`, chuyển `event_time`, đổi NaN
  thành null, bỏ `category_id`.
- Nạp bằng `insert_many(ordered=False)`, tạo 4 index.
- `[CHÈN BẢNG]` Thống kê nạp: số dòng đọc, số dòng loại, số tài liệu, thời
  gian (lấy từ đầu ra của `python src/import_data.py`).

### 3.3. Thiết kế cơ sở dữ liệu
- Các collection và quan hệ giữa chúng.
- `[CHÈN HÌNH]` Sơ đồ luồng dữ liệu (Mermaid trong `docs/kien-truc.md`, mục 1;
  chụp ảnh hoặc xuất sang PNG).
- `[CHÈN BẢNG]` Danh sách collection, số tài liệu, index (xem
  `docs/kien-truc.md`, mục 2).

### 3.4. Phân tích tổng quan
- Các chỉ số: tổng sự kiện, người dùng, sản phẩm, phiên, doanh thu.
- `[CHÈN HÌNH]` Hàng thẻ số liệu tổng quan - **Dashboard, Trang 1**, đầu trang.

### 3.5. Phễu chuyển đổi
- Số view, cart, purchase và các tỷ lệ chuyển đổi; nhận xét và lưu ý phễu đếm
  theo sự kiện.
- `[CHÈN HÌNH]` Biểu đồ phễu - **Dashboard, Trang 1**, mục "Phễu chuyển đổi".

### 3.6. Danh mục, thương hiệu và sản phẩm nổi bật
- Top 20 theo doanh thu và theo lượt mua; nhận xét sự khác nhau giữa hai tiêu
  chí.
- `[CHÈN HÌNH]` Top 20 danh mục và top 20 thương hiệu - **Dashboard, Trang 1**,
  mục "Top danh mục, thương hiệu và sản phẩm"; chụp cả hai lựa chọn "Doanh
  thu" và "Lượt mua".
- `[CHÈN BẢNG]` Top 20 sản phẩm - **Dashboard, Trang 1** (bảng cuối mục).

### 3.7. Hành vi theo thời gian
- Giờ cao điểm trong ngày, sự khác biệt giữa các thứ trong tuần; lưu ý giờ
  UTC.
- `[CHÈN HÌNH]` Số sự kiện và doanh thu theo giờ - **Dashboard, Trang 1**, mục
  "Hành vi theo thời gian", tab "Theo giờ trong ngày".
- `[CHÈN HÌNH]` Số sự kiện và doanh thu theo thứ - cùng mục, tab "Theo thứ
  trong tuần".

### 3.8. Phân nhóm khách hàng RFM
- Cách tính R, F, M; cách chấm điểm phân vị; quy tắc phân 5 nhóm.
- Nhận xét: nhóm nào đóng góp nhiều doanh thu nhất, quy mô từng nhóm, gợi ý
  hành động kinh doanh cho từng nhóm.
- `[CHÈN HÌNH]` Số khách và doanh thu theo nhóm - **Dashboard, Trang 1**, mục
  "Phân nhóm khách hàng (RFM)".
- `[CHÈN BẢNG]` Bảng tổng hợp RFM (số khách, % khách, doanh thu, % doanh thu,
  Recency/Frequency/Monetary trung bình) - cùng mục (collection
  `stats_rfm_summary`).

## Chương 4. Xây dựng hệ thống gợi ý

### 4.1. Kiến trúc tổng thể
- Pipeline 4 bước và vai trò MongoDB / Python.
- `[CHÈN HÌNH]` Sơ đồ kiến trúc (cùng sơ đồ luồng dữ liệu ở mục 3.3).

### 4.2. Xây dựng bảng tương tác
- Bảng `products` và `interactions` tạo bằng `$group` + `$out`.
- Điểm tương tác: view 1, cart 3, purchase 5; lý do chọn các trọng số.
- Ngưỡng `MIN_USER_INTERACTIONS` và số người dùng bị loại.
- `[CHÈN BẢNG]` Ví dụ vài tài liệu `interactions` và `products`.

### 4.3. Ma trận thưa và mã hoá chỉ số
- Mã hoá `user_id`, `product_id` thành chỉ số liên tục; ma trận CSR user x
  item; mật độ ma trận.
- `[CHÈN BẢNG]` Kích thước ma trận, số tương tác, mật độ (lấy từ trang
  "Đánh giá mô hình" - **Dashboard, Trang 4**, mục "Tham số và dữ liệu huấn
  luyện").

### 4.4. Huấn luyện mô hình ALS
- Tham số: `factors=64`, `regularization=0.05`, `alpha=1.0`, `iterations=20`.
- Quy trình: đánh giá bằng tách 80/20, sau đó huấn luyện lại trên toàn bộ dữ
  liệu.
- Thời gian huấn luyện.

### 4.5. Phương pháp đánh giá
- Tách ngẫu nhiên 80/20 các tương tác (`train_test_split`, `random_state`
  cố định).
- Đo Precision, MAP, NDCG, AUC tại K = 10 bằng `ranking_metrics_at_k`.
- Baseline "sản phẩm phổ biến nhất" tính từ tập train, đo cùng cách.
- Nêu rõ giới hạn: đánh giá offline, tách ngẫu nhiên (không theo thời gian).

### 4.6. Sinh gợi ý và các bảng kết quả
- `recommendations`: top-10 cho mọi người dùng, theo lô 50.000, loại sản phẩm
  đã tương tác.
- `similar_items`: top-10 sản phẩm tương tự (độ tương tự cosine) cho sản phẩm
  có từ 5 tương tác.
- `popular_products`: top-50 cho cold-start.
- `[CHÈN BẢNG]` Ví dụ một tài liệu của mỗi collection (xem
  `docs/kien-truc.md`, mục 2).

### 4.7. Dashboard
- Cấu trúc 4 trang; dùng `st.cache_resource` cho kết nối và `st.cache_data`
  cho truy vấn; đọc các collection đã tính sẵn nên phản hồi nhanh.
- Chọn biểu đồ phù hợp từng loại dữ liệu; mỗi biểu đồ có bảng dữ liệu đi kèm.

## Chương 5. Kết quả và demo

### 5.1. Môi trường thực nghiệm
- Cấu hình máy (CPU, RAM, hệ điều hành), phiên bản Python/MongoDB/thư viện,
  giá trị `SAMPLE_PERCENT` đã dùng, thời gian từng bước của pipeline (đầu ra
  của `python src/run_pipeline.py`).
- `[CHÈN BẢNG]` Bảng thời gian từng bước: nạp, phân tích, bảng tương tác, huấn
  luyện.

### 5.2. Kết quả đánh giá mô hình
- So sánh ALS và baseline theo Precision, MAP, NDCG, AUC @10; nhận xét mức cải
  thiện (số lần).
- `[CHÈN HÌNH]` Biểu đồ cột so sánh ALS và baseline - **Dashboard, Trang 4**.
- `[CHÈN BẢNG]` Bảng chỉ số kèm cột "ALS / Baseline (lần)" - **Dashboard, Trang
  4**.
- Thử nghiệm tham số: chạy lại với nhiều giá trị `--factors`, `--iterations`,
  `--regularization`, `--alpha`; mỗi lần chạy được lưu trong `model_metrics`.
- `[CHÈN BẢNG]` Bảng so sánh các bộ tham số - **Dashboard, Trang 4**, mục "Lịch
  sử các lần huấn luyện".
- Phân tích: nhận xét độ chênh lệch giữa các độ đo (ví dụ AUC trong top-K có
  thể chênh ít hơn Precision/NDCG) và ảnh hưởng của việc loại người dùng ít
  tương tác.

### 5.3. Demo gợi ý cho người dùng
- Kịch bản 1: người dùng có lịch sử - so sánh 20 tương tác gần nhất với 10 sản
  phẩm được gợi ý, nhận xét độ phù hợp.
- Kịch bản 2: người dùng mới - hiển thị sản phẩm phổ biến kèm ghi chú
  cold-start.
- `[CHÈN HÌNH]` Trang gợi ý với một người dùng có lịch sử - **Dashboard, Trang
  2**.
- `[CHÈN HÌNH]` Trang gợi ý với `user_id` không tồn tại (cold-start) -
  **Dashboard, Trang 2**.

### 5.4. Demo sản phẩm tương tự
- Chọn vài sản phẩm, nhận xét các sản phẩm tương tự có cùng danh mục/phân
  khúc giá hay không.
- `[CHÈN HÌNH]` Trang sản phẩm tương tự - **Dashboard, Trang 3**.

### 5.5. Thảo luận
- Ưu điểm: pipeline chạy hoàn toàn trên MongoDB và Python, tái lập được
  (`--seed`); aggregation giảm khối lượng dữ liệu kéo về Python.
- Hạn chế: đánh giá offline; ALS cần ma trận vừa RAM; phễu tính theo sự kiện;
  điểm tương tác là tổng thô; dữ liệu một tháng.

> **Lưu ý khi viết chương này.** Dữ liệu giả lập (`scripts/generate_sample.py`)
> có tỷ lệ chuyển đổi cao hơn thật và cấu trúc sở thích rất rõ, nên chỉ số trên
> dữ liệu mẫu **cao bất thường** (ALS hơn baseline hơn 10 lần) và không đại diện
> cho dữ liệu thật. Báo cáo phải dùng số liệu chạy trên `2019-Oct.csv`; có thể
> nhắc dữ liệu mẫu như một phép kiểm thử pipeline.

## Chương 6. Kết luận và hướng phát triển

### 6.1. Kết quả đạt được
- Đối chiếu từng mục tiêu ở chương 1 với kết quả thực tế.

### 6.2. Hạn chế
- Nêu thẳng các hạn chế ở mục 5.5.

### 6.3. Hướng phát triển
- Đánh giá theo thời gian (train các ngày đầu, test các ngày cuối).
- Chuẩn hoá điểm tương tác (log, chặn trần); thử mô hình khác (BPR, LightFM)
  hoặc kết hợp đặc trưng nội dung (danh mục, thương hiệu, giá).
- Phễu theo phiên và theo người dùng duy nhất; phân tích theo cohort.
- Triển khai sharding trên cụm MongoDB; huấn luyện phân tán khi dữ liệu vượt
  RAM một máy.
- Gợi ý theo thời gian thực và A/B testing để đo tác động kinh doanh.

## Tài liệu tham khảo

- Hu, Y., Koren, Y., Volinsky, C. (2008). *Collaborative Filtering for Implicit
  Feedback Datasets*. IEEE International Conference on Data Mining (ICDM).
- Koren, Y., Bell, R., Volinsky, C. (2009). *Matrix Factorization Techniques for
  Recommender Systems*. IEEE Computer, 42(8).
- Tài liệu MongoDB: Aggregation Pipeline, Indexes, Sharding.
- Tài liệu thư viện `implicit` (Ben Frederickson): ALS và
  `implicit.evaluation`.
- Kechinov, M. *eCommerce behavior data from multi category store* (REES46),
  Kaggle.

## Phụ lục

- **Phụ lục A.** Hướng dẫn cài đặt và chạy (tham chiếu `README.md`).
- **Phụ lục B.** Mã nguồn các module chính (hoặc đường dẫn kho mã GitHub).
- **Phụ lục C.** Schema đầy đủ của các collection (`docs/kien-truc.md`).
- **Phụ lục D.** Đầu ra đầy đủ của `python src/run_pipeline.py` trên dữ liệu
  thật.

---

## Danh mục hình và bảng cần chuẩn bị

| Hình / bảng | Nguồn lấy |
|---|---|
| Sơ đồ luồng dữ liệu | `docs/kien-truc.md`, mục 1 |
| Thẻ số liệu tổng quan, phễu | Dashboard, Trang 1 |
| Top danh mục, thương hiệu, sản phẩm | Dashboard, Trang 1 |
| Hành vi theo giờ và theo thứ | Dashboard, Trang 1 |
| Phân nhóm RFM + bảng tổng hợp | Dashboard, Trang 1 |
| Gợi ý cho người dùng (có lịch sử và cold-start) | Dashboard, Trang 2 |
| Sản phẩm tương tự | Dashboard, Trang 3 |
| So sánh ALS và baseline, tham số, lịch sử huấn luyện | Dashboard, Trang 4 |
| Thời gian từng bước pipeline | Đầu ra `python src/run_pipeline.py` |
| Thống kê nạp dữ liệu | Đầu ra `python src/import_data.py` |

Ảnh chụp màn hình nên đặt trong `docs/images/` (đã có sẵn thư mục) và đặt tên
rõ ràng, ví dụ `tong-quan.png`, `pheu.png`, `goi-y-nguoi-dung.png`,
`danh-gia-mo-hinh.png`.
