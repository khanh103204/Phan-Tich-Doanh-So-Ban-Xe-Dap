# 🚴 NỀN TẢNG PHÂN TÍCH DOANH SỐ DOANH NGHIỆP  
## Kiến Trúc Big Data Phân Tích & Dự Báo Doanh Thu

---

## 1. Tóm Tắt Điều Hành (Executive Summary)

Dự án xây dựng một nền tảng phân tích dữ liệu lớn (Big Data Analytics Platform) trên hệ sinh thái Hadoop nhằm xử lý, phân tích và dự báo doanh số bán xe đạp trong môi trường phân tán.

Hệ thống được thiết kế theo tư duy kiến trúc dữ liệu doanh nghiệp (Enterprise Data Platform), mô phỏng cách các tổ chức tài chính hoặc ngân hàng xử lý dữ liệu giao dịch quy mô lớn để:

- Lưu trữ dữ liệu trên hệ thống phân tán
- Xây dựng data warehouse phục vụ phân tích
- Xử lý dữ liệu hiệu năng cao
- Triển khai mô hình Machine Learning
- Hỗ trợ ra quyết định dựa trên dữ liệu

Công nghệ sử dụng:

- Apache Hadoop  
- HDFS  
- Apache Hive  
- Apache Spark  
- Spark MLlib  
- RStudio  

---

## 2. Bối Cảnh Nghiệp Vụ

Trong môi trường doanh nghiệp (ngân hàng, bán lẻ, thương mại điện tử), việc quản lý dữ liệu giao dịch đóng vai trò quan trọng trong:

- Theo dõi hiệu suất doanh thu
- Phân tích lợi nhuận theo khu vực
- Phân khúc khách hàng
- Dự báo doanh thu tương lai

Dự án tái hiện bài toán đó với dữ liệu doanh số bán xe đạp.

Các câu hỏi kinh doanh đặt ra:

1. Khu vực nào mang lại doanh thu cao nhất?
2. Nhóm khách hàng nào đóng góp nhiều nhất vào tổng doanh thu?
3. Có thể dự báo chính xác doanh thu tháng tiếp theo không?
4. Phân cụm khách hàng giúp tối ưu chiến lược kinh doanh như thế nào?

---

## 3. Kiến Trúc Hệ Thống

### Luồng xử lý dữ liệu tổng thể

Raw Transaction Data  
        ↓  
HDFS (Tầng lưu trữ phân tán)  
        ↓  
Hive Data Warehouse (Tầng phân tích SQL)  
        ↓  
Spark Processing & Machine Learning  
        ↓  
Visualization & Insight (RStudio)  

---

## 4. Thiết Kế Kiến Trúc Theo Tầng

### 4.1 Tầng Lưu Trữ Dữ Liệu (Storage Layer)

- Lưu trữ dữ liệu thô trên HDFS
- Đảm bảo khả năng mở rộng ngang (Horizontal Scalability)
- Cơ chế chịu lỗi (Fault Tolerance)

---

### 4.2 Tầng Data Warehouse (Hive)

- Thiết kế schema phục vụ phân tích
- Xây dựng các truy vấn SQL:
  - Doanh thu theo tháng
  - Doanh thu theo khu vực
  - Doanh thu theo phân khúc khách hàng
- Mô phỏng mô hình Data Mart trong môi trường ngân hàng

---

### 4.3 Tầng Xử Lý Phân Tán (Spark)

- Làm sạch và tiền xử lý dữ liệu
- Chuẩn hóa dữ liệu
- Feature engineering
- Tính toán phân tán với hiệu năng cao

---

### 4.4 Tầng Phân Tích Nâng Cao (Advanced Analytics)

#### 🔹 Phân Cụm Khách Hàng – K-Means

- Xác định nhóm khách hàng giá trị cao
- Nhóm khách hàng trung bình
- Nhóm khách hàng tiềm năng tăng trưởng

#### 🔹 Dự Báo Doanh Thu – Hồi Quy

- Huấn luyện mô hình dự báo doanh thu tháng tiếp theo
- Đánh giá độ chính xác mô hình
- Ứng dụng cho lập kế hoạch tài chính

---

## 5. Khả Năng Phân Tích

### 5.1 Revenue Intelligence

- Phân tích xu hướng doanh thu theo thời gian
- So sánh hiệu suất theo khu vực
- Phân tích đóng góp theo phân khúc

### 5.2 Customer Intelligence

- Phân cụm khách hàng dựa trên hành vi chi tiêu
- Xác định nhóm chiến lược
- Hỗ trợ tối ưu chiến dịch marketing

### 5.3 Predictive Analytics

- Dự báo doanh thu
- Đánh giá hiệu năng mô hình
- Hỗ trợ lập kế hoạch kinh doanh

---

## 6. Trực Quan Hóa & Báo Cáo

Thực hiện trên RStudio:

- Biểu đồ cột (Bar Chart)
- Biểu đồ đường (Line Chart)
- Heatmap phân bố hiệu suất
- Trực quan hóa kết quả phân cụm

Mục tiêu: mô phỏng dashboard phân tích cấp quản lý trong môi trường doanh nghiệp.

---

## 7. Công Nghệ Sử Dụng

| Thành phần | Công nghệ |
|------------|------------|
| Distributed Storage | HDFS |
| Data Warehouse | Apache Hive |
| Processing Engine | Apache Spark |
| Machine Learning | Spark MLlib |
| Visualization | RStudio |
| Ngôn ngữ | Python, SQL, R |

---

## 8. Kết Quả Đạt Được

✔ Xây dựng thành công pipeline Big Data trên môi trường Hadoop  
✔ Xử lý dữ liệu lớn trong môi trường phân tán  
✔ Xây dựng mô hình phân cụm khách hàng bằng K-Means  
✔ Triển khai mô hình hồi quy dự báo doanh thu  
✔ Mô phỏng kiến trúc phân tích dữ liệu cấp doanh nghiệp  

Tổng doanh thu cao nhất của 10 sản phẩm

![image](https://github.com/user-attachments/assets/141b3ff9-8f0c-4312-a111-6a88980c877c)


10 sản phẩm bán chạy nhất

![image](https://github.com/user-attachments/assets/b748a214-9673-4658-ba8a-8c1ebb242327)



---

## 9. Giá Trị Kỹ Thuật & Học Thuật

- Hiểu và triển khai hệ sinh thái Hadoop
- Thiết kế kiến trúc dữ liệu phân tầng
- Ứng dụng Machine Learning trong môi trường Big Data
- Xây dựng hệ thống phân tích theo chuẩn doanh nghiệp

---

## 10. Tác Giả

Đặng Văn Khánh 
Data Engineer Intern  

GitHub:  
https://github.com/khanh103204/Phan-Tich-Doanh-So-Ban-Xe-Dap



