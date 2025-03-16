# Cài đặt các thư viện nếu chưa có
install.packages("sparklyr")
install.packages("dplyr")
install.packages("ggplot2")
install.packages("forecast")
install.packages("rpart")
install.packages("rpart.plot")
install.packages("xgboost")
install.packages("Matrix")
install.packages("F:/Big_Data/xgboost_1.7.8.1.zip", repos = NULL, type = "win.binary")

# Nạp các thư viện
library(sparklyr)
library(dplyr)
library(ggplot2)
library(forecast)
library(rpart)
library(rpart.plot)
library(xgboost)
library(Matrix)

# Kết nối Spark
sc <- spark_connect(master = "local")


# Thiết lập đường dẫn Spark
Sys.setenv(SPARK_HOME = "C:/spark-3.5.4-bin-hadoop3")


# Tạo kết nối Spark
sc <- spark_connect(
  master = "local",
  spark_home = Sys.getenv("SPARK_HOME")
)

# Kiểm tra kết nối
print(sc)

# Đọc dữ liệu
df <- read.csv("F:/Big_Data/BTL_Big_Data/Sales.csv/Sales.csv")


# Chuyển đổi sang Spark DataFrame
df_spark <- sdf_copy_to(sc, df, overwrite = TRUE)

# Kiểm tra giá trị NA
df_spark %>%
  summarise(across(everything(), ~sum(as.integer(is.na(.))))) %>%
  collect()

# Loại bỏ trùng lặp
df_cleaned_spark <- df_spark %>% distinct()

library(dplyr)
library(dbplyr)

df_cleaned_spark <- df_cleaned_spark %>%
  mutate(Date = to_date(Date),  # Chuyển sang kiểu Date
         Year = year(Date))     # Trích xuất năm từ Date

# Tính tổng doanh thu theo năm
sales_by_year_spark <- df_cleaned_spark %>%
  group_by(Year) %>%
  summarise(Total_Revenue = sum(Revenue)) %>%
  arrange(Year)
# Chuyển đổi về R DataFrame
sales_by_year <- collect(sales_by_year_spark) 
# Hiển thị kết quả
print("Tổng doanh thu theo năm:")
print(sales_by_year)


# Vẽ biểu đồ doanh thu theo năm
ggplot(sales_by_year, aes(x = Year, y = Total_Revenue)) +
  geom_line(color = "blue", size = 1) +
  geom_point(color = "red", size = 2) +
  labs(title = "Xu hướng doanh số theo năm", x = "Năm", y = "Tổng doanh thu") +
  theme_minimal()

# Tính top 10 sản phẩm theo doanh thu
top_products_by_revenue_spark <- df_cleaned_spark %>%
  group_by(Product) %>%
  summarise(Total_Revenue = sum(Revenue)) %>%
  arrange(desc(Total_Revenue))
# Chuyển đổi về R DataFrame
top_products_by_revenue <- collect(top_products_by_revenue_spark) %>% head(10)
# Hiển thị kết quả
print("Top 10 sản phẩm có doanh thu cao nhất:")
print(top_products_by_revenue)
# Vẽ biểu đồ
ggplot(top_products_by_revenue, aes(x = reorder(Product, Total_Revenue), y = Total_Revenue)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 10 sản phẩm có doanh thu cao nhất", x = "Sản phẩm", y = "Tổng doanh thu") +
  theme_minimal()




# Tính top 10 sản phẩm theo số lượng bán
top_products_by_quantity_spark <- df_cleaned_spark %>%
  group_by(Product) %>%
  summarise(Total_Quantity = sum(Order_Quantity)) %>%
  arrange(desc(Total_Quantity))
# Chuyển đổi về R DataFrame
top_products_by_quantity <- collect(top_products_by_quantity_spark) %>% head(10)
# Hiển thị kết quả
print("Top 10 sản phẩm bán chạy nhất:")
print(top_products_by_quantity)
# Vẽ biểu đồ
ggplot(top_products_by_quantity, aes(x = reorder(Product, Total_Quantity), y = Total_Quantity)) +
  geom_bar(stat = "identity", fill = "orange") +
  coord_flip() +
  labs(title = "Top 10 sản phẩm bán chạy nhất", x = "Sản phẩm", y = "Số lượng bán") +
  theme_minimal()




# 1️⃣ Decision Tree để phân tích doanh thu
df_decision_tree <- sales_by_year
# Huấn luyện mô hình Decision Tree
model_tree <- rpart(Total_Revenue ~ Year, data = df_decision_tree, method = "anova")
# Tạo tập dữ liệu dự báo (năm 2025-2028)
future_years <- data.frame(Year = seq(2025, 2028, by = 1))
# Dự đoán doanh thu bằng Decision Tree
future_years$Predicted_Revenue <- predict(model_tree, future_years)
# Vẽ biểu đồ Decision Tree dự báo doanh thu
ggplot() +
  geom_line(data = sales_by_year, aes(x = Year, y = Total_Revenue), color = "blue", size = 1) + 
  geom_point(data = sales_by_year, aes(x = Year, y = Total_Revenue), color = "red", size = 2) +
  geom_line(data = future_years, aes(x = Year, y = Predicted_Revenue), color = "orange", size = 1, linetype = "dashed") +
  geom_point(data = future_years, aes(x = Year, y = Predicted_Revenue), color = "brown", size = 2) +
  labs(title = "Dự báo doanh thu với Decision Tree", x = "Năm", y = "Tổng doanh thu") +
  theme_minimal()
# độ chính xác Dự đoán doanh thu trên tập dữ liệu hiện có
pred_tree <- predict(model_tree, df_decision_tree)
# Tính RMSE (Root Mean Squared Error)
rmse_tree <- sqrt(mean((df_decision_tree$Total_Revenue - pred_tree)^2))
# Tính MAE (Mean Absolute Error)
mae_tree <- mean(abs(df_decision_tree$Total_Revenue - pred_tree))
# Tính MAPE (Mean Absolute Percentage Error)
mape_tree <- mean(abs((df_decision_tree$Total_Revenue - pred_tree) / df_decision_tree$Total_Revenue)) * 100
# Hiển thị kết quả
print(paste("Decision Tree - RMSE:", round(rmse_tree, 2)))
print(paste("Decision Tree - MAE:", round(mae_tree, 2)))
print(paste("Decision Tree - MAPE:", round(mape_tree, 2), "%"))



# 2️⃣ Dự báo doanh thu bằng XGBoost
# Chuyển đổi dữ liệu cho XGBoost
train_data <- as.matrix(sales_by_year$Year)  # Định dạng ma trận
train_label <- sales_by_year$Total_Revenue

# Kiểm tra dữ liệu
print(head(train_data))
print(head(train_label))

# Tạo ma trận XGBoost
dtrain <- xgb.DMatrix(data = train_data, label = train_label)

# Huấn luyện mô hình XGBoost với tham số tối ưu
model_xgb <- xgboost(
  data = dtrain, 
  objective = "reg:squarederror", 
  max_depth = 3, 
  eta = 0.1, 
  nrounds = 100,
  verbose = 0
)

# Dự báo doanh thu cho các năm 2025-2028
years_future <- as.matrix(seq(2025, 2028, by = 1))  # Định dạng đúng
predicted_revenue <- predict(model_xgb, years_future)

# Chuyển dữ liệu thành DataFrame
forecast_df <- data.frame(Year = seq(2025, 2028, by = 1), Revenue = predicted_revenue)

# Vẽ biểu đồ dự báo với XGBoost
ggplot() +
  geom_line(data = sales_by_year, aes(x = Year, y = Total_Revenue), color = "blue", size = 1) + 
  geom_point(data = sales_by_year, aes(x = Year, y = Total_Revenue), color = "red", size = 2) +
  geom_line(data = forecast_df, aes(x = Year, y = Revenue), color = "green", size = 1, linetype = "dashed") +
  geom_point(data = forecast_df, aes(x = Year, y = Revenue), color = "purple", size = 2) +
  labs(title = "Dự báo doanh thu với XGBoost", x = "Năm", y = "Tổng doanh thu") +
  theme_minimal()


# 3️⃣ Dự báo doanh thu bằng ARIMA
sales_ts <- ts(sales_by_year$Total_Revenue, start = min(sales_by_year$Year), frequency = 1)

if (all(sales_ts > 0)) {
  model_arima <- auto.arima(sales_ts)
  forecast_arima <- forecast(model_arima, h = 3)
  
  # Vẽ biểu đồ dự báo ARIMA
  autoplot(forecast_arima) +
    labs(title = "Dự báo doanh thu với ARIMA", x = "Năm", y = "Tổng doanh thu") +
    theme_minimal()
} else {
  print("Dữ liệu không phù hợp cho ARIMA do chứa giá trị âm hoặc bằng 0.")
}


# Dự đoán trên tập dữ liệu hiện tại
arima_fitted <- fitted(model_arima)
# Tính RMSE (Root Mean Squared Error)
rmse_arima <- sqrt(mean((sales_by_year$Total_Revenue - arima_fitted)^2))
# Tính MAE (Mean Absolute Error)
mae_arima <- mean(abs(sales_by_year$Total_Revenue - arima_fitted))
# Tính MAPE (Mean Absolute Percentage Error)
mape_arima <- mean(abs((sales_by_year$Total_Revenue - arima_fitted) / sales_by_year$Total_Revenue)) * 100
# Hiển thị kết quả
print(paste("ARIMA - RMSE:", round(rmse_arima, 2)))
print(paste("ARIMA - MAE:", round(mae_arima, 2)))
print(paste("ARIMA - MAPE:", round(mape_arima, 2), "%"))


# 4️⃣ Phân tích yếu tố quan trọng trong XGBoost
importance_matrix <- xgb.importance(model = model_xgb)
xgb.plot.importance(importance_matrix)

# Ngắt kết nối Spark
spark_disconnect(sc)

