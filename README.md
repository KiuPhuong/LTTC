# Stock Price Prediction Project

## 📌 Giới thiệu

**Stock Price Prediction** là một project ứng dụng **Machine Learning và Deep Learning** để dự đoán giá cổ phiếu dựa trên dữ liệu giá trong quá khứ.

Project triển khai và so sánh nhiều mô hình dự đoán chuỗi thời gian, bao gồm:

* **K-Nearest Neighbors (KNN)**
* **Long Short-Term Memory (LSTM)**
* **Transformer**

Các mô hình được sử dụng để học các đặc trưng từ dữ liệu lịch sử của cổ phiếu và dự đoán xu hướng/giá trong tương lai.

---

## ⚙️ Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/KiuPhuong/Stock-prediction.git
cd Stock-prediction
```

### 2. Tạo môi trường Python

Sử dụng **Conda** để tạo môi trường:

```bash
conda create -n stock-prediction python=3.10 -y
```

Kích hoạt môi trường:

```bash
conda activate stock-prediction
```

### 3. Cài đặt các thư viện

Cài đặt toàn bộ dependencies từ file `requirements.txt`:

```bash
pip install -r requirements.txt
```

Sau khi cài đặt hoàn tất, môi trường đã sẵn sàng để chạy project.

---

## ▶️ Chạy Project

Mở Jupyter Notebook:

```bash
jupyter notebook
```

Sau đó lựa chọn notebook tương ứng với mô hình muốn chạy:

* `Predicting Stock Prices with KNN.ipynb`
* `Stock Price Prediction Project using TensorFlow with LSTM.ipynb`
* `Transformer stock prediction.ipynb`

Ngoài ra, project còn có thư mục:

```text
Portfolio-Optimization/
```

dành cho phần **Portfolio Optimization**.
