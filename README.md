# Sign Language Recognition Project </l>
## 1. Giới thiệu:  
a) Đề tài môn Trí tuệ nhân tạo IT4042, học kỳ 20182  
Giảng viên: TS Thân Quang Khoát
Nhóm: 5  HEDPSI ASK60  
Nguyễn Hoàng Dũng  
Phạm Minh Khang  
Võ Quốc Tuấn  
Nguyễn Duy Ý  
b) Bài toán :  
Nhận diện ngôn ngữ ký hiệu (thủ ngữ).  
Đề tài thực hiện ở mức cơ bản, nhận diện đánh vần chữ cái bằng tay , xử lý ảnh tĩnh với 24 lớp.

## 2. Cấu trúc thư mục:  
* preprocessing.py:  
Thư viện các hàm tiền xử lý  
* train.py:  
Chương trình train CNN Model  
* slr_models:  
Thư mục chưa các model đã train, sử dụng cho realtime_slr  
* realtime_slr.py:  
Chương trình ứng dụng model, nhận diện thời gian thực.  
* data_generate : (ongoing)  
Công cụ sinh dữ liệu để huấn luyện cho v3, ứng dụng vào nhận diện realtime. Đang hoàn thiện ..
## 3. Hướng dẫn cài đặt: 

### 3.1 Cài đặt môi trường:   
    Tạo môi trường ảo mới, gợi ý sử dụng anaconda:  
    conda create -n ai_proj python
    conda activet ai_proj  

    Cài đặt các thư viện cần thiết:  
    pip install -r requirements.txt  

## 3.2 Tài dữ liệu :  
    Dữ liệu hiện tại được lấy từ Dataset Sign Language MNIST trên kaggle 
    [link](https://www.kaggle.com/datamunge/sign-language-mnist)
    Tải về, giải nén, đặt vào thư mục inp/ trong folder chứa code 

## 3.3 Tự train model: 
    Chỉnh sửa các thông số trong train.py cho phù hợp, sau đó thực hiện chạy:  
    python train.py  

## 3.4 Chạy realtime:  
    python realtime_slr.py

    Bấm 'b' đề chụp background (không cho tay vào)  
    Ấn 'r' để reset background  
    Ấn 'ESC' để tắt 