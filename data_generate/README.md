# Hướng dẫn 
Các bước thực hiện :
1. Tạo môi trường mới :  
conda -n ai_proj python
conda activate ai_proj

2. Cài đặt các thư viện cần thiết:  
pip install -r requirements.txt

3. Chạy chương trình chụp ảnh:   
python data_gen.py [nhãn ảnh]
ví dụ :   
Sinh các ảnh cho chữ cái a  
python data_gen.py a

* Đầu tiên để camera máy tính cố định, nhấn chụp phông bằng phím 'b'. Sau khi phông đã chụp, bắt đầu cho tay vào để chụp theo thứ tự cho tới khi đủ 200 cái.  
* Sẽ có 3 màn hình: Màn hình camera hiện tại, màn hình tay được cắt ra sẽ chụp, và màn hình ảnh chụp cuối. Nếu muốn xóa ảnh chụp cuối ấn phím 'r'.
* Ấn chụp phím 'c'
* Thoát ấn phím 'ESC', tất cả ảnh đã chụp sẽ lưu lại.
4. Chụp ảnh:  
Thực hiện thao tác cho tất cả các ký tự   
a b c d e f g h i k l m n o p q r s t u v w x y   
Mở sẵn cái ảnh này vừa xem vừa thao tác
![](images/amer_sign2.png)

![](images/american_sign_language.png)

Ảnh demo thực hiện 
![](images/demo.png)  
# Lưu ý khi chụp ảnh: 
+ Cố gắng để phông đằng sau ít nhòe nhất có thể 
+ File sẽ bị ghi đè nếu trùng tên 
