# PixelPingu

# Tham gia cuộc thi **HCMUS-CTF 2025** cùng nhóm **YusatoCTF**

Chúng tôi – nhóm **YusatoCTF** – vừa tham gia cuộc thi **HCMUS-CTF 2025**, trong đó có một thử thách thú vị yêu cầu phân tích và khai thác mã nguồn kèm theo. Các thành viên trong nhóm đã cùng nhau phối hợp để giải bài này với các bước như sau:

## 🔍 Phân tích đề bài

Bài này tập trung vào việc khai thác từ **2 file chính** và **2 model** có sẵn trong source code:

- `app.py`
- `judge.py`
- Model 1 (được import hoặc load từ source)
- Model 2 (tương tự)

## 🧠 Prompt đề bài

> Phân tích 2 file `app.py`, `judge.py` và 2 model

## 📂 Các bước thực hiện

1. **Phân tích mã nguồn**:  
   Nhóm đã đọc kỹ và phân tích logic hoạt động của `app.py` và `judge.py`, xác định các chức năng chính, điểm đầu vào/ra, và cách mà các model được sử dụng.

2. **Viết script kiểm tra model**:  
   Nhằm hiểu rõ cấu trúc, input/output và hành vi của hai model, chúng tôi đã tạo một đoạn script riêng để kiểm tra lần lượt từng model. Việc này giúp nhận diện định dạng dữ liệu mà server mong đợi cũng như cách model phản hồi.

3. **Tạo script tự động kết nối và khai thác**:  
   Sau khi hiểu được toàn bộ flow, nhóm đã viết một script Python có chức năng tự động kết nối tới server, thực hiện thao tác gửi input và nhận output một cách tuần tự nhằm "vẽ chim cánh cụt" – đúng theo yêu cầu của đề bài.

---

💡 File này được tạo ra nhằm tổng hợp quá trình làm bài, cũng như chia sẻ lại kinh nghiệm và cách tiếp cận của nhóm **HCMUS-CTF 2025** cho những ai quan tâm đến thử thách này.

👉 [Google Drive - Mã nguồn giải mã](https://drive.google.com/file/d/1HRUMaou3gSlXAgIX2WUUNLNnlzylTDIr/view?usp=sharing)

