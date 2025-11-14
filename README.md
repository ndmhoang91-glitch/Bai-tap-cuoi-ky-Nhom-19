# 🤖 Chatbot Hỗ Trợ Sinh Viên – Đại Học Cần Thơ

---

## 🌟 Giới thiệu

Dự án **Chatbot Giáo Dục** được phát triển bởi **Nhóm 19**

👩‍💻 Thành viên Nhóm 19

Họ và Tên	MSSV	Vai trò

Nguyễn Đặng Minh Hoàng M5125003 – Xử lý dữ liệu & tích hợp

Trầm Thanh Phú M5125021 - Tối ưu hội thoại & huấn luyện mô hình

Danh Thế Anh M5125001

Nguyễn Minh Thạnh M5125012

Với mục tiêu xây dựng một **trợ lý ảo thân thiện**, giúp sinh viên tra cứu nhanh thông tin học vụ, quy định, và hỗ trợ định hướng học tập.

Chatbot sử dụng **ngôn ngữ tự nhiên (NLP)** và công nghệ **LangChain + Ollama**, cho phép truy xuất thông tin từ cơ sở tri thức tùy chỉnh (`kien_thuc_giao_duc.txt`) và phản hồi chính xác bằng tiếng Việt.

---

## 🧠 Tính năng nổi bật

✅ **Tra cứu thông tin học vụ**
- Xem điểm, lịch học, lịch thi, tín chỉ, học phí.
- Truy cập nhanh:
  - 📘 Cổng xem điểm: [https://qldt.ctu.edu.vn](https://qldt.ctu.edu.vn)
  - 🧾 Đăng ký học phần: [https://dkmh.ctu.edu.vn](https://dkmh.ctu.edu.vn)
  - 📅 Lịch học & thi: [https://thisinh.ctu.edu.vn](https://thisinh.ctu.edu.vn)

✅ **Tư vấn học tập & quy chế**
- Quy định học vụ, bảo lưu, học bổng, xét tốt nghiệp.
- Mẹo học tập, kỹ năng mềm và hướng dẫn tra cứu tài liệu.

✅ **Thông tin hành chính**
- Liên hệ các phòng ban: Đào tạo, CTSV, Ký túc xá, IT Support.
- Tra cứu biểu mẫu hành chính, lịch nghỉ lễ, hỗ trợ kỹ thuật.

✅ **Không trả lời ngoài phạm vi**
Chatbot được giới hạn trong chủ đề **học sinh – sinh viên**, không phản hồi các câu hỏi về thời sự, giải trí, chính trị hoặc công nghệ ngoài học vụ.

---

## 📂 Cấu trúc thư mục dự án

```bash
chatbot_giao_duc/
│
├── data/
│   ├── kien_thuc_giao_duc.txt      # Tập tin chứa kiến thức giáo dục & thông tin trường
│
├── app.py                          # File chính để chạy chatbot
├── requirements.txt                # Thư viện cần thiết
├── README.md                       # Mô tả dự án
└── utils/                          # Các hàm hỗ trợ NLP, xử lý dữ liệu, v.v.
```
🚀 Cách chạy chatbot (local)
1️⃣ Cài đặt thư viện

pip install -r requirements.txt

2️⃣ Chạy chatbot

python app.py

3️⃣ Bắt đầu trò chuyện

Chatbot chạy trên http://127.0.0.1:5000 (hoặc terminal).

Giao tiếp bằng tiếng Việt tự nhiên.

👩‍💻 Thành viên Nhóm 19

Họ và Tên	MSSV	Vai trò

 Hoàng		Trưởng nhóm – xử lý dữ liệu & tích hợp

 Phú		Tối ưu hội thoại & huấn luyện mô hình
...	...	...

🌐 Tài nguyên & Liên hệ

#Nội dung	Liên kết

🌍 Website trường	https://www.ctu.edu.vn

🎓 Cổng sinh viên	https://qldt.ctu.edu.vn

🧾 Đăng ký học phần	https://dkmh.ctu.edu.vn

📚 Thông tin tuyển sinh	https://tuyensinh.ctu.edu.vn

❤️ Ghi chú
Dự án được thực hiện nhằm mục đích học tập và nghiên cứu trong khuôn khổ môn học AI & Ứng dụng.
Mọi thông tin học vụ được lấy từ nguồn chính thức của Trường Đại học Cần Thơ (CTU).
