# Bai-tap-cuoi-ky-Nhom-19
Xây dựng chatbot để trả lời câu hỏi từ dữ liệu theo chủ đề được thu thập bởi nhóm
🎓 Chatbot Hỗ Trợ Giáo Dục – Đại Học Cần Thơ
🌟 Giới thiệu

Dự án Chatbot Giáo Dục được phát triển bởi Nhóm 19 – Trường Đại học Cần Thơ, với mục tiêu xây dựng một hệ thống trò chuyện tự động thông minh, có khả năng:

Hỗ trợ sinh viên tra cứu thông tin học vụ (điểm, đăng ký học phần, thời khóa biểu, lịch thi, học phí, v.v.)

Cung cấp kiến thức giáo dục, tư vấn ngành học, và giải đáp thắc mắc về quy định – chính sách của trường.

Mở rộng khả năng tư vấn học tập bằng AI và xử lý ngôn ngữ tự nhiên (NLP).

🧠 Tính năng nổi bật

✅ Tra cứu thông tin học vụ:
Chatbot có thể cung cấp nhanh các đường dẫn chính thức đến:

Cổng xem điểm: https://qldt.ctu.edu.vn/

Đăng ký học phần: https://dkmh.ctu.edu.vn/

Lịch học, lịch thi: https://thisinh.ctu.edu.vn/

✅ Hỗ trợ học tập:
Chatbot trả lời các câu hỏi về:

Quy chế đào tạo, tín chỉ, học bổng.

Thông tin ngành học, chương trình đào tạo.

Liên hệ khoa – phòng ban của trường.

✅ Tích hợp dữ liệu mở rộng:
Sử dụng file kien_thuc_giao_duc.txt làm kho tri thức nội bộ giúp chatbot phản hồi chính xác và tự nhiên hơn.

✅ Khả năng mở rộng:
Hệ thống có thể tích hợp với:

API ngôn ngữ tự nhiên (LangChain, OpenAI, HuggingFace...)

Giao diện Web hoặc Telegram Bot để sinh viên dễ sử dụng.

🛠️ Cấu trúc thư mục
chatbot_giao_duc/
│
├── data/
│   ├── kien_thuc_giao_duc.txt      # Tập tin chứa kiến thức giáo dục & thông tin trường
│
├── app.py                          # File chính để chạy chatbot
├── requirements.txt                # Thư viện cần thiết
├── README.md                       # Mô tả dự án
└── utils/                          # Các hàm hỗ trợ NLP, xử lý dữ liệu, v.v.

🚀 Cách chạy chatbot (local)
1️⃣ Cài đặt thư viện
pip install -r requirements.txt

2️⃣ Chạy chatbot
python app.py

3️⃣ Truy cập

Chatbot chạy trên http://127.0.0.1:5000 (hoặc cổng bạn định nghĩa trong app)

Giao tiếp bằng tiếng Việt tự nhiên.

👩‍💻 Thành viên nhóm 19
Họ và Tên	MSSV	Vai trò
Nguyễn Duy Mạnh Hoàng	M5...	Trưởng nhóm, xử lý dữ liệu & tích hợp
Trầm Thanh Phú	M5125021	Thiết kế & tối ưu hội thoại
Thế Anh
Thạnh
...	...	...
🌐 Liên hệ & Tài nguyên

Website trường: https://www.ctu.edu.vn

Cổng sinh viên: https://qldt.ctu.edu.vn

Đăng ký học phần: https://dkmh.ctu.edu.vn

Thông tin tuyển sinh: https://tuyensinh.ctu.edu.vn

❤️ Ghi chú

Dự án được xây dựng nhằm mục đích học tập và nghiên cứu.
Mọi thông tin học vụ đều được lấy từ nguồn chính thức của Trường Đại học Cần Thơ.
