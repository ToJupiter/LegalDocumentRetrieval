# 🇻🇳 Akasuki - SoICT Hackathon 2024: Tìm kiếm văn bản pháp luật

[![Status](https://img.shields.io/badge/Status-In%20Progress-green)](https://shields.io/)
[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)

**Kaggle Notebook:** [Kaggle](https://www.kaggle.com/code/hctingnht/ldr24-soict)

Tại thời điểm hiện tại, các thành phần cần thiết cho việc chạy notebook - dataset, các file Input/Output đã đều có trong Kaggle. Tài liệu hướng dẫn chi tiết sẽ có thể được viết trong tương lai.

## ✨ **Ý tưởng thiết kế**

**🎯 Nhiệm vụ:** Tìm kiếm văn bản pháp luật liên quan đến một truy vấn trong kho dữ liệu cho trước.

**🤝 Đội thi:** Akasuki

### 🚀 **Hướng giải quyết**

Sử dụng hai mô hình ngôn ngữ tiếng Việt mã nguồn mở:

1. **`bkai-foundation-models/vietnamese-bi-encoder` (Bi-encoder)** 🤖
2. **`namdp-ptit/ViRanker` (Cross-encoder)** ⚙️

### 🛠️ **Phần 1: Fine-tune mô hình Bi-encoder**

*   **📝 Tiền xử lý và nhúng vector:**
    *   Sử dụng `bi-encoder` để tạo vector embedding cho các truy vấn và văn bản trong `corpus` và `training data`.
    🔤➡️🔢

*   **🎯 Fine-tuning:**
    *   Xây dựng các cặp `(truy vấn, văn bản)` đối lập:
        *   **✅ Cặp đúng (positive):** Lấy từ `training data`.
        *   **❌ Cặp sai (negative):** Chọn từ những cặp không có trong `training data` và có độ tương đồng `cosine` thấp.
          ➕➖
    *   Fine-tune `bi-encoder` với hàm loss `MultipleNegativesRankingLoss`.
        🏋️‍♂️

### 🔍 **Phần 2: Dự đoán với Public Test**

*   **📊 Vector embedding truy vấn:**
    *   Sử dụng `bi-encoder` đã `fine-tune` để nhúng vector cho các truy vấn trong `public test`.
        🔤➡️🔢

*   **🧲 Chọn ra 50 ứng viên có cosine similarity tốt nhất để re-rank:**
    *   Tính `cosine similarity` giữa vector truy vấn và vector văn bản trong `corpus`.
    *   Chọn 50 ứng viên tiềm năng nhất.
        ↔️

*   **🥇 Re-ranking:**
    *   Dùng `cross-encoder` (`namdp-ptit/ViRanker`) để đánh giá lại và sắp xếp thứ hạng 50 ứng viên.
        🏆

*   **🏆 Kết quả:**
    *   Chọn 10 ứng viên có điểm cao nhất từ bước `re-ranking`.
        🔟

---
## 🌟 Nhóm chúng mình

| Role                          | Contributor(s)                 |
| ----------------------------- | ------------------------------ |
| **💻 Thiết kế mô hình** |   [Phan Hoang Hai](https://github.com/ToJupiter), [Dang Phuong Nam](https://github.com/fdv45fs) |
| **🗄️ Thực hiện**                 | [Phan Hoang Hai](https://github.com/ToJupiter)|

Xin cảm ơn và rất mong được ghi nhận đóng góp của các bạn 😊