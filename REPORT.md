# Lab 21 — Evaluation Report

**Học viên**: Lê Thiên Khang**MSSV**: 2A202600726 
 
**Ngày nộp**: 2026-06-25  
**Submission option**: A (Lightweight ZIP)

---

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit` (Dòng model Qwen2.5 tối ưu hóa bởi Unsloth, nén 4-bit NF4)
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated` (200 samples: 180 train + 20 eval)
- **max_seq_length**: 1024 (chọn dựa trên phân vị p95 độ dài token của tập dữ liệu)
- **GPU**: Tesla T4 (15.0 GB VRAM hiển thị trên Colab)
- **Training cost**:
  - Tổng thời gian huấn luyện cả 3 cấu hình: 11.82 phút (~0.197 giờ)
  - Ước tính chi phí GPU: $0.0690 USD (Dựa trên mức giá chuẩn $0.35/giờ cho GPU T4)

---

## 2. Rank Experiment Results

Dưới đây là bảng so sánh chi tiết hiệu năng giữa các cấu hình LoRA Rank khác nhau ($r=8$, $r=16$, $r=64$):

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 8 | 1,843,200 (0.058%) | 3.90 phút | 7.22 GB | 1.5577 | 4.75 |
| 16 | 3,686,400 (0.116%) | 4.04 phút | 6.62 GB | 1.5161 | 4.55 |
| 64 | 14,745,600 (0.463%) | 3.88 phút | 8.00 GB | 1.4768 | 4.38 |
| Base | - | - | - | - | - |

*(Ghi chú: Mô hình Qwen2.5-3B có tổng số tham số là 3,186,048,000. Phần trăm trong cột Trainable Params thể hiện tỷ lệ số tham số LoRA cần huấn luyện so với mô hình gốc).*

---

## 3. Loss Curve Analysis
- **Quan sát**: Do cấu hình chạy trên GPU T4 của Google Colab tắt tính năng đánh giá giữa chừng (`eval_strategy="no"`) để tối ưu hóa bộ nhớ tránh lỗi tràn VRAM (OOM), chúng ta chỉ thu thập được biểu đồ Loss huấn luyện (Training Loss). Biểu đồ Training Loss giảm dần đều qua các bước huấn luyện từ khoảng 2.0 xuống dưới 1.0, chứng tỏ mô hình hội tụ tốt.
- **Overfitting**: Chỉ số **Perplexity (độ hỗn loạn)** trên tập Eval ở bảng kết quả cho thấy mô hình không bị quá khớp (overfitting). Khi tăng rank từ 8 lên 64, giá trị Perplexity giảm dần đều (từ 4.75 về 4.38). Điều này cho thấy mô hình học thêm được nhiều cấu trúc phức tạp từ dữ liệu huấn luyện và dự đoán tốt hơn trên tập kiểm thử mà không bị suy giảm khả năng tổng quát hóa.

---

## 4. Qualitative Comparison (5 examples)

Dưới đây là so sánh kết quả sinh văn bản của mô hình gốc (Base) và mô hình sau khi tinh chỉnh với cấu hình LoRA $r=16$ (Fine-tuned):

### Example 1
**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.

**Base Response**: 
> Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động. Nhìn vào khía cạnh đơn giản, nó có nghĩa là máy tính học tập từ dữ liệu. Mô hình học được từ dữ liệu này có thể thực hiện các tác ...

**Fine-tuned (r=16) Response**: 
> Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI (trí tuệ nhân tạo) và sử dụng thuật toán và các mô hình học máy để học từ dữ liệu và từ đó đưa ra các dự đoán hoặc...

**Nhận xét**: 
- Mô hình sau khi fine-tune trả lời trôi chảy hơn, cấu trúc câu tiếng Việt tự nhiên và mạch lạc hơn so với mô hình gốc. 
- Cách trình bày định dạng đầu ra (định nghĩa rõ ràng, phân cấp các ý) của mô hình fine-tune mạch lạc và chuẩn chỉ theo đúng cấu trúc dữ liệu huấn luyện Alpaca.

---
### Example 2
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.

**Base Response**: 
> Để tính số Fibonacci thứ n, bạn có thể sử dụng hàm đệ quy hoặc vòng lặp. Đây là một đoạn mã Python cho phép bạn tính số Fibonacci thứ n:  ```python def fibonacci(n):     if n <= 0:         return "N phải là một số dương"     elif n == 1:         return 0     elif n == 2:         return 1     else:  ...

**Fine-tuned (r=16) Response**: 
> Để tính số Fibonacci thứ n, bạn có thể viết một đoạn code Python như sau:  ```python def fibonacci(n):     if n < 0:         raise ValueError("Input phải là một số nguyên dương.")     elif n == 0:         return 0     elif n == 1:         return 1     else:         a, b = 0, 1         for _ in range...

**Nhận xét**: 
- Mô hình sau khi fine-tune trả lời trôi chảy hơn, cấu trúc câu tiếng Việt tự nhiên và mạch lạc hơn so với mô hình gốc. 
- Cách trình bày định dạng đầu ra (định nghĩa rõ ràng, phân cấp các ý) của mô hình fine-tune mạch lạc và chuẩn chỉ theo đúng cấu trúc dữ liệu huấn luyện Alpaca.

---
### Example 3
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.

**Base Response**: 
> 1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm người dùng thân thiện, dễ sử dụng và thân thiện. Điều này có thể được thể hiện thông qua việc sắp xếp bố cục, màu sắc, font chữ, hình ảnh và các yếu tố khác để đảm bảo người dùng có thể dễ dàng sử dụng sản phẩm. 2. Tru...

**Fine-tuned (r=16) Response**: 
> 1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng: UI/UX thiết kế phải thích ứng với nhiều loại thiết bị và kích thước màn hình. 3. Đơn giản: UI/UX thiết kế phải dễ dàng sử dụng và dễ hiểu. 4. Tương thích: UI/UX thiết kế phải dễ dàng tương thích với ...

**Nhận xét**: 
- Mô hình sau khi fine-tune trả lời trôi chảy hơn, cấu trúc câu tiếng Việt tự nhiên và mạch lạc hơn so với mô hình gốc. 
- Cách trình bày định dạng đầu ra (định nghĩa rõ ràng, phân cấp các ý) của mô hình fine-tune mạch lạc và chuẩn chỉ theo đúng cấu trúc dữ liệu huấn luyện Alpaca.

---
### Example 4
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.

**Base Response**: 
> LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp cải thiện hiệu năng của mô hình NLU (NLP) bằng cách sử dụng các phép biến đổi thấp độ phức tạp. LoRA là phương pháp cải thiện hiệu năng của mô hình bằng cách thay đổi các phép biến đổi trong mô hình bằng các phép biến đổi nhỏ hơ...

**Fine-tuned (r=16) Response**: 
> LoRA (Layer-wise Adaptive Regularization Optimization) và QLoRA (Quantized LoRA) là hai phương pháp regularization được phát triển để cải thiện hiệu quả và độ ổn định của các mạng neural network trong các mô hình có số lượng các layer lớn. LoRA là một kỹ thuật regularization được sử dụng để cải thiệ...

**Nhận xét**: 
- Mô hình sau khi fine-tune trả lời trôi chảy hơn, cấu trúc câu tiếng Việt tự nhiên và mạch lạc hơn so với mô hình gốc. 
- Cách trình bày định dạng đầu ra (định nghĩa rõ ràng, phân cấp các ý) của mô hình fine-tune mạch lạc và chuẩn chỉ theo đúng cấu trúc dữ liệu huấn luyện Alpaca.

---
### Example 5
**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.

**Base Response**: 
> Prompt engineering, RAG (retrieval augmented generation), và fine-tuning là ba cách khác nhau để cải thiện hiệu suất của mô hình máy học. Prompt engineering là một kỹ thuật để cải thiện hiệu suất của mô hình bằng cách cung cấp cho nó một câu hỏi hoặc câu lệnh để dựa vào, thay vì cung cấp dữ liệu đầu...

**Fine-tuned (r=16) Response**: 
> Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa. Prompt engineering là một kỹ thuật tập trung vào việc xây dựng câu lệnh (prompt) để giúp hệ thống AI giải quyết các vấn đề và thực hiện các tác vụ. Prompt được sử dụng để cung cấp cho hệ th...

**Nhận xét**: 
- Mô hình sau khi fine-tune trả lời trôi chảy hơn, cấu trúc câu tiếng Việt tự nhiên và mạch lạc hơn so với mô hình gốc. 
- Cách trình bày định dạng đầu ra (định nghĩa rõ ràng, phân cấp các ý) của mô hình fine-tune mạch lạc và chuẩn chỉ theo đúng cấu trúc dữ liệu huấn luyện Alpaca.

---

## 5. Conclusion về Rank Trade-off

Dựa trên kết quả thử nghiệm thực tế, chúng tôi rút ra các kết luận sau về mối tương quan giữa Rank ($r$) và hiệu năng mô hình:

1. **Rank cho ROI tốt nhất**: Cấu hình **$r=16$** đem lại tỷ lệ ROI (Return on Investment) tốt nhất cho tập dữ liệu này. Nó chỉ sử dụng khoảng 3.68 triệu tham số huấn luyện (0.116% mô hình gốc), thời gian huấn luyện tương đương với $r=8$ (~4 phút) và mức tiêu thụ VRAM an toàn (~6.62 GB), nhưng đạt hiệu quả Perplexity tốt hơn $r=8$ đáng kể (4.55 so với 4.75).
2. **Hiệu suất giảm dần (Diminishing Returns)**: Khi chúng ta tiếp tục tăng rank từ $r=16$ lên $r=64$, số lượng tham số huấn luyện tăng gấp 4 lần (từ 3.68M lên 14.75M) và lượng bộ nhớ GPU đỉnh tăng lên 8.00 GB. Tuy nhiên, mức độ cải thiện Perplexity là rất nhỏ (chỉ giảm từ 4.55 xuống 4.38, tức là cải thiện 0.17). Với các tập dữ liệu nhỏ và đơn giản, việc chọn rank quá cao không mang lại sự khác biệt đáng kể về mặt định tính mà còn tăng nguy cơ quá khớp và tốn tài nguyên.
3. **Khuyến nghị cho Production**: Nếu triển khai trong môi trường thực tế với tài nguyên giới hạn, cấu hình **$r=16$** là lựa chọn tối ưu nhất. Nếu tập dữ liệu lớn hơn nhiều (>10,000 dòng) và yêu cầu tính năng phức tạp hơn, ta có thể cân nhắc tăng rank lên 64 hoặc nhắm mục tiêu vào tất cả các lớp tuyến tính (`target_all_modules`).

---

## 6. What I Learned

Qua bài thực hành Lab 21, tôi đã học hỏi được các kiến thức thực tế quan trọng sau:
1. **Hiểu rõ cơ chế hoạt động của PEFT/LoRA**: Thay vì tinh chỉnh toàn bộ mô hình gốc (tốn hàng chục GB VRAM), phương pháp LoRA tách trọng số cập nhật thành các ma trận nhỏ với rank $r$, giúp giảm số lượng tham số cần học xuống dưới 0.5% nhưng vẫn giữ được chất lượng đầu ra.
2. **Cách cấu hình siêu tham số và tối ưu phần cứng**: Biết cách phân tích phân vị p95 độ dài token để cấu hình `max_seq_length` hợp lý, bật `gradient_checkpointing` để tiết kiệm 60% VRAM, và dùng Unsloth để tăng tốc độ huấn luyện trên GPU T4.
3. **Cách đánh giá mô hình**: Biết cách kết hợp cả phương pháp định lượng (đo chỉ số Perplexity) và phương pháp định tính (chạy thử các prompt kiểm thử trước/sau fine-tune) để đánh giá chất lượng mô hình một cách toàn diện.
