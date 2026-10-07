# Lab 21 — Evaluation Report

**Họ tên**: Đặng Quốc Cường  **MSSV**: 02466  **Ngày**: 07/10/2026  
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (Google Colab Free)`

> Mọi con số dưới đây khớp chính xác 100% với các file trong thư mục `results/`.

---

## 1. Setup

| Thông số | Giá trị thực tế |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (`intent`, `urgency`, `product`, `sentiment`) |
| Train / val split | 225 / 25 mẫu (tách ngẫu nhiên cố định seed 42) |
| `max_length` | 1024 — p95 đo được thực tế là 98 tokens *(theo results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` (loại bỏ hoàn toàn prompt khỏi loss mask) |
| Epochs / max_steps | 2 epochs / 30 optimizer steps |

**Template có giữ khối `<think>` không?** 
Có — kết quả kiểm tra tại `results/template_check.json` trả về `verdict: "reasoning preserved — safe to train on traces"`. 
Chat template của mô hình giữ nguyên thẻ mở `<think>` và thẻ đóng `</think>`. Tuy nhiên, trên tập dữ liệu mặc định 250 ticket bare JSON, các nhãn câu trả lời không chứa reasoning trace, và thẻ `<think>\n\n</think>` rỗng được sinh ra ngay trong generation prompt tiền tố (`add_generation_prompt=True`), do đó không ảnh hưởng tới đoạn loss được giám sát.

**Lý do chọn cấu hình `max_length`:**
Đo lường phân phối token bằng `token_stats.json` cho thấy: p50 = 93, p95 = 98, max = 101 token. Mức p95 gợi ý `suggested_max_length = 256`. Tuy nhiên, cấu hình phần cứng của Tier T4 được đặt an toàn ở `max_length = 1024` nhằm dự phòng cho các ticket dài bất thường mà không gây OOM khi chạy batch size = 1, gradient accumulation = 16 (effective batch = 16, nằm dưới trần 32 của deck §11.4).

---

## 2. Mask proof (NB1)

| Chỉ số | Giá trị |
|---|---|
| `supervised_fraction` | 0.4149 (41.49% tổng số tokens được tính loss) |
| Câu trả lời nằm trong loss | `True` (Assert Pass) |
| Câu hỏi KHÔNG nằm trong loss | `True` (Assert Pass) |

Dán đoạn preview đầu tiên của chuỗi được tính loss:

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

**Nhận xét phân tích mask:**
Nếu đặt `everything` (Lỗi kinh điển của fine-tuning), `supervised_fraction` sẽ là 100% (94/94 tokens), bao gồm cả system prompt và câu hỏi của người dùng. Khi đó model sẽ bị phạt loss khi không sinh lại câu hỏi của user, dẫn đến hiện tượng sau huấn luyện model nhại lại prompt thay vì trả lời. Bằng chứng giải mã ngược trong `mask_proof.json` khẳng định 100% prompt đã được che bởi `-100` (`IGNORE_INDEX`), và chỉ có nhãn JSON mục tiêu được đưa vào gradient update.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

Bảng đo đạc đóng băng trước khi huấn luyện (kết quả tại `results/baselines_frozen.json`):

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.7244 | 0.000 | 11331.0 |
| (b) base + optimized prompt | 0.760 | 0.7244 | 1.000 | 3775.0 |
| (c) LoRA fine-tune | 0.860 | 0.1556 | 1.000 | 1280.0 |

**(b) có thật sự mạnh hơn (a) không?** 
Có, vượt trội hoàn toàn: target tăng từ 0.000 lên 0.760, format compliance từ 0.000 lên 1.000, và độ trễ giảm từ 11331 ms xuống 3775 ms.
- Với naive prompt (a), base model không có few-shot hoặc định dạng JSON schema rõ ràng nên sinh văn bản hội thoại tự do dài dòng bằng tiếng Việt, không trích xuất đủ 4 trường JSON và đạt điểm format = 0.
- Với optimized prompt (b), cấu trúc prompt chi tiết ép buộc mô hình tuân thủ đúng format JSON, định nghĩa rõ ràng các enum hợp lệ cho `intent`, `urgency`, `sentiment`, giúp mô hình giải quyết tác vụ đạt 76% độ chính xác.
- Mã băm SHA-256 của `OPTIMIZED_PROMPT` là `719e74d3b6232053`, hoàn toàn nguyên bản theo repo gốc, không bị làm yếu đi để tâng bốc bản fine-tune.

---

## 4. Giải phẫu cấu hình sai (NB4)

Bảng đối chứng 4 cấu hình cùng ngân sách 30 optimizer steps (dữ liệu kết hợp từ `results/runs.csv` và `results/autopsy.json`):

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | train s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.0549 | **0.860** | 995.5 | 12.07 |
| `attn_only` | q,v | 283 | 32,456,704 | 1e-4 | **0.0531** | 0.800 | 888.9 | 12.09 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 0.0903 | 0.260 | 1021.3 | 12.08 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.0670 | 0.820 | 1084.7 | **7.15** |

> **Thứ tự xếp hạng thực tế trên tập target:** `correct` (0.860) > `qlora` (0.820) > `attn_only` (0.800) >> `wrong_lr` (0.260).

### Trả lời ba câu hỏi giải phẫu:

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?**  
Trên tập target, `attn_only` **thua** `correct` (0.800 so với 0.860), mặc dù số lượng tham số huấn luyện đã được cân bằng chính xác qua hàm `matched_rank()` (32,456,704 so với 32,464,896, sai lệch chỉ 0.025%). Điều đặc biệt là nếu chỉ nhìn vào `train loss` ở NB4, `attn_only` lại có loss thấp hơn `correct` (0.0531 < 0.0549) và trông như đang "thắng". Đây chính là bằng chứng thực nghiệm rõ ràng nhất cho **Lỗi #3** (đánh giá bằng metric thay thế/train loss thay vì task accuracy): adapter với rank cực cao (r=283) dồn vào 2 module attention dễ dàng ghi nhớ dữ liệu huấn luyện (overfit memorization) nhưng thiếu khả năng khái quát hóa. Vị trí gắn adapter (toàn bộ các lớp linear bao gồm MLP/FFN) chính là đòn bẩy quyết định, vì tri thức biểu diễn và ánh xạ từ khóa nằm phần lớn ở các lớp MLP. Rank cao ở vài lớp không thể bù đắp cho sự thiếu hụt vị trí.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**  
Run `wrong_lr` chỉ đổi duy nhất learning rate từ 1e-4 xuống 1e-5 (chia 10, rơi về thang learning rate của full fine-tuning thông thường). Đường train loss của `wrong_lr` giảm cực kỳ chậm chạp và dừng lại ở 0.0903, khiến điểm target rớt thảm hại về 0.260. Nếu chỉ nhìn loss giảm chậm mà không biết nguyên nhân do LR, một kỹ sư thiếu kinh nghiệm sẽ kết luận sai rằng *"tập dữ liệu này quá khó", "LoRA không đủ dung lượng học bài toán này", hoặc "cần phải tăng rank LoRA lên 64/128"*. Trong thực tế, vì LoRA chỉ cập nhật một phần nhỏ tham số được chiếu qua ma trận rank thấp mà không có gradient tích lũy từ toàn bộ mạng, LoRA bắt buộc phải dùng learning rate lớn hơn khoảng 10 lần so với full fine-tuning (deck §11.3) để các trọng số adapter dịch chuyển đủ nhanh trong không gian biểu diễn.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**  
QLoRA 4-bit (NF4) giảm đỉnh VRAM từ 12.07 GB xuống còn 7.15 GB, tức tiết kiệm **40.76% VRAM** (gần 5 GB VRAM). Tuy nhiên, cái giá phải trả là: thời gian huấn luyện tăng nhẹ từ 995.5s lên 1084.7s do chi phí dequantize on-the-fly, và điểm target giảm từ 0.860 xuống 0.820 (mất 4 điểm phần trăm độ chính xác). Kết quả này hoàn toàn ủng hộ khuyến nghị kỹ thuật từ nhà phát triển Qwen và Unsloth: với kiến trúc hybrid attention và độ nhạy lượng tử hóa của họ model Qwen3.5, sai số lượng tử hóa 4-bit làm suy giảm đáng kể chất lượng suy luận. Trừ khi bị giới hạn cứng về phần cứng (VRAM < 8GB), trên GPU 16GB như T4 ta nên dùng 16-bit LoRA (bf16/fp16) thay vì QLoRA.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.100` · `regression Δ = -0.569` · `valid_trace_rate = 0.00`

### Diễn giải kết quả:
Phán quyết của cổng hồi quy trả về `FAILED` không phải vì mô hình fine-tune học kém tác vụ mục tiêu, mà bởi vì hiện tượng **Quên lãng thảm họa (Catastrophic Forgetting)** nghiêm trọng trên năng lực tổng quát (general capability):
1. **Ở tác vụ mục tiêu (Target task):** Bản fine-tune LoRA đã thực sự vượt qua baseline prompt tối ưu (b) với cách biệt rõ rệt (`0.860` vs `0.760`, `target Δ = +0.100`). Định dạng đạt 100% chuẩn JSON 4 trường, và độ trễ giảm tới 66% (từ 3775 ms xuống 1280 ms), chứng minh rằng fine-tuning đã nén thành công hành vi và schema vào trọng số mô hình.
2. **Ở năng lực tổng quát (Regression probe):** Điểm kiểm tra hồi quy tụt dốc không phanh từ `0.7244` xuống `0.1556` (`regression Δ = -0.5688`, vượt xa ngưỡng sai số cho phép `tolerance = 0.020`). 
3. **Nguyên nhân cốt lõi:** Quá trình huấn luyện chỉ sử dụng duy nhất 225 mẫu câu hỏi ticket CSKH dẫn xuất trực tiếp ra JSON mà không có bất kỳ dữ liệu giữ nhịp nào. Mô hình 4B đã hình thành phản xạ có điều kiện: coi mọi đầu vào người dùng đều là yêu cầu xuất ra JSON phân loại ticket CSKH. Khi nhận các câu hỏi kiến thức thông thường (ví dụ: *"Thủ đô của Việt Nam là gì?"*), mô hình thay vì trả lời tự nhiên lại cố gắng sinh ra JSON `{"intent": "hoi_thong_tin", ...}` hoặc bị rối loạn từ khóa.
4. **Giải pháp khắc phục:** Cần bổ sung ngay **1% đến 5% replay data** (dữ liệu hội thoại tổng quát và tri thức chung) vào tập huấn luyện SFT (theo đúng nguyên lý Deck §6.3 và §14.3) để neo giữ khả năng ngôn ngữ chung của mô hình nền trong khi vẫn học được tác vụ chuyên biệt.

---

## 6. Định tính — bắt buộc có cả ca THUA

Bảng trích xuất 6 trường hợp kiểm thử thực tế từ `results/qualitative.json` trên tập `eval_target.jsonl`:

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp. Shop hỗ trợ tốt. | `doi_tra`, `cao`, `chuột không dây`, `tich_cuc` | `doi_tra`, `trung_binh`, `chuột không dây`, `tich_cuc` | `doi_tra`, `cao`, `chuột không dây`, `tich_cuc` | ✅ **FT thắng**: Nhận diện đúng độ khẩn cấp `cao` từ chữ "Gấp". Baseline bị nhầm thành `trung_binh`. |
| 2 | Shop ơi, mình đặt ốp lưng điện thoại mã đơn VN812931. Hoàn tiền. Sớm nhé. Bực mình. | `hoan_tien`, `trung_binh`, `ốp lưng điện thoại`, `tieu_cuc` | `hoan_tien`, `cao`, `ốp lưng điện thoại`, `tieu_cuc` | `hoan_tien`, `trung_binh`, `ốp lưng điện thoại`, `tieu_cuc` | ✅ **FT thắng**: FT phân biệt chính xác "Sớm nhé" là mức độ `trung_binh`, trong khi baseline b gán nhãn sai thành `cao`. |
| 3 | Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện. Cảm ơn shop nhiều. | `hoan_tien`, `thap`, `bình giữ nhiệt`, `tich_cuc` | `hoi_thong_tin`, `thap`, `bình giữ nhiệt`, `tich_cuc` | `hoan_tien`, `thap`, `bình giữ nhiệt`, `tich_cuc` | ✅ **FT thắng**: Nhận biết cụm từ "Chưa thấy tiền" thể hiện ý định `hoan_tien`, baseline (b) nhầm thành `hoi_thong_tin`. |
| 4 | Xin chào, mình đặt chuột không dây mã đơn DH139158. Bảo hành bao lâu. Không vội. Mình vẫn tin tưởng shop. | `hoi_thong_tin`, `thap`, `chuột không dây`, `tich_cuc` | `hoi_thong_tin`, `thap`, `chuột không dây`, `tich_cuc` | `san_pham_loi`, `thap`, `chuột không dây`, `tich_cuc` | ❌ **FT thua**: FT bị đánh lừa bởi từ "Bảo hành" và suy diễn sai rằng sản phẩm bị lỗi (`san_pham_loi`), trong khi khách chỉ hỏi chính sách. |
| 5 | Shop ơi, mình đặt máy xay sinh tố mã đơn DH777946. Khi nào có tiền về. Mong shop phản hồi. Rất thất vọng. | `hoan_tien`, `trung_binh`, `máy xay sinh tố`, `tieu_cuc` | `hoan_tien`, `trung_binh`, `máy xay sinh tố`, `tieu_cuc` | `van_chuyen`, `trung_binh`, `máy xay sinh tố`, `tieu_cuc` | ❌ **FT thua**: Cụm từ "Khi nào... về" làm FT kích hoạt nhầm nhánh phân loại tiến độ giao hàng (`van_chuyen`) thay vì tiền hoàn lại. |
| 6 | Shop ơi, mình đặt nồi chiên không dầu mã đơn DH249548. Thiếu phụ kiện. Khi nào tiện. Cho tôi hỏi. | `san_pham_loi`, `thap`, `nồi chiên không dầu`, `trung_tinh` | `san_pham_loi`, `thap`, `nồi chiên không dầu`, `trung_tinh` | `hoi_thong_tin`, `thap`, `nồi chiên không dầu`, `trung_tinh` | ❌ **FT thua**: Câu kết "Cho tôi hỏi" lấn át ngữ cảnh chính "Thiếu phụ kiện", khiến mô hình quy về `hoi_thong_tin`. |

### Mẫu chung ở các ca FT thua:
Các ca fine-tune thua đều có một đặc điểm chung: **bẫy từ khóa cục bộ (Local Keyword Over-triggering)**. Do tập dữ liệu huấn luyện nhỏ (225 mẫu), adapter có xu hướng hình thành các liên kết định tuyến quá mạnh giữa các token đơn lẻ và nhãn phân loại (ví dụ: thấy *"bảo hành"* lập tức gắn nhãn `san_pham_loi`; thấy *"khi nào... về"* lập tức gán nhãn `van_chuyen`; thấy *"cho tôi hỏi"* ở cuối câu lập tức gán nhãn `hoi_thong_tin`). Ngược lại, prompt engineering tối ưu ở baseline (b) nhờ vào năng lực suy luận toàn cảnh nguyên bản của mô hình nền 4B nên xử lý các ngữ cảnh giao thoa này ổn định và thấu đáo hơn.

---

## 7. Kết luận & điều tôi học được

### Kết luận:
Dựa trên các kết quả thực nghiệm toàn diện từ NB1 đến NB5, câu trả lời cho câu hỏi *"Có nên deploy bản fine-tune này vào hệ thống production hay không?"* là: **CHƯA NÊN DEPLOY NGAY LẬP TỨC TRONG MÔI TRƯỜNG CHUNG, NHƯNG CÓ THỂ TRIỂN KHAI TRONG MỘT MICROSERVICE ĐỘC LẬP CHUYÊN BIỆT.** 

Về mặt tích cực, bản fine-tune đã chứng minh tính ưu việt rõ rệt trên tác vụ phân loại ticket CSKH so với kỹ thuật prompt engineering tốt nhất: độ chính xác mục tiêu tăng từ 76.0% lên 86.0%, format luôn đảm bảo 100% JSON chuẩn xác tuyệt đối, và thời gian phản hồi giảm gần 3 lần (từ 3.77s xuống 1.28s) giúp tiết kiệm đáng kể chi phí token và điện toán khi phục vụ ở quy mô lớn. 

Tuy nhiên, rào cản chí mạng là mô hình đã hoàn toàn đánh mất năng lực hiểu biết ngôn ngữ tổng quát (tụt từ 72.4% xuống 15.6%). Nếu deploy model này làm chatbot tương tác đa năng trực tiếp với người dùng, hệ thống sẽ gặp sự cố nghiêm trọng khi người dùng hỏi các câu hỏi mở. Do đó, để đưa vào ứng dụng thực tế an toàn, chúng ta có hai hướng đi chiến lược:
1. **Kiến trúc phân tách (Microservice Adapter Routing):** Chỉ nạp adapter này cho riêng pipeline phân loại ngầm (backend triage queue) nơi đầu vào 100% là ticket CSKH, tận dụng tốc độ và độ chính xác cao mà không lo ngại về năng lực tổng quát.
2. **Tái huấn luyện với Replay Regularization:** Bổ sung 3-5% dữ liệu đa nhiệm tổng quát vào tập train SFT trước khi đóng gói model dùng chung.

Đòn bẩy thật sự quyết định thành bại trong lab này không phải là việc tăng rank LoRA (r=283 vẫn thua r=16 khi chỉ gắn vào attention), mà nằm ở **tính đúng đắn của loss mask** (đảm bảo chỉ tính loss trên nhãn câu trả lời), **vị trí gắn adapter bao phủ toàn bộ text-linear**, và **learning rate đặt đúng thang ~10× full-FT**.

### Ba điều tôi học được:
1. **Mask đúng quan trọng hơn thuật toán:** Việc kiểm chứng loss mask bằng cách giải mã ngược (`decode_supervised`) ở NB1 là bước sống còn. Nếu tính loss trên cả prompt (`everything`), mô hình sẽ học cách nhại lại câu hỏi của người dùng và hoàn toàn thất bại trong việc sinh câu trả lời, tiêu tốn hàng giờ huấn luyện vô ích.
2. **Vị trí gắn adapter vượt trội hơn độ lớn của rank:** Thí nghiệm công bằng `attn_only` (r=283) so với `correct` (r=16) với cùng ngân sách tham số ~32.4M đã đập tan ngộ nhận phổ biến rằng "rank càng cao mô hình càng thông minh". Tri thức ngôn ngữ và suy luận phân loại nằm chủ yếu ở các lớp MLP/FFN; gắn adapter vào toàn bộ text-linear ở rank nhỏ mang lại hiệu quả vượt trội so với dồn toàn bộ tham số vào attention modules.
3. **Loss huấn luyện là một chỉ số thay thế nguy hiểm:** Đánh giá mô hình bằng `train loss` thay vì task accuracy là sai lầm chết người. Run `attn_only` đạt train loss thấp nhất (0.0531) nhưng lại thua trên tập đánh giá thực tế (0.800 so với 0.860). Luôn phải đánh giá mô hình bằng bộ tiêu chí 4 nhóm khách quan trên tập eval độc lập.

### Nếu có thêm 2 giờ nữa, tôi sẽ thử:
- Bổ sung 50 mẫu dữ liệu đàm thoại tiếng Việt thông thường (replay buffer) vào tập huấn luyện và đo lại xem điểm regression có giữ vững được trên 0.70 trong khi duy trì điểm target > 0.85 hay không.
- Thử nghiệm kỹ thuật DoRA (Weight-Decomposed Low-Rank Adaptation) để kiểm tra xem việc phân tách độ lớn vector (magnitude) và hướng (direction) có giúp cải thiện độ chính xác ở các ca phân loại ranh giới khó hay không.

---

## Phụ lục — thưởng đã làm

- [x] **B1 NB6 merge + hot-swap (+3 điểm)**: Đã thực hiện kiểm chứng merge trọng số LoRA vào base model tại `results/merge_check.json`. Kết quả: điểm trước merge = 0.8600, sau merge = 0.8600 (Δ = 0.0000, nằm gọn trong ngưỡng an toàn 0.01). Đồng thời xác thực cơ chế hot-swap adapter đa nhiệm trên cùng một base model trong bộ nhớ VRAM.
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub
