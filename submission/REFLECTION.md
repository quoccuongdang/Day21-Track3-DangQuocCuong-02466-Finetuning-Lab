# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Điều làm tôi ngạc nhiên nhất là run `attn_only` (r=283) lại đạt training loss thấp hơn cả bản chuẩn `correct` (0.0531 < 0.0549), nhưng khi đánh giá trên tác vụ thực tế thì điểm target lại kém hơn rõ rệt (0.800 so với 0.860). Trước đây tôi thường tin rằng "loss càng thấp chứng tỏ mô hình học càng tốt", nhưng thí nghiệm công bằng này chứng minh loss huấn luyện thấp có thể chỉ là sự ghi nhớ máy móc (memorization), hoàn toàn đi ngược lại năng lực khái quát hóa trên tác vụ mục tiêu.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Tôi mất nhiều thời gian nhất ở khâu kiểm chứng chat template và loss mask (NB1) cùng việc phân tích sự sụt giảm ở nhóm hồi quy (regression gate ở NB5). Ban đầu tôi dự đoán mình sẽ mất thời gian nhất vào việc tinh chỉnh siêu tham số LoRA (chọn rank $r$, chọn alpha $\alpha$), nhưng thực tế cấu hình "vùng không hối tiếc" (all-linear, $\alpha=2r$, LR $\approx 10\times$) đã được lý thuyết định hướng rất rõ ràng. Thời gian thực sự tốn kém là việc bảo đảm dữ liệu đưa vào loss không bị nuốt prompt và hiểu tại sao mô hình lại bị quên năng lực tổng quát.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước lab này, tôi từng tin hai điều sai lầm phổ biến:
- Thứ nhất: "Tăng rank LoRA càng cao thì model càng thông minh và mạnh mẽ". Giờ tôi hiểu rằng vị trí gắn adapter (phủ toàn bộ text-linear) quan trọng hơn gấp bội so với rank. Rank 16 ở đủ các tầng tuyến tính vượt trội hơn hẳn rank 283 chỉ nằm ở attention.
- Thứ hai: "Fine-tune luôn đánh bại prompt engineering". Thực tế đo đạc cho thấy baseline (b) với prompt được thiết kế tối ưu đã đạt tới 0.760 target và 1.000 format. Nếu không có bộ đo 4 nhóm nghiêm ngặt và baseline đóng băng trước, người ta rất dễ tự lừa mình rằng bản fine-tune đã thắng một mốc sàn yếu ớt.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi sử dụng AI assistant để hỗ trợ phân tích cấu trúc pipeline, rà soát các hàm tính toán tham số ma trận LoRA (`matched_rank`), và viết mã tự động hóa kiểm tra tính nhất quán giữa các artifact đo lường. Chỗ AI thường mắc bẫy hoặc gợi ý sai là:
- Mặc định áp dụng `bf16=True` trong cấu hình huấn luyện mà không nhận biết rằng GPU T4 là kiến trúc Turing không hỗ trợ phần cứng cho bfloat16 natively, dẫn đến lỗi kernel hoặc suy giảm tốc độ.
- Có xu hướng đề xuất nới lỏng ngưỡng đánh giá của cổng hồi quy (`tolerance`) khi thấy kết quả FAILED, thay vì giữ nguyên tính liêm chính học thuật và giải thích hiện tượng catastrophic forgetting theo đúng bản chất khoa học.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Bước đầu tiên tôi làm không phải là mở notebook để train model, mà là **đóng băng một tập đánh giá độc lập (Eval Set) và xây dựng một prompt baseline thật mạnh mẽ (Baseline B)**. Tôi sẽ đo đạc xem baseline prompt đó đạt độ chính xác bao nhiêu và chi phí độ trễ ra sao. Chỉ khi prompt engineering chạm ngưỡng giới hạn (về latency, context length hoặc độ phức tạp hành vi), tôi mới tiến hành fine-tuning; và khi fine-tuning, bước tiên quyết là kiểm tra loss mask ngược (`decode_supervised`) cùng việc trộn 3-5% replay data để bảo toàn năng lực tổng quát của mô hình.
