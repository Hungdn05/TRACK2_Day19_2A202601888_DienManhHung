# Reflection — Lab 19

**Tên:** Điền Mạnh Hùng
**Cohort:** A20-K2
**Path đã chạy:** lite

---

## Câu hỏi (≤ 200 chữ)

> Trên golden set 50 queries, mode nào thắng ở loại query nào (`exact` /
> `paraphrase` / `mixed`), và tại sao? Khi nào bạn **không** dùng hybrid
> (i.e. khi nào pure BM25 hoặc pure vector là lựa chọn đúng)?

**Trả lời:**

Trên golden set 50 queries, **Hybrid (RRF=60) thắng trung bình** với 78.6% Precision@10, cao hơn cả BM25 (77.8%) và Semantic (73.2%).

- **`exact` queries (15 queries):** BM25 và Hybrid hòa nhau (96.7%). Keyword signal đã đủ mạnh vì từ verbatim xuất hiện trong docs. Hybrid không cải thiện vì không có paraphrased component.

- **`paraphrase` queries (15 queries):** Tất cả đều yếu (24-33%). Bge-small-en-v1.5 là English-trained model, yếu trên Vietnamese paraphrases. Hybrid không giúp được nhiều.

- **`mixed` queries (20 queries):** **Hybrid thắng rõ rệt (100.0%)** so với BM25 (97.0%) và Semantic (98.5%). Đây là pattern production thực tế nhất vì user thật ít khi viết 100% exact hoặc 100% paraphrase.

**Khi không dùng hybrid:**
1. Query có từ verbatim rõ ràng + corpus đồng nhất → BM25 đủ, hybrid thêm overhead không đáng.
2. Tài nguyên cực kỳ hạn chế (embedded device, latency budget < 5ms) → chỉ dùng BM25.
3. Corpus đa ngôn ngữ với embedding model không hỗ trợ tốt → pure BM25 hoặc chọn embedding model phù hợp trước.

---

## Điều ngạc nhiên nhất khi làm lab này

Điều ngạc nhiên nhất là **semantic cache có thể rò rỉ dữ liệu giữa các tenant** nếu không đặt `namespaced=True`. Đây không phải bug code mà là architectural oversight — hệ thống coi đó là "cache hit thành công" và không có exception hay log nào cảnh báo. Lỗi này thuộc OWASP LLM08:2025 và có thể xảy ra trong production mà không ai phát hiện.

---

## Bonus challenge

- [x] Đã làm bonus (NB5-NB8 nâng cao)
- [ ] Pair work với: _(nếu có)_
