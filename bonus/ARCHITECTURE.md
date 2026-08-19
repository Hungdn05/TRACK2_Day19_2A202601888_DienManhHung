# Bonus Architecture — AI Personal Assistant with Hybrid Memory

**Contributors:** Điền Mạnh Hùng  
**Date:** 2026-08-19  
**Lab:** Day 19 — Vector Store + Feature Store

---

## Sơ Đồ Kiến Trúc

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           USER QUERY                                         │
│                    "Trợ lý nhớ gì về tôi?"                                  │
└─────────────────────────────┬────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          HYBRIDMEMORYAGENT                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 1. FEATURE STORE LOOKUP (< 10ms)                                    │     │
│  │    user_id → {topic_affinity, reading_speed, preferred_language,     │     │
│  │               queries_last_hour, distinct_topics_24h}               │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 2. VECTOR STORE QUERY (Hybrid: BM25 + Semantic)                    │     │
│  │    Filter: user_id = current_user                                   │     │
│  │    Query: user query → top-K episodic memories                       │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
│                              │                                               │
│                              ▼                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐     │
│  │ 3. CONTEXT ASSEMBLER                                                │     │
│  │    "User likes <topic_affinity> reading at <speed>wpm.               │     │
│  │     Recent activity: <queries_last_hour> queries in 1h.              │     │
│  │     Top memories: <top-3 episodic>"                                 │     │
│  └─────────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────┬────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          LLM RESPONSE                                        │
│              (Personalized answer based on memory)                            │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                             │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────────┐   │
│  │  EPISODIC MEMORY (Qdrant)  │    │  STABLE PROFILE (Feast)            │   │
│  │  ─────────────────────────  │    │  ─────────────────────────────────  │   │
│  │  Collection: memories        │    │  Feature View: user_profile        │   │
│  │  Fields:                     │    │  - topic_affinity                 │   │
│  │  - id (uuid)                 │    │  - reading_speed_wpm              │   │
│  │  - vector (384d)             │    │  - preferred_language             │   │
│  │  - payload:                  │    │                                    │   │
│  │    user_id                   │    │  Feature View: query_velocity     │   │
│  │    content                   │    │  - queries_last_hour              │   │
│  │    timestamp                 │    │  - distinct_topics_24h            │   │
│  │    source                    │    │                                    │   │
│  │  Filter: payload.user_id     │    │  Online Store: SQLite             │   │
│  │  TTL: 90 days (configurable)│    │  TTL: 30 days / 1 hour            │   │
│  └─────────────────────────────┘    └─────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Chunking Strategy — Episodic Memory

### Decision: Semantic Chunking với Overlapping Windows

**Vấn đề:** Làm sao chia conversation/document thành chunks để embed + retrieve hiệu quả?

### Các lựa chọn đã xem xét:

| Strategy | Cách làm | Ưu điểm | Nhược điểm |
|----------|----------|----------|------------|
| **Per-message** | Mỗi message = 1 chunk | Đơn giản, không mất context | Chunk quá ngắn, vector yếu, retrieval noise cao |
| **Per-conversation** | Cả conversation = 1 chunk | Vector mạnh | Chunk quá dài, retrieval không granular, overflow context |
| **Fixed-token** | Chia 512 tokens/chunk | Predictable, đơn giản | Cắt giữa câu, mất semantic coherence |
| **Semantic** | Cắt theo sentence/paragraph boundary | Giữ nguyên semantic unit | Phức tạp hơn, cần NLP |
| **Semantic + Overlap** | Semantic chunk + 20% overlap | Giữ coherence + surface cross-chunk relationships | Double storage cho overlap |

### Quyết định: **Semantic Chunking với 20% Overlapping Windows**

**Tại sao:**

1. **Semantic boundaries** giữ nguyên câu/đoạn hoàn chỉnh → vector representation mạnh hơn vì chunk có complete meaning.

2. **20% overlap** cho phép retrieval trả về chunk chứa context trước/sau nếu query match vào phần overlap → LLM có đủ context để understand.

3. **Tradeoff thực tế:**
   - Storage: ~20% extra (acceptable vì Qdrant storage rẻ)
   - Retrieval quality: ↑ significant cho multi-sentence queries
   - Latency: ≈ same (overlap tính toán nhẹ)

**Implementation:**
```python
def semantic_chunk(text: str, max_tokens: int = 512, overlap: float = 0.2) -> list[str]:
    sentences = split_by_sentence(text)  # dùng underthesea cho tiếng Việt
    chunks, current = [], []
    current_tokens = 0
    
    for sentence in sentences:
        tokens = count_tokens(sentence)
        if current_tokens + tokens > max_tokens:
            chunks.append(" ".join(current))
            # Keep last 20% as overlap
            overlap_size = max(1, int(len(current) * overlap))
            current = current[-overlap_size:] + [sentence]
            current_tokens = sum(count_tokens(s) for s in current)
        else:
            current.append(sentence)
            current_tokens += tokens
    
    if current:
        chunks.append(" ".join(current))
    return chunks
```

---

## 2. Feature Schema — User Profile

### Decision: Hybrid Schema (Tabular + Latent Embedding)

**Vấn đề:** User profile cần những features gì để personalize AI assistant?

### Các lựa chọn đã xem xét:

| Pattern | Features | Ưu điểm | Nhược điểm |
|---------|----------|----------|------------|
| **Tabular only** | topic_affinity, reading_speed, language | Simple, fast lookup | Không capture complex preferences |
| **Embedding only** | user_preference_vector (256d) | Capture nuanced preferences | Opaque, khó debug, storage lớn |
| **Hybrid** | Tabular (structured) + Embedding (latent) | Best of both | Phức tạp hơn |

### Quyết định: **Hybrid Schema**

**Tabular Features (explicit, interpretable):**
```python
user_profile_features = {
    "user_id": str,                    # Entity key
    "topic_affinity": str,             # "cloud,security,ai_ml" — đa topic
    "reading_speed_wpm": int,           # 150-400
    "preferred_language": str,          # "vi" / "en" / "mix"
    "interaction_style": str,           # "concise" / "detailed" / "technical"
    "event_timestamp": datetime,        # For PIT join correctness
}
TTL: 30 days (profile thay đổi chậm)
```

**Latent Embedding (implicit, nuanced):**
```python
# Derived from conversation history bằng simple average của conversation vectors
user_embedding_features = {
    "user_id": str,
    "preference_vector": list[float],   # 384d, updated weekly
    "communication_style_vector": list[float],  # 64d
}
TTL: 7 days (vì preference có thể drift nhanh hơn structured features)
```

**Tại sao hybrid:**
- **Tabular** cho features cần explainability (topic_affinity dùng để filter/search)
- **Embedding** cho features cần nuance (communication style không easy để explicit encode)
- Retrieval: tabular dùng cho pre-filtering, embedding dùng cho re-ranking

**Tradeoff:**
- Storage: +384d vector per user (~1.5KB) — acceptable
- Lookup: 2 lookups thay vì 1 — negligible impact (< 1ms)
- Maintenance: complexity tăng, nhưng manageable với Feast

---

## 3. Freshness Strategy — When Does Memory Update?

### Decision: Tiered Freshness (Streaming + Batch)

**Vấn đề:** User vừa đọc document mới, bao lâu thì "trợ lý nhớ gì về tôi?" phản ánh điều đó?

### Các lựa chọn đã xem xét:

| Strategy | Latency | Cost | Use Case |
|----------|---------|------|----------|
| **Streaming (Push API)** | Sub-second | High (real-time compute) | Fraud detection, stock prices |
| **Micro-batch (5 min)** | 5 minutes | Medium | User activity, trending topics |
| **Daily batch** | 24 hours | Low | Stable features ( demographics) |

### Quyết định: **Tiered Freshness**

**Tier 1 — Real-time (Streaming):**
```python
# Khi user TƯƠNG TÁC (query, click, read):
# → Immediately upsert to Qdrant episodic memory
# → Immediately update query_velocity in Feast online store
# Latency: < 100ms (async)
```

**Tier 2 — Near-real-time (5 min batch):**
```python
# Topic affinity updates, preference vector refresh
# Dùng Feast's streaming feature view hoặc scheduled job
# Latency: 5 minutes acceptable cho preference drift
```

**Tier 3 — Daily batch:**
```python
# User profile refresh (reading_speed recalculation)
# Demographic features
# Latency: 24 hours OK cho stable features
```

**Tại sao tiered:**
- **Episodic memory**: Real-time vì nó là "short-term memory" — user expect immediate recall
- **Query velocity**: Real-time vì nó ảnh hưởng trực tiếp response (fatigue detection)
- **Topic affinity**: Near-real-time (5 min) vì user có thể explore new topic
- **Demographics**: Daily vì stable, không cần real-time

**Tradeoff:**
- Cost: Real-time tốn compute, nhưng cho personal assistant scale thì acceptable
- Complexity: 3 tiers cần orchestration, nhưng Feast handle được
- Consistency: Có thể có eventual consistency issues, cần idempotent operations

---

## Rejected Alternative: Lưu Episodic trong Feature Store

**Đã xem xét:** Dùng Feast embedding feature view để lưu episodic memories

```python
# ❌ REJECTED: Tưởng lưu episodic trong Feature Store
episodic_feature_view = FeatureView(
    name="episodic_features",
    entities=[user],
    ttl=timedelta(days=90),
    schema=[
        Feature(name="recent_memories", dtype=ValueType.FLOAT_LIST),
        Feature(name="conversation_count", dtype=ValueType.INT64),
    ]
)
```

**Lý do reject:**
1. **Re-index cycle khác hẳn:**
   - Episodic: new memory mỗi conversation (~minutes to hours)
   - User profile: update weekly/monthly
   - Nếu combine → phải re-materialize cả user profile khi chỉ có episodic update

2. **Retrieval pattern khác:**
   - Feature Store: point lookup by entity key
   - Vector Store: similarity search by query vector
   - Hybrid use case cần BOTH → không fit trong 1 system

3. **TTL semantics khác:**
   - Episodic: 90 days, nhưng nên có decay (old memories less relevant)
   - Profile: 30 days, absolute freshness matters

4. **Query latency:**
   - Feature Store lookup: < 10ms
   - Vector similarity: 10-50ms
   - Combine = slow feature retrieval = slow LLM response

**Conclusion:** Keep episodic in dedicated vector store (Qdrant), keep stable profile in feature store (Feast). Each system optimized for its use case.

---

## Vietnamese-Context Considerations

### 1. Code-Switching (Vi/En Mix)

**Vấn đề:** User Việt Nam thường mix tiếng Việt và tiếng Anh trong cùng câu.

**Ví dụ:**
- "Hôm nay tôi đọc về Kubernetes deployment"
- "Cho tôi xem cách setup CI/CD pipeline"
- "Kubernetes cluster của tôi bị OOM error"

**Design decisions:**
- **Tokenizer:** Dùng `underthesea` cho tiếng Việt + space-based cho English words
- **Embedding model:** BGE-M3 (multilingual) thay vì BGE-small-en
- **Chunk boundary:** Không cắt giữa code snippet và comment tiếng Việt
- **BM25:** Cần tokenizer hỗ trợ Vietnamese (underthesea.word_tokenize)

### 2. Phonetic Typo Handling

**Vấn đề:** User có thể type sai chính tả, đặc biệt với từ tiếng Anh.

**Ví dụ:**
- "Kubernates" thay vì "Kubernetes"
- "mixels" thay vì "mixels" (không, sorry)
- "depoly" thay vì "deploy"

**Design decisions:**
- **Fuzzy matching:** Dùng BM25 với edit-distance pre-processing
- **Spell correction:** Optional preprocessing step trước khi embed
- **Fallback:** Nếu semantic search miss, thử lại với keyword expansion

### 3. Privacy — Decree 13 / PDPD

**Vấn đề:** Nghị định 13/2023/NĐ-CP về bảo vệ dữ liệu cá nhân.

**Design decisions:**
- **User isolation:** Mỗi user có collection riêng trong Qdrant HOẶC filter by user_id payload
- **Data residency:** Option để store data in Vietnam (Qdrant supports on-prem)
- **Consent tracking:** Lưu ý khi nào user đồng ý cho memory, có audit trail
- **Right to delete:** Implement hard delete trong cả Qdrant và Feast

### 4. Cultural Context — Formality Levels

**Vấn đề:** Tiếng Việt có nhiều mức độ formal/informal.

**Ví dụ:**
- "Cho tôi hỏi" (formal)
- "Cho mình hỏi" (semi-formal)
- "Mình hỏi" (informal)

**Design decisions:**
- **User preference:** Explicit feature `communication_style` (formal/semi/informal)
- **Response tone:** Adjust LLM prompt based on this feature
- **Chunking:** Không cắt trong middle của respectful/formal phrasing

---

## What This POC Doesn't Handle Yet

1. **Multi-device sync:** Memory trên device A không có trên device B
2. **Encryption at rest:** Sensitive memories có thể bị leak nếu disk compromised
3. **Memory editing/deletion:** User không thể edit specific memory, chỉ có delete all
4. **Memory consolidation:** 1000 similar memories không được gộp thành 1 summary
5. **Collaborative memory:** Shared memories giữa multiple users (family assistant)
6. **Priority/explicit memory:** User không thể mark memory là "important"

---

## 1 Prompt Hiệu Quả Nhất

**Prompt cho việc tạo demo queries:**
> "Generate 5 Vietnamese user queries for a personal AI assistant that demonstrates different memory retrieval scenarios: (1) simple factual recall, (2) requires user profile context, (3) requires recent activity, (4) paraphrase query where vector search wins, (5) mixed query where hybrid search wins. Make queries realistic and specific to cloud computing topics."

**1 Prompt Thất Bại:**
> "Write a RAG system" (too vague, AI generates generic boilerplate without understanding the specific Vietnamese context requirements)

---

## Summary

| Decision | Choice | Why |
|----------|--------|-----|
| Chunking | Semantic + 20% overlap | Best retrieval quality vs storage tradeoff |
| Feature Schema | Hybrid (tabular + embedding) | Explainability + nuance |
| Freshness | Tiered (real-time / 5min / daily) | Match feature semantics to update frequency |
| Episodic storage | Qdrant (not Feast) | Different retrieval patterns, re-index cycles |
| Vietnamese support | BGE-M3 + underthesea | Multilingual embedding + proper tokenization |
| Privacy | User isolation + Vietnam data residency option | Decree 13 compliance |
