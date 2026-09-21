# Nhóm B - Multi-Turn Evaluation Metrics

## Tổng quan

Nhóm B (Multi-Turn) bao gồm 5 chỉ số chính đánh giá khả năng duy trì ngữ cảnh và theo dõi trạng thái qua nhiều lượt hội thoại.

## Bảng tổng hợp

| Chỉ số | Công thức | File | Dòng |
|---------|-----------|------|------|
| `context_slot_accuracy` | `matched_slots / total_required_slots` | `run_trajectory_eval.py` | 127-164 |
| `coreference_accuracy` | `1.0` nếu match, `0.0` nếu không | `run_trajectory_eval.py` | 167-191 |
| `follow_up_understanding` | `max(0, 0.5*slot + 0.3*coref + 0.2*(1-rep_penalty))` | `run_trajectory_eval.py` | 459-472 |
| `state_tracking_accuracy` | `avg_slot_acc * (1 - avg_contradiction_rate)` | `run_trajectory_eval.py` | 563-566 |
| `cross_turn_consistency` | `1.0 - avg_contradiction_rate` | `run_trajectory_eval.py` | 567 |

---

## Chi tiết từng chỉ số

### 1. context_slot_accuracy

**Mục đích**: Kiểm tra model có nhớ và trích xuất đúng các thông tin quan trọng từ context không.

**File/Method**: `run_trajectory_eval.py`, `DeterministicValidator.evaluate_context_slots()` (dòng 127-164)

**Công thức**:
```
matched_count = Σ (1 cho mỗi slot được match)
total_slots = Số lượng required_slots
context_slot_accuracy = matched_count / total_slots
```

**Ví dụ**:
```python
required_slots = {
    "oil_skin": ["ít dầu", "mềm mịn"],
    "pores": ["lỗ chân lông", "thu nhỏ"],
    "swelling": ["giảm sưng", "bọng mắt"]
}

Model trả lời: "Da tôi đổ ít dầu, mềm mịn và lỗ chân lông được thu nhỏ"

→ oil_skin: ✓ (match "ít dầu", "mềm mịn")
→ pores: ✓ (match "lỗ chân lông", "thu nhỏ")
→ swelling: ✗ (không match)

context_slot_accuracy = 2/3 = 0.667
```

---

### 2. coreference_accuracy

**Mục đích**: Kiểm tra model có hiểu và resolve đại từ/ngôn ngữ tham chiếu không (theo cơ chế Dual-Mode: nhắc trực tiếp hoặc thỏa mãn ngữ cảnh).

**File/Method**: `run_trajectory_eval.py`, `DeterministicValidator.evaluate_coreference()`

**Công thức**:
```
# Chế độ 1: Explicit Mention (Nhắc trực tiếp tên thực thể)
explicit_matched = (any target in expected_targets exists in norm_answer)

# Chế độ 2: Context Grounding (Hiểu ngầm & trả lời đúng ngữ cảnh liên đới)
# Nếu câu hỏi chỉ hỏi về thuộc tính phụ thuộc (được đưa đi đâu, thời gian dệt, v.v.)
# mà model không cần lặp lại tên thực thể nhưng đạt context_slot_accuracy >= 0.5
context_grounded = (not explicit_matched and (slot_acc >= 0.5 or matched_slots_count > 0))

coreference_accuracy = 1.0 if (explicit_matched or context_grounded) else 0.0
```

**Ví dụ**:
```python
# Ví dụ 1 - Explicit Mention:
# Ground Truth: referent_phrase: "nhân vật trên", expected_target: ["nguyễn văn quảng"]
# Model: "Bí thư Nguyễn Văn Quảng đề nghị..."
# → resolution_mode = "explicit_mention", coreference_accuracy = 1.0

# Ví dụ 2 - Context Grounding:
# Ground Truth: referent_phrase: "nguồn gốc đó", expected_target: ["brazil"]
# User: "Sau khi tơ được thu hoạch từ nguồn gốc đó, chúng được đưa đến đâu...?"
# Model: "Các sợi kén tơ thô được mang đến xưởng dệt Perrin – Pháp... gần ba tháng..."
# → Trả lời đúng các slot ngữ cảnh liên đới mà không cần lặp từ "Brazil"
# → resolution_mode = "context_grounded", coreference_accuracy = 1.0
```

---

### 3. follow_up_understanding (có Repetition Penalty)

**Mục đích**: Đánh giá model có hiểu câu hỏi tiếp nối và trả lời không bị lặp ý.

**File**: `run_trajectory_eval.py`, trong `TrajectoryEvaluator.evaluate_scenario()` (dòng 459-472)

**Công thức**:
```python
# Chỉ tính cho turn >= 2 (turn 1 không có follow-up)

slot_score = context_slot_accuracy của turn hiện tại
coref_score = coreference_accuracy (hoặc 1.0 nếu không có coreference)

# --- REPETITION PENALTY ---
if prior_answers is not empty:
    prior = prior_answers[-1].lower()
    current = final_answer.lower()

    prior_words = set(prior.split())
    current_words = set(current.split())

    overlap_count = |prior_words ∩ current_words|
    overlap_ratio = overlap_count / |current_words|

    if overlap_ratio > 0.85:
        rep_penalty = 0.3        # Penalize repetition
    else:
        rep_penalty = 0.0        # No penalty
else:
    rep_penalty = 0.0

# Final score
follow_up_score = max(0.0, 
    0.5 * slot_score + 
    0.3 * coref_score + 
    0.2 * (1.0 - rep_penalty)
)
```

**Ví dụ**:
```python
# Case 1: Lặp ý
Turn 1: "Tơ lụa Hermès đến từ Brazil..."
Turn 2: "Tơ lụa Hermès đến từ Brazil và được dệt tại Pháp..."
→ overlap_ratio = 0.85 → rep_penalty = 0.3
→ follow_up_score bị giảm 0.2 * 0.3 = 0.06 điểm

# Case 2: Không lặp
Turn 2: "Xưởng dệt Perrin nằm ở Pháp, hoạt động 50 năm..."
→ overlap_ratio = 0.30 → rep_penalty = 0.0
→ follow_up_score không bị ảnh hưởng
```

---

### 4. state_tracking_accuracy

**Mục đích**: Kiểm tra model có theo dõi và duy trì trạng thái nhất quán qua các turn không.

**File**: `run_trajectory_eval.py`, trong `TrajectoryEvaluator._aggregate_conversation_metrics()` (dòng 563-566)

**Công thức**:
```python
# Bước 1: Tính contradiction_rate trung bình
contradiction_rates = [claim_eval["contradiction_rate"] for each turn]
avg_contradiction_rate = Σ(contradiction_rates) / total_turns

# Bước 2: Tính context_slot_accuracy trung bình
slot_scores = [slot_eval["context_slot_accuracy"] for each turn]
avg_slot_acc = Σ(slot_scores) / total_turns (nếu có slot)

# Bước 3: State tracking accuracy
state_tracking_accuracy = max(0.0, avg_slot_acc * (1.0 - avg_contradiction_rate))
```

**Ví dụ**:
```python
Turn 1: contradiction_rate = 0.0, slot_acc = 1.0
Turn 2: contradiction_rate = 0.0, slot_acc = 0.8
Turn 3: contradiction_rate = 0.1, slot_acc = 0.9

avg_contradiction_rate = (0.0 + 0.0 + 0.1) / 3 = 0.033
avg_slot_acc = (1.0 + 0.8 + 0.9) / 3 = 0.9
state_tracking_accuracy = 0.9 * (1.0 - 0.033) = 0.87
```

---

### 5. cross_turn_consistency

**Mục đích**: Kiểm tra câu trả lời có nhất quán với chính nó và với các turn trước không.

**File**: `run_trajectory_eval.py`, trong `TrajectoryEvaluator._aggregate_conversation_metrics()` (dòng 567)

**Công thức**:
```python
contradiction_rates = [claim_eval["contradiction_rate"] for each turn]
avg_contradiction_rate = Σ(contradiction_rates) / total_turns

cross_turn_consistency = 1.0 - avg_contradiction_rate
```

**Giải thích**:
- Nếu không có mâu thuẫn (contradiction_rate = 0) → consistency = 1.0
- Nếu có mâu thuẫn cao (contradiction_rate = 0.5) → consistency = 0.5

---

## Các chỉ số phụ trợ

| Chỉ số | Giải thích |
|---------|------------|
| `contradiction_rate` | Tỷ lệ claims mâu thuẫn với evidence (trong claim_evaluation) |
| `grounded_claim_precision` | Tỷ lệ claims được hỗ trợ bởi evidence |
| `unsupported_claim_rate` | Tỷ lệ claims không được hỗ trợ |

---

## Code References

### File: `run_trajectory_eval.py`

```python
# Class: DeterministicValidator (dòng 79-192)
class DeterministicValidator:
    def evaluate_turn_tools(...)      # dòng 83
    def evaluate_context_slots(...)   # dòng 127
    def evaluate_coreference(...)      # dòng 167

# Class: AtomicClaimVerifier (dòng 198-329)
class AtomicClaimVerifier:
    async def verify_claims(...)     # dòng 211
    def _heuristic_fallback(...)    # dòng 313
    def _extract_evidence_quote(...)# dòng 313 (mới thêm)

# Class: TrajectoryEvaluator (dòng 393-618)
class TrajectoryEvaluator:
    async def evaluate_scenario(...)  # dòng 400
    def _aggregate_conversation_metrics(...)  # dòng 505
```
