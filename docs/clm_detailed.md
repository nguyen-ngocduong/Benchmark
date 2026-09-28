# CLM (Contrastive Language Models) - Detailed Documentation

**Model:** CLM-v0.1-8B  
**Organization:** Contrastive-LM  
**HuggingFace:** https://huggingface.co/Contrastive-LM/CLM-v0.1-8B  
**Documentation Date:** 28/09/2026

---

## 📋 Table of Contents

1. [Overview](#1-overview)
2. [Problem Statement](#2-problem-statement)
3. [Architecture](#3-architecture)
4. [Training Pipeline](#4-training-pipeline)
5. [Benchmark Results](#5-benchmark-results)
6. [Usage & API](#6-usage--api)
7. [Fine-tuning](#7-fine-tuning)
8. [Limitations](#8-limitations)
9. [Future Work](#9-future-work)
10. [References](#10-references)

---

## 1. Overview

**CLM-v0.1-8B** (Contrastive Language Model) là một "System One" model được thiết kế cho việc ra quyết định nhanh và có khả năng tổng quát hóa cao.

### Key Characteristics

| Property | Value |
|----------|-------|
| **Model Type** | System One - Fast decision-making |
| **Parameters** | 8B (20M trainable + 8B frozen) |
| **Base Encoder** | Qwen/Qwen3-8B-Base |
| **License** | Apache 2.0 |
| **Downloads (last month)** | 766 |

### Core Innovation

CLM sử dụng **contrastive learning** để kết nối **states** (trạng thái) và **actions** (hành động), cho phép:
- Scoring và ranking candidates hiệu quả
- Fast inference với caching
- Generalization across tasks

---

## 2. Problem Statement

### 2.1 Traditional LLM Limitations

Các large language models truyền thống (System Two models) gặp khó khăn với:

| Problem | Description |
|---------|-------------|
| **Slow inference** | Cần generate token-by-token, tốn thời gian |
| **Expensive** | Chi phí cao cho mỗi inference |
| **Inconsistent tool use** | Khó đạt được consistency trong function calling |
| **Long-horizon tasks** | Khó maintain context qua nhiều steps |

### 2.2 CLM Solution

CLM tiếp cận bằng cách:

```
┌─────────────────────────────────────────────────────────────┐
│                    CLM Approach                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  State (Input)          Action (Output)                     │
│      │                        ▲                            │
│      │                        │                            │
│      ▼                        │                            │
│  ┌────────┐            ┌─────────────┐                     │
│  │ State  │───────────▶│   Action    │                     │
│  │  Head  │   Score    │    Head     │                     │
│  └────────┘            └─────────────┘                      │
│       ▲                                               │    │
│       │           Contrastive Learning                 │    │
│       └───────────────────────────────────────────────┘    │
│                                                             │
│  Objective: Maximize similarity(state, correct_action)       │
│             Minimize similarity(state, wrong_action)        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**Key insight:** Thay vì generate từng token, CLM học cách **score/rank** các candidates đã được định nghĩa trước.

---

## 3. Architecture

### 3.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    CLM-v0.1-8B Architecture                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input: State + Candidates                                  │
│          │                                                  │
│          ▼                                                  │
│   ┌─────────────────────────────────────────────────┐      │
│   │         Frozen Qwen3-8B Encoder                   │      │
│   │  (Last-token-pooled embeddings)                  │      │
│   │                                                 │      │
│   │   Qwen/Qwen3-8B-Base → Qwen/Qwen3-8B          │      │
│   └─────────────────────────────────────────────────┘      │
│          │                                                  │
│          │  Frozen (không train)                          │
│          ▼                                                  │
│   ┌─────────────────────────────────────────────────┐      │
│   │       Trainable Projection Heads (20M params)    │      │
│   │                                                 │      │
│   │   ┌────────────┐      ┌────────────┐           │      │
│   │   │ State Head │      │Action Head │           │      │
│   │   └────────────┘      └────────────┘           │      │
│   └─────────────────────────────────────────────────┘      │
│          │                                                  │
│          ▼                                                  │
│   Output: Score cho mỗi (state, action) pair                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Component Details

#### 3.2.1 Base Encoder (Frozen)

| Property | Value |
|----------|-------|
| **Model** | Qwen/Qwen3-8B-Base → Qwen/Qwen3-8B |
| **Pooling** | Last-token-pooled embeddings |
| **Parameters** | 8B (frozen) |
| **Purpose** | Encode state và action thành vectors |

#### 3.2.2 Projection Heads (Trainable)

| Head | Parameters | Purpose |
|------|------------|---------|
| **State Head** | ~10M | Project state embeddings |
| **Action Head** | ~10M | Project action embeddings |

**Total trainable:** ~20M parameters

#### 3.2.3 Loss Function

```
Bidirectional InfoNCE Loss
```

```
L = -log( exp(sim(s, a_pos) / τ) / Σ exp(sim(s, a_i) / τ) )
```

Trong đó:
- `s` = state embedding
- `a_pos` = positive (correct) action
- `a_i` = all candidates (positive + negative)
- `τ` = temperature
- `sim` = cosine similarity

### 3.3 State & Action Caching

**Key Feature:** States và actions được encode riêng biệt, cho phép:

```
┌─────────────────────────────────────────────────────────────┐
│                    State/Action Caching                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   State 1 ──▶ [Encode] ──▶ s1 (cached)                     │
│                          │                                  │
│                          ▼                                  │
│   Candidate 1 ──▶ [Encode] ──▶ a1 (cached)                 │
│   Candidate 2 ──▶ [Encode] ──▶ a2 (cached)                 │
│   Candidate 3 ──▶ [Encode] ──▶ a3 (cached)                 │
│                                                             │
│   Khi state mới tương tự state 1:                          │
│   → Reuse s1 từ cache!                                     │
│   → Chỉ encode action candidates                           │
│                                                             │
│   Speedup: 2.8× cho repeated states                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. Training Pipeline

CLM được train qua **3 stages** với tổng cộng ~91M samples:

```
┌─────────────────────────────────────────────────────────────┐
│                 CLM Training Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Stage 1: Pre-training (~60M samples)                       │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ Data: Nemotron DQA question-answer pairs            │  │
│  │ Objective: Learn state-action associations          │  │
│  │ Duration: ~5 days on 8×H100                         │  │
│  └─────────────────────────────────────────────────────┘  │
│                          │                                 │
│                          ▼                                 │
│  Stage 2: Mid-training (~30M samples)                       │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ Data: Synthetic hard negatives (Gemini 2.5 Flash) │  │
│  │ Objective: Improve discrimination                  │  │
│  │ Focus: Challenging cases                           │  │
│  └─────────────────────────────────────────────────────┘  │
│                          │                                 │
│                          ▼                                 │
│  Stage 3: Post-training (~1M samples)                      │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ Data: Agentic trajectories                          │  │
│  │   - ADP dataset                                    │  │
│  │   - Endless-Terminals                              │  │
│  │   - LiteCoder-Terminal-SFT                         │  │
│  │ Objective: Task-specific optimization              │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.1 Training Data Details

| Stage | Dataset | Size | Source |
|-------|---------|------|--------|
| Pre-training | Nemotron DQA | ~60M | NVIDIA Nemotron |
| Mid-training | Synthetic hard negatives | ~30M | Gemini 2.5 Flash-Lite |
| Post-training | Agentic trajectories | ~1M | ADP, Endless-Terminals, LiteCoder |

### 4.2 Hard Negative Mining

Mid-training sử dụng **hard negative mining** để cải thiện:

```
┌─────────────────────────────────────────────────────────────┐
│                 Hard Negative Mining                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   State: "Customer hỏi về invoice bị charge 2 lần"        │
│                                                             │
│   Positive: "Transfer to billing department"                │
│                                                             │
│   Hard Negatives (difficult to distinguish):               │
│   ✗ "Transfer to technical support" (semantically close)  │
│   ✗ "This is not my department" (vague)                   │
│   ✗ "Let me transfer you" (incomplete)                    │
│                                                             │
│   Easy Negatives (clearly wrong):                           │
│   ✗ "What is 2+2?" (irrelevant)                           │
│   ✗ "Order a pizza" (unrelated)                           │
│                                                             │
│   → Hard negatives make model discriminate better           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Benchmark Results

### 5.1 Zero-Shot Evaluation

CLM-8B (zero-shot) được đánh giá trên nhiều tasks:

| Benchmark | CLM-8B | Jev | Notes |
|-----------|--------|-----|-------|
| **T-Rex** | On par | Baseline | Generalization |
| **BFCL v4** | On par | Baseline | Function calling |
| **WikiRacing** | On par | Baseline | Multi-hop reasoning |
| **Super Mario** | On par | Baseline | Game playing |

**Key Finding:**
> "CLM-8B performs **on par with Jev** while running **up to 9× faster**"

### 5.2 Agentic Benchmarks (Fine-tuned)

Khi fine-tuned, CLM đạt SOTA trên các benchmarks khó:

| Benchmark | CLM-8B (fine-tuned) | Previous SOTA | Notes |
|-----------|---------------------|--------------|-------|
| **Terminal-Bench 2.1** | **87.6%** | Below pass@1 | SOTA |
| **DeepSWE** | **81.6%** | Below pass@1 | SOTA |

**Key Finding:**
> Jev **fails** (below pass@1) on these long-horizon verification tasks, while CLM achieves SOTA.

### 5.3 Speed Comparisons

| Scenario | CLM-8B | Jev | Speedup |
|----------|--------|-----|---------|
| Zero-shot inference | Fast | Baseline | **9× faster** |
| Fine-tuned verifier (H100) | 4.1–5.7× | Baseline | **4-6× faster** |
| Ranking ~1k candidates | 13× | Baseline | **13× faster** |
| Repeated states | 2.8× | Baseline | **2.8× faster** (caching) |

### 5.4 Benchmark Details

#### Terminal-Bench 2.1
- **Task:** Terminal command completion
- **Setting:** Long-horizon, multi-step
- **CLM Result:** 87.6% (SOTA)
- **Jev Result:** Failed (below pass@1)

#### DeepSWE
- **Task:** Software engineering bug detection
- **Setting:** Requires deep understanding
- **CLM Result:** 81.6% (SOTA)
- **Jev Result:** Failed (below pass@1)

#### BFCL v4 (Berkeley Function Calling Leaderboard)
- **Task:** Function calling / tool use
- **CLM Result:** On par with Jev
- **Latency:** 9× lower

---

## 6. Usage & API

### 6.1 Installation

```bash
pip install contrastive-lm
```

### 6.2 Server Setup

```bash
# 1. Start vLLM server cho base encoder
vllm serve Qwen/Qwen3-8B \
    --served-model-name qwen3-8b \
    --runner pooling \
    --max-model-len 2048 \
    --port 8090 &

# 2. Start CLM API server
clm-serve

# API playground available at: http://localhost:8700/
```

### 6.3 Python API

#### 6.3.1 Basic Usage - System One

```python
from clm import CLMClient, Choice, Noul, Score

# Initialize client
client = CLMClient()

# System One - Typed Questions
r = client.system_one(
    state="Customer: my invoice was charged twice and nobody answers the phone!",
    questions={
        "urgency": Noul(instructions="Is this urgent?"),
        "department": Choice(
            instructions="Which team should handle this?",
            criteria={
                "billing": "Charges, invoices, refunds",
                "technical": "Bugs and outages"
            }
        ),
        "frustration": Score(
            instructions="How frustrated is the customer?",
            criteria=["Calm", "Frustrated", "Very angry"]
        ),
    },
)

# Access results
print(r.department)       # "billing"
print(r.department_prob)  # 0.92
```

#### 6.3.2 Ranking Candidates

```python
from clm import Engine

engine = Engine(emb_url="http://127.0.0.1:8090/v1/embeddings")

results = engine.rank(
    question="What causes tides on Earth?",
    candidates=[
        "The Moon's gravitational pull.",
        "Photosynthesis in plants.",
        "Because the Earth is round."
    ]
)

# Output:
# [{'rank': 1, 'candidate': "The Moon's gravitational pull.", 'prob': 0.93},
#  {'rank': 2, 'candidate': 'Because the Earth is round.', 'prob': 0.05},
#  {'rank': 3, 'candidate': 'Photosynthesis in plants.', 'prob': 0.02}]
```

#### 6.3.3 Function Calling Format

```python
# Convert tools to Choice format
tools = [
    {
        "name": "web_search",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            }
        }
    }
]

# Format as questions
questions = {
    "tool": Choice(
        instructions="Select the appropriate tool to use",
        criteria={tool["name"]: tool["description"] for tool in tools}
    )
}

# Run inference
result = client.system_one(
    state=conversation_history,
    questions=questions
)

# Extract predicted tool
predicted_tool = result.tool  # e.g., "web_search"
```

### 6.4 API Response Format

```python
class CLMResponse:
    """Response từ system_one()"""
    
    # For Choice questions
    choice: str           # Selected option
    choice_prob: float    # Probability of selected choice
    
    # For Noul questions
    noul: str             # Yes/No/Neither
    noul_prob: float      # Confidence
    
    # For Score questions
    score: float          # Numeric score
    score_prob: float     # Confidence
```

---

## 7. Fine-tuning

### 7.1 Why Fine-tune?

| Setting | Use Case | Performance |
|---------|----------|-------------|
| **Zero-shot** | Quick deployment, general tasks | On par with Jev |
| **Fine-tuned** | Specialized tasks, SOTA results | 87.6% Terminal-Bench |

### 7.2 Fine-tuning Process

```bash
# 1. Install CLM from source
git clone https://github.com/Contrastive-LM/CLM.git && cd CLM && pip install -e .

# 2. Download task-specific heads
hf download Contrastive-LM/deepswe-clm-heads-8k heldout_tasks.json --local-dir heads/deepswe

# 3. Fine-tune
python train/finetune.py \
    --task clm \
    --init-ckpt "$(clm-download)" \
    --out-dir runs/deepswe \
    --holdout-tasks heads/deepswe/heldout_tasks.json \
    --batch 512
```

### 7.3 Fine-tuning Requirements

| Resource | Requirement |
|----------|--------------|
| **GPU** | 1× H100 (80GB) or equivalent |
| **Training time** | ~2-4 hours per task |
| **Data** | Task-specific heldout tasks JSON |
| **Trainable params** | Only 20M (projection heads) |

### 7.4 Fine-tuned Variants Available

| Model | Task | HuggingFace |
|-------|------|-------------|
| `deepswe-clm-heads-8k` | Software bug detection | Available |
| `terminal-clm-heads-8k` | Terminal command | Available |
| 7+ quantized variants | Various | Available |

---

## 8. Limitations

### 8.1 Technical Limitations

| Limitation | Description | Mitigation |
|------------|-------------|------------|
| **Encoder-locked** | Must use Qwen3-8B with last-token pooling | Use provided vLLM setup |
| **No generation** | Only scores/reranks candidates | Use with generative model |
| **Relative probabilities** | Scores are relative to candidate set | Ensure comprehensive candidates |
| **Fine-tuning needed** | SOTA requires fine-tuned heads | Fine-tune or use provided variants |

### 8.2 Usage Limitations

```
┌─────────────────────────────────────────────────────────────┐
│                    Usage Constraints                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ❌ Cannot generate new text                               │
│  ❌ Cannot handle unseen actions                           │
│  ❌ Probabilities depend on candidate set quality          │
│                                                             │
│  ✅ Can rank any number of candidates                       │
│  ✅ Very fast inference (cached)                           │
│  ✅ Generalizes across domains                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Future Work

### 9.1 Upcoming Models

| Model | Release | Features |
|-------|---------|----------|
| **CLM-35B** | Early October 2026 | Multimodal, stronger generalization |
| Larger variants | TBD | Higher accuracy |

### 9.2 Roadmap

- [ ] CLM-35B multimodal support
- [ ] Additional fine-tuned checkpoints
- [ ] Python wheel distribution
- [ ] Streaming API support

---

## 10. References

### 10.1 Official Resources

| Resource | Link |
|----------|------|
| **HuggingFace** | https://huggingface.co/Contrastive-LM/CLM-v0.1-8B |
| **GitHub** | https://github.com/Contrastive-LM/CLM |
| **Blog** | https://contrastive-lm.notion.site |
| **Discord** | https://discord.gg/5dAQEDJBs |

### 10.2 Citation

```bibtex
@misc{kwok2026contrastivelanguagemodels,
  title={Contrastive Language Models: A System One Model for Fast and Generalizable Decision-Making},
  author={Jacky Kwok, Hangoo Kang, Tarun Suresh, Jon Saad-Falcon, Marco Pavone, Christopher Ré, Azalia Mirhoseini},
  year={2026},
  note={Notion Blog},
  url={https://contrastive-lm.notion.site}
}
```

### 10.3 Related Models

| Model | Description |
|-------|-------------|
| `Contrastive-LM/CLM-v0.1-8B` | Base zero-shot model |
| `Contrastive-LM/deepswe-clm-heads-8k` | DeepSWE fine-tuned |
| `Contrastive-LM/terminal-clm-heads-8k` | Terminal fine-tuned |
| 7+ quantized variants | Memory-optimized versions |

---

## Appendix A: Quick Start Code

```python
# Complete example: Function Calling
from clm import CLMClient, Choice

client = CLMClient()

# Define tools
tools = {
    "web_search": "Search the web for information",
    "get_weather": "Get current weather for a location",
    "send_email": "Send an email to a recipient"
}

# Conversation state
state = """
System: You are a helpful assistant.
User: What is the weather in Hanoi today?
"""

# Run inference
result = client.system_one(
    state=state,
    questions={
        "tool": Choice(
            instructions="Select the appropriate tool",
            criteria=tools
        )
    }
)

print(f"Predicted tool: {result.tool}")  # "get_weather"
print(f"Probability: {result.tool_prob}")  # 0.95
```

---

## Appendix B: Architecture Diagram (ASCII)

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           CLM System Architecture                        │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   User Input                    CLM Processing                    Output │
│   ──────────                    ──────────────                    ────── │
│                                                                          │
│   ┌──────────┐                   ┌─────────────────┐                    │
│   │  State   │──────────────────▶│  State Encoder  │                    │
│   │ (query)  │                   │  (Qwen3-8B)     │                    │
│   └──────────┘                   └────────┬────────┘                    │
│                                           │                              │
│   ┌──────────┐                            │                              │
│   │Candidates│────────────────────────────┼──────────────────────────┐   │
│   │ (tools)  │                            │                          │   │
│   └──────────┘                   ┌────────▼────────┐                 │   │
│                                   │  Action Encoder │                 │   │
│                                   │  (Qwen3-8B)     │                 │   │
│                                   └────────┬────────┘                 │   │
│                                           │                          │   │
│                                           ▼                          │   │
│                                   ┌─────────────────┐                 │   │
│                                   │   Scoring       │                 │   │
│                                   │   (InfoNCE)     │                 │   │
│                                   └────────┬────────┘                 │   │
│                                           │                          │   │
│   ◀───────────────────────────────────────┘                          │   │
│   │                                                                   │   │
│   ▼                                                                   │   │
│   ┌──────────────────────────────────────────────────────────────┐     │   │
│   │                    Ranked Candidates                         │     │   │
│   │   1. web_search (0.92)                                      │     │   │
│   │   2. get_weather (0.05)                                    │     │   │
│   │   3. send_email (0.03)                                      │     │   │
│   └──────────────────────────────────────────────────────────────┘     │   │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

*Document created: 28/09/2026*  
*Last updated: 28/09/2026*
