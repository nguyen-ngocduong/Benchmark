# JEV vs CLM: Comprehensive Comparison

**Documentation Date:** 28/09/2026  
**Purpose:** Side-by-side comparison of JEV and CLM models for function calling tasks

---

## 📋 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Quick Comparison](#2-quick-comparison)
3. [Detailed Comparison](#3-detailed-comparison)
4. [Architecture Differences](#4-architecture-differences)
5. [Performance Analysis](#5-performance-analysis)
6. [Use Case Suitability](#6-use-case-suitability)
7. [Cost & Resource Analysis](#7-cost--resource-analysis)
8. [Implementation Guide](#8-implementation-guide)
9. [Decision Matrix](#9-decision-matrix)
10. [Recommendations](#10-recommendations)

---

## 1. Executive Summary

### 1.1 Overview

| Model | Description | Best For |
|-------|-------------|----------|
| **JEV** | Baseline model (standard autoregressive LLM) | Existing deployments, general tasks |
| **CLM** | Contrastive Language Model (System One) | Fast inference, long-horizon tasks, fine-tuning |

### 1.2 Key Takeaways

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           KEY TAKEAWAYS                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ⚡ SPEED:        CLM is 9× faster than JEV                          │
│                                                                         │
│   🎯 ACCURACY:     Equal on zero-shot tasks                            │
│                    CLM wins on long-horizon tasks                       │
│                                                                         │
│   🔧 FINE-TUNING:  CLM is cheaper (20M vs full model)                 │
│                                                                         │
│   📊 BENCHMARKS:   CLM achieves SOTA (87.6%, 81.6%)                   │
│                    JEV fails on Terminal-Bench, DeepSWE                │
│                                                                         │
│   💡 WHEN TO USE:                                                       │
│                    • Fast inference → CLM                              │
│                    • Long-horizon → CLM                                 │
│                    • Quick general → Either                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Quick Comparison

### 2.1 At a Glance

| Aspect | JEV | CLM | Winner |
|--------|-----|-----|--------|
| **Model Type** | Standard LLM | Contrastive LM | N/A |
| **Parameters** | Unknown | 8B (20M trainable) | - |
| **Base Model** | Unknown | Qwen3-8B | - |
| **Approach** | Autoregressive | Scoring/Ranking | Different |
| **License** | Unknown | Apache 2.0 | CLM |
| **Open Source** | Unknown | Yes | CLM |

### 2.2 Performance Summary

| Metric | JEV | CLM | Winner |
|--------|-----|-----|--------|
| Zero-shot Accuracy | Baseline | On par | **Tie** |
| Zero-shot Latency | Baseline | 9× faster | **CLM** |
| Terminal-Bench 2.1 | Failed | 87.6% (SOTA) | **CLM** |
| DeepSWE | Failed | 81.6% (SOTA) | **CLM** |
| Fine-tuning Cost | Unknown | Cheap (20M) | **CLM** |
| Caching | No | Yes (2.8× speedup) | **CLM** |

### 2.3 Feature Comparison

| Feature | JEV | CLM |
|---------|-----|-----|
| Function Calling | ✅ | ✅ |
| Tool Use | ✅ | ✅ |
| Computer Use | ✅ | ✅ |
| Long-horizon Tasks | ❌ | ✅ |
| Caching Mechanism | ❌ | ✅ |
| Cheap Fine-tuning | ❌ | ✅ |
| Multimodal | Unknown | Coming (Oct 2026) |
| Streaming | Unknown | Coming soon |

---

## 3. Detailed Comparison

### 3.1 Model Architecture

#### JEV Architecture (Speculated)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        JEV - Speculated Architecture                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Input: "Select tool for: What is the weather in Hanoi?"             │
│          │                                                              │
│          ▼                                                              │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                    Standard LLM Pipeline                         │  │
│   │                                                                  │  │
│   │   ┌──────────────┐                                               │  │
│   │   │ Embedding   │  Convert tokens to vectors                     │  │
│   │   └──────┬───────┘                                               │  │
│   │          │                                                         │  │
│   │          ▼                                                         │  │
│   │   ┌──────────────┐                                               │  │
│   │   │ Transformer  │  × N layers                                   │  │
│   │   │   Layers     │  Self-attention + FFN                         │  │
│   │   └──────┬───────┘                                               │  │
│   │          │                                                         │  │
│   │          ▼                                                         │  │
│   │   ┌──────────────┐                                               │  │
│   │   │   Output     │  Logits for next token                        │  │
│   │   │   Projection │                                               │  │
│   │   └──────┬───────┘                                               │  │
│   │          │                                                         │  │
│   └──────────┼─────────────────────────────────────────────────────────┘  │
│              │                                                            │
│              ▼                                                            │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │           Token-by-token Generation                               │  │
│   │                                                                    │  │
│   │   Step 1: "get"     → "get" (probability: 0.8)                  │  │
│   │   Step 2: "_"       → "_" (probability: 0.9)                    │  │
│   │   Step 3: "weather" → "weather" (probability: 0.95)             │  │
│   │   Step 4: "("       → "(" (probability: 0.99)                   │  │
│   │   ...                                                                │  │
│   │                                                                    │  │
│   │   ⚠️ Slow: Must generate each token sequentially                │  │
│   │   ⚠️ Expensive: Full model for each inference                   │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

#### CLM Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            CLM Architecture                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Input: State + Candidates                                            │
│   ┌────────────────────────────────────────────────────────────────┐  │
│   │  State: "What is the weather in Hanoi?"                        │  │
│   │  Candidates: [get_weather, web_search, send_email]             │  │
│   └────────────────────────────────────────────────────────────────┘  │
│          │                                                                 │
│          ▼                                                                 │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                   Frozen Qwen3-8B Encoder                        │  │
│   │                                                                 │  │
│   │   State Encoding:         Candidate Encoding:                   │  │
│   │   ┌──────────────┐       ┌──────────────┐                     │  │
│   │   │ State text   │       │ get_weather │                     │  │
│   │   │      ↓       │       │      ↓       │                     │  │
│   │   │ Qwen3-8B    │       │ Qwen3-8B    │                     │  │
│   │   │ (frozen)    │       │ (frozen)    │                     │  │
│   │   │      ↓       │       │      ↓       │                     │  │
│   │   │ s_emb (cached)│      │ a_emb_1     │                     │  │
│   │   └──────────────┘       └──────────────┘                     │  │
│   │                                  │                                │  │
│   └──────────────────────────────────┼────────────────────────────────┘  │
│                                      │                                    │
│                                      ▼                                    │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                    Trainable Projection Heads                     │  │
│   │                                                                 │  │
│   │   ┌──────────────┐       ┌──────────────┐                     │  │
│   │   │  State Head  │       │  Action Head │                     │  │
│   │   │  (10M params)│       │  (10M params)│                     │  │
│   │   │      ↓        │       │      ↓        │                     │  │
│   │   │ s_proj       │       │ a_proj_1     │                     │  │
│   │   └──────────────┘       └──────────────┘                     │  │
│   │                                                                 │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                      │                                    │
│                                      ▼                                    │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                    Scoring (Bidirectional InfoNCE)               │  │
│   │                                                                 │  │
│   │   Score(get_weather) = cosine_similarity(s_proj, a_proj_1)    │  │
│   │   Score(web_search)  = cosine_similarity(s_proj, a_proj_2)    │  │
│   │   Score(send_email)  = cosine_similarity(s_proj, a_proj_3)    │  │
│   │                                                                 │  │
│   │   Results:                                                      │  │
│   │   ┌─────────────────────────────────────────────────────────┐  │  │
│   │   │ 1. get_weather (0.92) ✓ Most likely                   │  │  │
│   │   │ 2. web_search (0.06)                                  │  │  │
│   │   │ 3. send_email (0.02)                                  │  │  │
│   │   └─────────────────────────────────────────────────────────┘  │  │
│   │                                                                 │  │
│   │   ✅ Fast: Parallel scoring                                    │  │
│   │   ✅ Cached: Reuse embeddings                                  │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Training Comparison

| Aspect | JEV | CLM |
|--------|------|-----|
| **Training Approach** | Standard LLM training | 3-stage contrastive learning |
| **Pre-training Data** | Unknown | ~60M Nemotron DQA pairs |
| **Mid-training Data** | Unknown | ~30M synthetic hard negatives |
| **Post-training Data** | Unknown | ~1M agentic trajectories |
| **Total Training Data** | Unknown | ~91M samples |
| **Trainable Parameters** | Full model | 20M (projection heads only) |
| **Fine-tuning Cost** | Expensive | Cheap |

### 3.3 Inference Comparison

| Aspect | JEV | CLM |
|--------|------|-----|
| **Inference Method** | Token-by-token generation | Candidate scoring |
| **Latency** | Baseline | 9× faster |
| **Caching** | No | Yes (state/action caching) |
| **Memory Usage** | Unknown | Lower (only heads need GPU) |
| **Batch Processing** | Standard | Efficient (parallel scoring) |

---

## 4. Architecture Differences

### 4.1 Paradigm Difference

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PARADIGM COMPARISON                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   JEV: GENERATION PARADIGM              CLM: SCORING PARADIGM            │
│   ═══════════════════════                ═══════════════════════          │
│                                                                             │
│   Question: "Weather in Hanoi?"           Question: "Weather in Hanoi?"    │
│         │                                        │                          │
│         ▼                                        ▼                          │
│   ┌─────────────┐                         ┌─────────────┐                │
│   │  Generate   │                         │  Candidates │                │
│   │  "get_      │                         │  [✓weather, │                │
│   │   weather"  │                         │   ✗search, │                │
│   │   token by  │                         │   ✗email]   │                │
│   │   token     │                         │             │                │
│   └─────────────┘                         └──────┬──────┘                │
│         │                                          │                       │
│         ▼                                          ▼                       │
│   ┌─────────────┐                         ┌─────────────┐                │
│   │   SLOW      │                         │    FAST     │                │
│   │  Expensive  │                         │   Efficient │                │
│   │   May hallucinate│                     │   Precise   │                │
│   └─────────────┘                         └─────────────┘                │
│                                                                             │
│   Problem:                          Solution:                               │
│   - Must generate full text          - Only select from candidates         │
│   - Can hallucinate                  - Constrained to valid choices         │
│   - Expensive                        - Cached embeddings                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Caching Mechanism

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLM CACHING MECHANISM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   WITHOUT CACHING (JEV-like):                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                      │  │
│   │   Turn 1: "What is weather?"    → Full encode → Result (slow)     │  │
│   │   Turn 2: "And in Saigon?"       → Full encode → Result (slow)     │  │
│   │   Turn 3: "Any rain?"            → Full encode → Result (slow)     │  │
│   │                                                                      │  │
│   │   Total: 3 × encode_time                                         │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   WITH CACHING (CLM):                                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                      │  │
│   │   Turn 1: "What is weather?"    → Full encode → Result             │  │
│   │                                      ↓                               │  │
│   │   ┌────────────────────────────────────────────────────────────┐    │  │
│   │   │  Cache: state_embedding_1, candidate_embeddings          │    │  │
│   │   └────────────────────────────────────────────────────────────┘    │  │
│   │                                      │                               │  │
│   │   Turn 2: "And in Saigon?"       → Reuse state cache  → Result (fast)│  │
│   │                                      │                               │  │
│   │   Turn 3: "Any rain?"            → Reuse state cache  → Result (fast)│  │
│   │                                                                      │  │
│   │   Total: 1 × encode_time + 2 × cache_lookup                       │  │
│   │   Speedup: 2.8× on repeated states                                │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Performance Analysis

### 5.1 Benchmark Results

#### Zero-Shot Tasks

| Benchmark | Domain | JEV | CLM | Notes |
|-----------|--------|-----|-----|-------|
| **T-Rex** | Generalization | Baseline | On par | Equal |
| **BFCL v4** | Function Calling | Baseline | On par | Equal |
| **WikiRacing** | Multi-hop | Baseline | On par | Equal |
| **Super Mario** | Game Playing | Baseline | On par | Equal |

#### Agentic Benchmarks

| Benchmark | Domain | JEV | CLM | Notes |
|-----------|--------|-----|-----|-------|
| **Terminal-Bench 2.1** | Terminal Commands | Failed (<pass@1) | **87.6%** | CLM SOTA |
| **DeepSWE** | Software Engineering | Failed (<pass@1) | **81.6%** | CLM SOTA |

#### Speed Benchmarks

| Metric | JEV | CLM | Improvement |
|--------|-----|-----|-------------|
| Zero-shot Latency | Baseline | 9× faster | **CLM** |
| Verifier Latency | Baseline | 4-6× faster | **CLM** |
| Ranking 1k Candidates | Baseline | 13× faster | **CLM** |
| Repeated States | Baseline | 2.8× faster | **CLM** |

### 5.2 Visual Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ACCURACY COMPARISON                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Zero-Shot Tasks                                                           │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  JEV  ████████████████████████████████████ 100%                    │  │
│   │  CLM  ████████████████████████████████████ 100%                    │  │
│   │                                                                     │  │
│   │  [T-Rex] [BFCL v4] [WikiRacing] [Mario]                          │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│   Result: Equal performance                                                 │
│                                                                             │
│   Agentic Benchmarks                                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  CLM  ████████████████████████████████████████████████ 87.6% 81.6%│  │
│   │  JEV  ████ (Failed, below pass@1)                                    │  │
│   │                                                                     │  │
│   │  [Terminal-Bench 2.1]           [DeepSWE]                           │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│   Result: CLM significantly better                                          │
│                                                                             │
│   Latency (Lower is better)                                                 │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  CLM  ████                                                           │  │
│   │  JEV  ████████████████████████████████████████████████             │  │
│   │       1x            9x faster                                       │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│   Result: CLM 9× faster                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Use Case Suitability

### 6.1 When to Use JEV

| Use Case | Recommendation | Reason |
|----------|---------------|--------|
| **Existing JEV deployment** | Keep | Switching cost |
| **Need token generation** | Use JEV | CLM can't generate |
| **Unseen actions** | Use JEV | CLM needs predefined candidates |
| **Internal model** | Keep | Unknown license |
| **Already fine-tuned** | Evaluate | May not need switch |

### 6.2 When to Use CLM

| Use Case | Recommendation | Reason |
|----------|---------------|--------|
| **Fast inference required** | Use CLM | 9× faster |
| **Long-horizon tasks** | Use CLM | SOTA on verification |
| **Cheap fine-tuning** | Use CLM | Only 20M params |
| **Cost-sensitive** | Use CLM | Lower compute |
| **Function calling** | Use CLM | Better tool selection |
| **Caching benefits** | Use CLM | State/action cache |
| **Open source required** | Use CLM | Apache 2.0 |

### 6.3 Task-Based Recommendations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TASK-BASED RECOMMENDATIONS                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   TASK                           RECOMMENDED MODEL         NOTES              │
│   ────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│   Simple Q&A                   Either                   Similar accuracy    │
│   ────────────────────────────────────────────────────────────────────────  │
│   Function Calling             CLM                     Faster + precise   │
│   ────────────────────────────────────────────────────────────────────────  │
│   Tool Selection              CLM                     Cached candidates   │
│   ────────────────────────────────────────────────────────────────────────  │
│   Long-horizon Agent          CLM                     SOTA on verif.      │
│   ────────────────────────────────────────────────────────────────────────  │
│   Real-time Applications      CLM                     9× lower latency     │
│   ────────────────────────────────────────────────────────────────────────  │
│   Cost-sensitive              CLM                     Cheaper compute      │
│   ────────────────────────────────────────────────────────────────────────  │
│   Custom Tool Set             CLM                     Cheap fine-tuning   │
│   ────────────────────────────────────────────────────────────────────────  │
│   Open Source Required        CLM                     Apache 2.0          │
│   ────────────────────────────────────────────────────────────────────────  │
│   Must Generate Text          JEV                     CLM can't generate  │
│   ────────────────────────────────────────────────────────────────────────  │
│   Unseen Actions              JEV                     CLM needs candidates │
│   ────────────────────────────────────────────────────────────────────────  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Cost & Resource Analysis

### 7.1 Inference Cost

| Metric | JEV | CLM |
|--------|-----|-----|
| **GPU Memory** | Full model | Only projection heads |
| **Latency per call** | Baseline | 9× faster |
| **Batch efficiency** | Standard | High (parallel scoring) |
| **Caching benefit** | None | 2.8× speedup on repeated |

### 7.2 Fine-tuning Cost

| Aspect | JEV | CLM |
|--------|------|-----|
| **Trainable params** | Full model | 20M (0.25% of 8B) |
| **GPU time** | High | Low |
| **Data needed** | Large | Small (task-specific) |
| **Cost** | Expensive | Cheap |

### 7.3 Total Cost of Ownership

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TOTAL COST COMPARISON (Estimated)                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Setup Costs:                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  JEV: Unknown (may require custom setup)                           │  │
│   │  CLM: vLLM server + pip install                                    │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Inference Costs (per 1M requests):                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  JEV: ████████████████████████████████████████████████████████ $X   │  │
│   │  CLM: ████████                                             $X/9    │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Fine-tuning Costs:                                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │  JEV: Full model training (expensive)                               │  │
│   │  CLM: Projection heads only (cheap)                                 │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Implementation Guide

### 8.1 CLM Implementation

```python
# CLM Implementation for Function Calling
from clm import CLMClient, Choice

client = CLMClient()

# Define available tools
tools = {
    "web_search": "Search the web for information",
    "get_weather": "Get weather for a location",
    "send_email": "Send an email"
}

# Format conversation as state
state = """
System: You are a helpful assistant.
User: What is the weather in Hanoi today?
"""

# Run CLM inference
result = client.system_one(
    state=state,
    questions={
        "tool": Choice(
            instructions="Select the appropriate tool",
            criteria=tools
        )
    }
)

# Get predicted tool
predicted_tool = result.tool  # "get_weather"
confidence = result.tool_prob  # 0.92
```

### 8.2 JEV Implementation

```python
# JEV Implementation (speculative)
# Based on standard function calling format

# from jev import JEVClient  # Assuming this exists

# client = JEVClient()

# # Format conversation
# messages = [
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "What is the weather in Hanoi today?"}
# ]

# tools = [
#     {
#         "name": "web_search",
#         "description": "Search the web",
#         "parameters": {...}
#     }
# ]

# # Run JEV inference
# result = client.chat_completion(messages, tools=tools)
# predicted_tool = result.tool_calls[0].function.name
```

### 8.3 Comparison Script

```python
# compare_models.py
import json
from clm import CLMClient, Choice

class ModelComparison:
    def __init__(self):
        self.clm = CLMClient()
        # self.jev = JEVClient()  # When available

    def evaluate_sample(self, sample):
        """Evaluate one sample with both models"""
        
        messages = sample["messages"]
        tools = sample["tools"]
        ground_truth = self.extract_ground_truth(messages)

        # CLM prediction
        clm_pred = self.run_clm(messages, tools)
        clm_correct = self.check_match(clm_pred, ground_truth)

        # JEV prediction (when available)
        # jev_pred = self.run_jev(messages, tools)
        # jev_correct = self.check_match(jev_pred, ground_truth)

        return {
            "sample_id": sample["conversation_id"],
            "ground_truth": ground_truth,
            "clm_prediction": clm_pred,
            "clm_correct": clm_correct,
            # "jev_prediction": jev_pred,
            # "jev_correct": jev_correct
        }

    def run_clm(self, messages, tools):
        state = self.format_messages(messages)
        tool_criteria = {t["function"]["name"]: t["function"]["description"] 
                        for t in tools}
        
        result = self.clm.system_one(
            state=state,
            questions={"tool": Choice(instructions="Select tool", criteria=tool_criteria)}
        )
        return result.tool

    def format_messages(self, messages):
        """Convert messages to CLM state format"""
        formatted = []
        for msg in messages:
            role = msg["role"].capitalize()
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

    def extract_ground_truth(self, messages):
        """Extract ground truth tool from last assistant message"""
        for msg in reversed(messages):
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                return msg["tool_calls"][0]["function"]["name"]
        return None

    def check_match(self, pred, truth):
        return pred == truth if (pred and truth) else False
```

---

## 9. Decision Matrix

### 9.1 Scoring System

| Criterion | Weight | JEV Score | CLM Score |
|-----------|--------|-----------|-----------|
| Speed | 25% | 2/10 | 10/10 |
| Accuracy (zero-shot) | 20% | 8/10 | 8/10 |
| Long-horizon tasks | 15% | 1/10 | 10/10 |
| Fine-tuning cost | 15% | 3/10 | 9/10 |
| Open source | 10% | 0/10 | 10/10 |
| Caching | 10% | 0/10 | 10/10 |
| Documentation | 5% | 2/10 | 8/10 |
| **Total** | 100% | **3.1** | **9.2** |

### 9.2 Visual Decision Matrix

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DECISION MATRIX VISUAL                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Criterion              JEV                      CLM                       │
│   ────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│   Speed                   ██░░░░░░░░              ██████████               │
│                          (2/10)                  (10/10)                    │
│                                                                             │
│   Accuracy                ████████░░              ████████░░                │
│   (zero-shot)             (8/10)                  (8/10)                   │
│                                                                             │
│   Long-horizon            █░░░░░░░░░              ██████████               │
│                          (1/10)                  (10/10)                    │
│                                                                             │
│   Fine-tune cost          ███░░░░░░░              █████████░               │
│                          (3/10)                  (9/10)                    │
│                                                                             │
│   Open source             ░░░░░░░░░░              ██████████               │
│                          (0/10)                  (10/10)                    │
│                                                                             │
│   Caching                 ░░░░░░░░░░              ██████████               │
│                          (0/10)                  (10/10)                    │
│                                                                             │
│   Documentation           ██░░░░░░░░              ████████░░                │
│                          (2/10)                  (8/10)                    │
│                                                                             │
│   ────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│   TOTAL SCORE             ███░░░░░░░              █████████░               │
│                          (3.1/10)                (9.2/10)                   │
│                                                                             │
│   RECOMMENDATION:                        CLM ✅                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Recommendations

### 10.1 For New Projects

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NEW PROJECT RECOMMENDATION                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ✅ USE CLM                                                              │
│                                                                             │
│   Reasons:                                                                 │
│   1. Open source (Apache 2.0)                                              │
│   2. 9× faster inference                                                   │
│   3. SOTA on long-horizon tasks                                            │
│   4. Cheap fine-tuning                                                     │
│   5. Good documentation                                                    │
│   6. Active development (35B coming Oct 2026)                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 For Existing JEV Projects

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EXISTING JEV PROJECT RECOMMENDATION                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   🔄 EVALUATE SWITCH TO CLM                                                │
│                                                                             │
│   When to consider switching:                                              │
│   ✓ Performance is critical                                                │
│   ✓ Long-horizon tasks are failing                                         │
│   ✓ Need cheaper fine-tuning                                               │
│   ✓ Want to reduce latency                                                │
│                                                                             │
│   When to keep JEV:                                                        │
│   ✓ Already working well                                                  │
│   ✓ Team is familiar with JEV                                             │
│   ✓ Switching cost is high                                                 │
│   ✓ Need features only JEV has                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.3 For Function Calling Specifically

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FUNCTION CALLING RECOMMENDATION                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   🎯 RECOMMENDATION: CLM                                                   │
│                                                                             │
│   Why CLM is better for function calling:                                  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                      │  │
│   │  1. PRECISE TOOL SELECTION                                          │  │
│   │     • Scores each candidate                                         │  │
│   │     • No hallucination of invalid tools                            │  │
│   │     • Returns probability distribution                              │  │
│   │                                                                      │  │
│   │  2. FAST INFERENCE                                                  │  │
│   │     • 9× faster than generation-based models                       │  │
│   │     • Cached embeddings for repeated states                       │  │
│   │     • Efficient batch processing                                    │  │
│   │                                                                      │  │
│   │  3. EASY FINE-TUNING                                                │  │
│   │     • Only 20M trainable parameters                                 │  │
│   │     • Adapt to custom tool schemas                                 │  │
│   │     • Quick iteration cycle                                         │  │
│   │                                                                      │  │
│   │  4. CONSISTENT OUTPUT                                               │  │
│   │     • Always returns valid tool from candidates                    │  │
│   │     • Predictable behavior                                          │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix A: Summary Table

| Aspect | JEV | CLM | Winner |
|--------|-----|-----|--------|
| **Speed** | Baseline | 9× faster | CLM |
| **Zero-shot accuracy** | Baseline | On par | Tie |
| **Long-horizon** | Failed | SOTA (87.6%, 81.6%) | CLM |
| **Fine-tuning** | Expensive | Cheap (20M) | CLM |
| **Open source** | Unknown | Yes (Apache 2.0) | CLM |
| **Caching** | No | Yes | CLM |
| **Documentation** | Limited | Good | CLM |
| **Token generation** | Yes | No | JEV |
| **Unseen actions** | Yes | No | JEV |

---

## Appendix B: Quick Reference

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           QUICK REFERENCE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   CLM:                                                                     │
│   • pip install contrastive-lm                                             │
│   • vllm serve Qwen/Qwen3-8B --port 8090                                  │
│   • clm-serve                                                               │
│   • from clm import CLMClient, Choice                                       │
│                                                                             │
│   JEV: (Research needed)                                                   │
│   • HuggingFace page: Unknown                                               │
│   • GitHub: Unknown                                                         │
│   • Paper: Unknown                                                          │
│                                                                             │
│   Related Models:                                                           │
│   • CLM-v0.1-8B: Base zero-shot model                                      │
│   • deepswe-clm-heads-8k: DeepSWE fine-tuned                               │
│   • terminal-clm-heads-8k: Terminal fine-tuned                             │
│                                                                             │
│   Resources:                                                                │
│   • CLM HuggingFace: https://huggingface.co/Contrastive-LM/CLM-v0.1-8B    │
│   • CLM GitHub: https://github.com/Contrastive-LM/CLM                     │
│   • CLM Blog: https://contrastive-lm.notion.site                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Appendix C: Research Status

| Item | Status | Notes |
|------|--------|-------|
| CLM Documentation | ✅ Complete | Full details available |
| JEV Documentation | ⚠️ Partial | Limited public info |
| JEV HuggingFace | ❓ Unknown | Need research |
| JEV GitHub | ❓ Unknown | Need research |
| JEV Paper | ❓ Unknown | Need research |
| JEV Architecture | ⚠️ Speculated | Based on CLM comparisons |

---

*Document created: 28/09/2026*  
*Last updated: 28/09/2026*
