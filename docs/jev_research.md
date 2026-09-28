# JEV Model - Research Documentation

**Documentation Date:** 28/09/2026  
**Status:** Research in Progress - Limited Public Information

---

## ⚠️ Important Note

Thông tin về **JEV model** rất hạn chế trong các nguồn public. JEV được đề cập trong paper/leaderboard của CLM như là baseline để so sánh. Tài liệu này tổng hợp những gì có thể tìm được.

**Cần research thêm:**
- [ ] Tìm HuggingFace page của JEV
- [ ] Tìm paper gốc về JEV
- [ ] Xác định organization/author
- [ ] Kiến trúc chi tiết

---

## 📋 Table of Contents

1. [What We Know About JEV](#1-what-we-know-about-jev)
2. [Known Specifications](#2-known-specifications)
3. [Benchmark Comparisons](#3-benchmark-comparisons)
4. [Architecture Overview](#4-architecture-overview)
5. [Known Use Cases](#5-known-use-cases)
6. [Comparison with CLM](#6-comparison-with-clm)
7. [Research Todo List](#7-research-todo-list)
8. [Speculation vs Confirmed Facts](#8-speculation-vs-confirmed-facts)

---

## 1. What We Know About JEV

### 1.1 Primary Sources

JEV được đề cập trong:

| Source | Context |
|--------|---------|
| **CLM Model Card** | Baseline comparison |
| **CLM GitHub README** | Performance benchmark |
| **Terminal-Bench 2.1** | Failed benchmark (below pass@1) |
| **DeepSWE** | Failed benchmark (below pass@1) |

### 1.2 Confirmed Statements About JEV

Từ CLM documentation:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CLM Statements About JEV                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. "CLM-8B performs on par with Jev while running 9× faster"     │
│                                                                     │
│  2. "Jev fails (below pass@1) on Terminal-Bench 2.1"              │
│                                                                     │
│  3. "Jev fails (below pass@1) on DeepSWE"                        │
│                                                                     │
│  4. "Jev is the baseline for zero-shot comparisons"               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Known Specifications

### 2.1 Confirmed Specifications

| Property | Status | Source |
|----------|--------|--------|
| **Model Name** | JEV | CLM model card |
| **Type** | LLM / Agent model | Inferred |
| **Use Case** | Function calling, Computer use, Gaming | CLM benchmarks |
| **Baseline for** | CLM zero-shot comparison | CLM model card |

### 2.2 Unknown Specifications

| Property | Status |
|----------|--------|
| **Parameters** | Unknown |
| **Base model** | Unknown |
| **Architecture** | Unknown |
| **Training data** | Unknown |
| **Organization** | Unknown |
| **License** | Unknown |
| **HuggingFace link** | Unknown |
| **Paper** | Unknown |

---

## 3. Benchmark Comparisons

### 3.1 Zero-Shot Tasks

| Benchmark | JEV (baseline) | CLM-8B | Winner |
|-----------|----------------|--------|--------|
| T-Rex | Baseline | On par | Tie |
| BFCL v4 (Function Calling) | Baseline | On par | Tie |
| WikiRacing | Baseline | On par | Tie |
| Super Mario | Baseline | On par | Tie |

**Key Finding:** On zero-shot tasks, CLM performs **on par** with JEV.

### 3.2 Agentic Benchmarks

| Benchmark | JEV | CLM-8B (fine-tuned) | Notes |
|-----------|-----|---------------------|-------|
| **Terminal-Bench 2.1** | Failed (< pass@1) | **87.6% (SOTA)** | CLM wins |
| **DeepSWE** | Failed (< pass@1) | **81.6% (SOTA)** | CLM wins |

**Key Finding:** JEV **fails** on long-horizon verification tasks, while CLM achieves SOTA.

### 3.3 Speed Comparison

| Metric | JEV | CLM-8B | Improvement |
|--------|-----|--------|-------------|
| Zero-shot latency | Baseline | 9× faster | CLM wins |
| Verifier latency | Baseline | 4-6× faster | CLM wins |
| Ranking ~1k candidates | Baseline | 13× faster | CLM wins |

---

## 4. Architecture Overview

### 4.1 Known Architecture Details

**JEV appears to be a:**
- [ ] Standard autoregressive LLM (speculation)
- [ ] Capable of function calling (confirmed by benchmarks)
- [ ] Designed for agentic tasks (confirmed by benchmarks)

### 4.2 Speculated Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  JEV - Speculated Architecture                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Based on benchmark context, JEV likely uses:                      │
│                                                                     │
│   ┌───────────────────────────────────────────────────────────┐    │
│   │                     Standard LLM Architecture               │    │
│   │                                                            │    │
│   │   Input → Transformer Layers → Output                      │    │
│   │                    │                                      │    │
│   │                    ▼                                      │    │
│   │            Token-by-token generation                       │    │
│   │                                                            │    │
│   │   • Full model active per inference                       │    │
│   │   • No caching of intermediate states                     │    │
│   │   • No contrastive scoring mechanism                      │    │
│   │                                                            │    │
│   └───────────────────────────────────────────────────────────┘    │
│                                                                     │
│   This explains:                                                    │
│   ✓ Higher latency (vs CLM's 9× faster)                            │
│   ✓ Failure on long-horizon tasks (vs CLM's SOTA)                 │
│   ✓ Similar zero-shot accuracy (both are capable models)           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Known Use Cases

### 5.1 From Benchmark Names

JEV has been evaluated on:

| Benchmark | Domain | JEV Result |
|-----------|--------|------------|
| **BFCL v4** | Function Calling | Baseline |
| **T-Rex** | Generalization | Baseline |
| **WikiRacing** | Multi-hop Reasoning | Baseline |
| **Super Mario** | Game Playing | Baseline |
| **Terminal-Bench 2.1** | Terminal Commands | Failed |
| **DeepSWE** | Software Engineering | Failed |

### 5.2 Implied Capabilities

From successful zero-shot benchmarks:
- ✅ Function calling (BFCL v4)
- ✅ General reasoning (T-Rex)
- ✅ Multi-hop queries (WikiRacing)
- ✅ Game playing (Super Mario)

From failed benchmarks:
- ❌ Long-horizon verification (Terminal-Bench)
- ❌ Deep software understanding (DeepSWE)

---

## 6. Comparison with CLM

### 6.1 Side-by-Side Comparison

| Aspect | JEV | CLM | Notes |
|--------|-----|-----|-------|
| **Architecture** | Standard LLM (speculated) | Qwen3-8B + projection heads | CLM uses contrastive |
| **Approach** | Autoregressive generation | Scoring/ranking | Different paradigms |
| **Parameters** | Unknown | 8B (20M trainable) | Comparable |
| **Zero-shot accuracy** | Baseline | On par | Equal |
| **Zero-shot latency** | Baseline | 9× faster | CLM wins |
| **Long-horizon tasks** | Fails | SOTA | CLM wins |
| **Fine-tuning** | Unknown | Cheap (20M params) | CLM advantage |
| **Caching** | No | Yes (state/action) | CLM advantage |

### 6.2 Key Differences

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      JEV vs CLM - Key Differences                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   JEV                              CLM                                   │
│   ───                              ───                                   │
│                                                                         │
│   ┌─────────────────┐            ┌─────────────────┐                   │
│   │ Token Generation │            │ Candidate Scoring│                   │
│   │                 │            │                 │                    │
│   │  Input          │            │  Input          │                   │
│   │    │            │            │    │            │                   │
│   │    ▼            │            │    ▼            │                   │
│   │ ┌────────────┐ │            │ ┌────────────┐ │                   │
│   │ │ Transformer│ │            │ │ Qwen3-8B   │ │                   │
│   │ │  Layers   │ │            │ │ (frozen)   │ │                   │
│   │ └────────────┘ │            │ └─────┬──────┘ │                   │
│   │    │            │            │       │        │                   │
│   │    ▼            │            │       ▼        │                   │
│   │ Token-by-token  │            │ ┌────────────┐ │                   │
│   │ generation      │            │ │ Projection │ │                   │
│   │    │            │            │ │  Heads    │ │                   │
│   │    ▼            │            │ │(20M params│ │                   │
│   │  Slow           │            │ └─────┬──────┘ │                   │
│   │  Expensive      │            │       │        │                   │
│   └─────────────────┘            │       ▼        │                   │
│                                  │    Fast        │                   │
│                                  │    Scores      │                   │
│                                  └─────────────────┘                   │
│                                                                         │
│   Problem:                       Solution:                              │
│   - Slow inference               - Cached embeddings                   │
│   - Expensive per call           - Ranking vs generation                │
│   - Fails long-horizon           - Contrastive learning                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.3 When to Use Which

| Scenario | Recommendation | Reason |
|----------|---------------|--------|
| **Fast inference needed** | CLM | 9× faster |
| **Long-horizon tasks** | CLM | SOTA on verification |
| **Quick general tasks** | Either | Similar accuracy |
| **Custom tool set** | CLM | Cheap fine-tuning |
| **Already have JEV deployed** | Evaluate | May not need to switch |

---

## 7. Research Todo List

### 7.1 High Priority

- [ ] **Find JEV HuggingFace page**
  - Search: "JEV model huggingface"
  - Search: "Contrastive-LM JEV"
  
- [ ] **Find JEV paper**
  - Search: "JEV LLM paper"
  - Search: "JEV agent model arxiv"

- [ ] **Identify organization**
  - CLM is from Contrastive-LM
  - JEV may be from same or different org

### 7.2 Medium Priority

- [ ] Find JEV benchmark methodology
- [ ] Find JEV training details
- [ ] Find JEV licensing info
- [ ] Find JEV GitHub repository

### 7.3 Search Queries to Try

```bash
# Suggested search queries
"JEV model huggingface"
"JEV LLM agent"
"JEV function calling"
"JEV contrastive learning"
"site:huggingface.co JEV"
"JEV model arxiv"
"JEV decision making model"
"JEV vs CLM"
```

---

## 8. Speculation vs Confirmed Facts

### 8.1 Confirmed Facts

| Fact | Source |
|------|--------|
| JEV exists as a model | CLM model card |
| JEV is used as baseline | CLM model card |
| JEV tested on BFCL v4 | CLM benchmarks |
| JEV tested on Terminal-Bench | CLM benchmarks |
| JEV tested on DeepSWE | CLM benchmarks |
| JEV has higher latency than CLM | CLM model card |
| JEV fails on long-horizon tasks | CLM model card |

### 8.2 Speculations

| Speculation | Reasoning |
|-------------|-----------|
| JEV is a standard autoregressive LLM | Most models are; JEV has no stated innovation |
| JEV uses transformer architecture | Industry standard |
| JEV likely has 7-13B parameters | Common size for competitive models |
| JEV may not have contrastive learning | Would be mentioned if it did |

### 8.3 What We DON'T Know

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Unknown JEV Information                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ❌ Exact architecture                                            │
│   ❌ Number of parameters                                          │
│   ❌ Base model (if any)                                           │
│   ❌ Training data                                                 │
│   ❌ Training method                                               │
│   ❌ Organization / Author                                        │
│   ❌ License                                                       │
│   ❌ HuggingFace page                                              │
│   ❌ GitHub repository                                             │
│   ❌ Paper reference                                               │
│   ❌ How to use / API                                              │
│   ❌ Fine-tuning process                                           │
│   ❌ Quantization options                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Appendix A: CLM's Original Statements About JEV

### A.1 From Model Card

```
"CLM-8B performs on par with Jev while running up to 9× faster"
```

### A.2 From GitHub README

```
┌─────────────────────────────────────────────────────────────────────┐
│  Benchmark Results (from CLM README)                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Terminal-Bench 2.1:                                               │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ CLM (fine-tuned verifier): 87.6% (SOTA)                      │ │
│  │ Jev: Failed (below pass@1)                                   │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  DeepSWE:                                                          │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ CLM (fine-tuned verifier): 81.6% (SOTA)                       │ │
│  │ Jev: Failed (below pass@1)                                   │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  Zero-shot (computer-use, gaming, tool-calling):                  │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │ CLM-8B: On par with Jev, 9× lower latency                   │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### A.3 From Paper/Notion Blog

```
"CLM runs 4.1–5.7× faster than Jev on H100"
```

---

## Appendix B: Timeline of Events

Based on available information:

| Date | Event | Source |
|------|-------|--------|
| Unknown | JEV model developed | Unknown |
| Unknown | JEV benchmarks released | Terminal-Bench, DeepSWE |
| ~2026 | CLM developed by Contrastive-LM | CLM model card |
| ~2026 | CLM compared to JEV | CLM documentation |

---

## Appendix C: Related Models

### C.1 CLM Variants

| Model | Description |
|-------|-------------|
| `CLM-v0.1-8B` | Base zero-shot model |
| `deepswe-clm-heads-8k` | DeepSWE fine-tuned |
| `terminal-clm-heads-8k` | Terminal fine-tuned |

### C.2 Possible JEV Variants

Unknown - requires research

---

## Appendix D: Questions to Answer

### D.1 Technical Questions

1. What is JEV's architecture?
2. How many parameters does JEV have?
3. What base model does JEV use (if any)?
4. How does JEV handle function calling?
5. What is JEV's training approach?

### D.2 Practical Questions

1. How to access/inference JEV?
2. What is JEV's license?
3. Is JEV open-source?
4. How does JEV compare to other models like GPT-4, Claude?

---

*Document created: 28/09/2026*  
*Status: Research In Progress*  
*Last updated: 28/09/2026*
