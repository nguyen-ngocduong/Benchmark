# Tool-Calling Evaluation Framework

Đánh giá khả năng tool-calling của **Qwen3.5-0.8B** và **Gemma4-2B** trên Ollama.

## Cấu trúc

```
fixed-uuid/
├── chat_client.py              # Chat completion client (từ user)
├── eval/tool_call/
│   ├── tools.py               # 5 tool definitions
│   ├── mock_tools.py          # Mock responses + RobotState
│   ├── datasets/
│   │   └── summarization/
│   │       └── samples.json    # 10 samples từ vietnamese_news_human_ai
│   ├── scripts/
│   │   ├── load_news_dataset.py      # Load dataset từ HuggingFace
│   │   ├── judge_summarization.py   # Gemini 2.5 flash lite judge
│   │   ├── run_summarization_eval.py # Main evaluation runner
│   │   └── compare_summarization.py  # Compare 2 models
│   └── results/               # Kết quả evaluation
```

## Tools được định nghĩa

1. **get_gold_price** - Giá vàng Việt Nam (SJC, PNJ, DOJI...)
2. **get_weather** - Thời tiết các thành phố Việt Nam
3. **search_web** - Tìm kiếm web (tin tức, thể thao, bách khoa)
4. **query_knowledge_base** - Tra cứu kiến thức nội bộ VinDynamics
5. **execute_robot_command** - Điều khiển robot (đèn, di chuyển, gọi điện)

## Setup

```bash
# Cài đặt dependencies
pip install datasets huggingface_hub openai toon_format

# Kiểm tra Ollama đang chạy
curl http://localhost:11434/api/tags

# List models đã có
ollama list

# Pull models nếu cần
ollama pull qwen3.5:0.8b
ollama pull gemma4:2b

# Set GEMINI_API_KEY (cho summarization judge)
export GEMINI_API_KEY="your-gemini-api-key"
```

## Tasks cho NGUYEN NGOC DUONG (Summarization)

### 1. Load Dataset

```bash
python eval/tool_call/scripts/load_news_dataset.py --n 10
```

### 2. Run Evaluation

```bash
# Qwen3.5-0.8B
python eval/tool_call/scripts/run_summarization_eval.py --model qwen3.5:0.8b

# Gemma4-2B
python eval/tool_call/scripts/run_summarization_eval.py --model gemma4:2b
```

### 3. Compare Models

```bash
python eval/tool_call/scripts/compare_summarization.py \
    --model1 eval/tool_call/results/qwen3.5:0.8b/ \
    --model2 eval/tool_call/results/gemma4:2b/
```

## Scoring Criteria (Gemini 2.5 Flash Lite)

| Metric | Description | Score |
|--------|-------------|-------|
| Correctness | Độ chính xác thông tin | 1-5 |
| Completeness | Độ đầy đủ thông tin | 1-5 |
| Fluency | Độ mượt, tự nhiên | 1-5 |
| Conciseness | Độ ngắn gọn | 1-5 |
| **Overall** | Điểm tổng hợp | 1-5 |

## Output

### JSON Results
```json
{
  "timestamp": "2026-09-15T10:30:00",
  "model": "qwen3.5:0.8b",
  "total_samples": 10,
  "successful_judged": 10,
  "average_scores": {
    "correctness": 4.2,
    "completeness": 3.8,
    "fluency": 3.5,
    "conciseness": 4.0,
    "overall": 3.9
  },
  "by_source": {
    "human": {"count": 5, "avg_overall": 4.1},
    "ai": {"count": 5, "avg_overall": 3.7}
  },
  "latency": {"avg_ms": 450, "min_ms": 200, "max_ms": 1200}
}
```

### Terminal Summary
```
============================================================
  SUMMARIZATION QUALITY - qwen3.5:0.8b
============================================================
  Samples: 10/10 judged
------------------------------------------------------------
  Correctness:    4.2/5.0
  Completeness:   3.8/5.0
  Fluency:        3.5/5.0
  Conciseness:    4.0/5.0
------------------------------------------------------------
  ⭐ OVERALL:      3.9/5.0
------------------------------------------------------------
  Avg Latency:    450ms
============================================================
```

## Dataset

- **Nguồn**: [vietnamese_news_human_ai](https://huggingface.co/datasets/ICCIES-2025-DetectAI/vietnamese_news_human_ai)
- **Size**: 200,000 Vietnamese news articles
- **Balance**: 5 human-written + 5 AI-generated samples

---

## Tasks còn lại cho HA DUYEN HUNG

| Task | Status |
|------|--------|
| Tool selection test cases | Pending |
| Argument accuracy test cases | Pending |
| Multi-turn test cases | Pending |
| Edge cases (state-aware) | Pending |
| Scoring scripts | Pending |
