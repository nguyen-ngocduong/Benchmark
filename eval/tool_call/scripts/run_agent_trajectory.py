"""
Runner for Multi-Turn Agent Trajectory Generation.

Executes AI models (e.g. qwen3.5:0.8b, gemma4:e2b) on multi-turn benchmark scenarios,
recording full interaction trajectories into agent_output.jsonl.

Separation of Concerns:
- This script ONLY runs model inference & handles tool calling.
- Does NOT perform any Judge / evaluation scoring (Judge is kept completely separate).

Output format conforms to:
{
  "sample_id": "sample_001",
  "turns": [
    {
      "turn": 1,
      "user_message": "...",
      "agent_response": {
        "type": "tool_call",
        "tool_call": {
          "name": "web_search",
          "arguments": {"query": "...", "category": "news"}
        }
      },
      "tool_result": {
        "type": "tool_result",
        "tool_name": "web_search",
        "results": [{"snippet": "<ICCIES Text nguyên bản>"}]
      },
      "final_answer": "..."
    },
    {
      "turn": 2,
      "user_message": "...",
      "agent_response": {
        "type": "direct_answer"
      },
      "tool_result": null,
      "final_answer": "..."
    }
  ]
}

Usage:
    python -m eval.tool_call.scripts.run_agent_trajectory --model qwen3.5:0.8b --limit 1
    python -m eval.tool_call.scripts.run_agent_trajectory --model qwen3.5:0.8b
    python -m eval.tool_call.scripts.run_agent_trajectory --model gemma4:e2b --limit 1
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import httpx
from dotenv import load_dotenv

load_dotenv()

# Benchmark tool definition: sole tool web_search according to Section II
BENCHMARK_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Tìm kiếm thông tin trên web để trả lời câu hỏi của người dùng.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Nội dung cần tìm kiếm trên web."
                    },
                    "category": {
                        "type": "string",
                        "enum": ["news", "general"],
                        "description": "Loại thông tin cần tìm kiếm: 'news' cho tin tức bài báo sự kiện, 'general' cho thông tin chung."
                    }
                },
                "required": ["query", "category"]
            }
        }
    }
]

SYSTEM_PROMPT = (
    "Bạn là một trợ lý ảo AI thông minh và hữu ích có khả năng sử dụng công cụ để tra cứu thông tin và trả lời câu hỏi của người dùng.\n\n"
    "QUY TẮC HOẠT ĐỘNG:\n"
    "1. Khi người dùng hỏi về thông tin, sự kiện hoặc nội dung từ bài báo/thực tế mà chưa có trong lịch sử trò chuyện, "
    "bạn BẮT BUỘC PHẢI gọi công cụ `web_search` với category phù hợp ('news' hoặc 'general') và query ngắn gọn, rõ nghĩa để tìm kiếm.\n"
    "2. Khi thông tin đã có sẵn trong lịch sử hội thoại trước đó (ở các lượt trao đổi hoặc kết quả tìm kiếm của lượt trước), "
    "bạn KHÔNG ĐƯỢC gọi lại công cụ tìm kiếm, mà hãy trực tiếp trả lời câu hỏi dựa trên ngữ cảnh đã có.\n"
    "3. Câu trả lời cần chính xác, trung thực, bám sát dữ liệu nhận được từ kết quả công cụ và lịch sử trò chuyện, "
    "không tự suy diễn hay bịa đặt thông tin."
)


class AgentTrajectoryRunner:
    """Runs agent model on multi-turn scenarios and streams trajectories to JSONL."""

    def __init__(
        self,
        model_name: str = "qwen3.5:0.8b",
        base_url: Optional[str] = None,
        num_ctx: int = 2048,
        max_tokens: int = 800,
        temperature: float = 0.0,
        system_prompt: str = SYSTEM_PROMPT,
        tools: Optional[List[Dict]] = None
    ):
        self.model_name = model_name
        self.base_url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        if "gemma" in model_name.lower() and num_ctx > 1536:
            self.num_ctx = 1536
        else:
            self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt
        self.tools = tools if tools is not None else BENCHMARK_TOOLS

    async def cleanup_other_models(self):
        """Unload other models from Ollama RAM before running to prevent OOM."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.get(f"{self.base_url}/api/ps")
                if res.status_code == 200:
                    for m in res.json().get("models", []):
                        name = m.get("name")
                        if name and name != self.model_name:
                            await client.post(
                                f"{self.base_url}/api/generate",
                                json={"model": name, "keep_alive": 0}
                            )
        except Exception:
            pass

    async def call_model(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict]] = None,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Call local model via Ollama native /api/chat.
        Passes arguments as dict to prevent Go JSON parser 400 errors.
        """
        ollama_url = f"{self.base_url}/api/chat"
        num_predict = max_tokens or self.max_tokens

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": num_predict,
                "num_ctx": self.num_ctx
            }
        }
        if tools:
            payload["tools"] = tools

        start_time = time.perf_counter()
        async with httpx.AsyncClient(timeout=180.0) as client:
            for attempt in range(1, 4):
                try:
                    res = await client.post(ollama_url, json=payload)
                    latency_ms = (time.perf_counter() - start_time) * 1000

                    if res.status_code == 200:
                        data = res.json()
                        msg = data.get("message", {})
                        raw_tc = msg.get("tool_calls", [])
                        norm_tc = []
                        for tc in raw_tc:
                            fn = tc.get("function", {})
                            name = fn.get("name") or tc.get("name")
                            args = fn.get("arguments") or tc.get("arguments") or {}
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except Exception:
                                    pass
                            norm_tc.append({
                                "id": tc.get("id") or f"call_{int(time.time()*1000)}",
                                "name": name,
                                "arguments": args
                            })

                        content = msg.get("content") or ""
                        # If content is empty but model put text in thinking
                        if not content.strip() and msg.get("thinking"):
                            content = msg.get("thinking").strip()

                        return {
                            "content": content,
                            "tool_calls": norm_tc,
                            "latency_ms": latency_ms,
                            "error": None
                        }
                    else:
                        print(f"  ⚠️ [Ollama Error {res.status_code}] {res.text[:160]}")
                        if attempt < 3:
                            await asyncio.sleep(5)
                except Exception as e:
                    print(f"  ⚠️ [Ollama Request Exception attempt {attempt}] {str(e)[:160]}")
                    if attempt < 3:
                        await asyncio.sleep(5)

        return {
            "content": "",
            "tool_calls": [],
            "latency_ms": (time.perf_counter() - start_time) * 1000,
            "error": "Failed after retries"
        }

    async def run_scenario(self, scenario_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent trajectory across all turns of a single scenario.
        Returns the formatted scenario trajectory dictionary.
        """
        sample_id = scenario_data.get("sample_id") or scenario_data.get("id") or "unknown"
        source_data = scenario_data.get("source", {})
        article_text = (
            source_data.get("text", "")
            if isinstance(source_data, dict)
            else scenario_data.get("article_text", "")
        )

        scenario_body = scenario_data.get("scenario", {})
        turns = scenario_body.get("turns") if isinstance(scenario_body, dict) else None
        if not turns:
            turns = scenario_data.get("turns") or []

        print(f"\n{'='*65}")
        print(f"🎬 Executing Scenario: {sample_id} ({len(turns)} turns) | Model: {self.model_name}")
        print(f"{'='*65}")

        # Initialize conversation messages with system prompt
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        trajectory_turns: List[Dict[str, Any]] = []

        for turn_idx, turn_item in enumerate(turns, start=1):
            user_msg = turn_item.get("user_message", "")
            turn_num = turn_item.get("turn", turn_idx)

            print(f"\n[Turn {turn_num}]")
            print(f"  User: {user_msg}")

            # Append user message
            messages.append({"role": "user", "content": user_msg})

            # Call model with available tools
            model_res = await self.call_model(messages=messages, tools=self.tools)
            tool_calls = model_res.get("tool_calls", [])

            if tool_calls:
                tc = tool_calls[0]
                tool_name = tc.get("name", "web_search")
                tool_args = tc.get("arguments", {})
                tc_id = tc.get("id") or f"call_{sample_id}_{turn_num}"

                print(f"  🔧 Agent Tool Call: {tool_name}({tool_args}) [{model_res.get('latency_ms', 0):.0f}ms]")

                # Format agent_response for tool_call
                agent_response = {
                    "type": "tool_call",
                    "tool_call": {
                        "name": tool_name,
                        "arguments": tool_args
                    }
                }

                # Construct tool_result with exact raw article text snippet
                tool_result = {
                    "type": "tool_result",
                    "tool_name": tool_name,
                    "results": [
                        {
                            "snippet": article_text
                        }
                    ]
                }

                # Append assistant tool call to message history (arguments as dict for Ollama)
                messages.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": tc_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tool_args
                        }
                    }]
                })

                # Append tool result to message history
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "name": tool_name,
                    "content": json.dumps(tool_result, ensure_ascii=False)
                })

                # Call model again to generate final verbal response
                summary_res = await self.call_model(messages=messages, tools=None)
                final_answer = summary_res.get("content", "").strip()

                # Append final answer to conversation history
                messages.append({"role": "assistant", "content": final_answer})
                print(f"  💬 Final Answer: {final_answer[:120]}... [{summary_res.get('latency_ms', 0):.0f}ms]")

            else:
                # Direct answer without tool calling
                direct_answer = model_res.get("content", "").strip()
                print(f"  💬 Direct Answer: {direct_answer[:120]}... [{model_res.get('latency_ms', 0):.0f}ms]")

                agent_response = {
                    "type": "direct_answer"
                }
                tool_result = None
                final_answer = direct_answer

                # Append direct answer to conversation history
                messages.append({"role": "assistant", "content": direct_answer})

            # Record turn in trajectory
            turn_record = {
                "turn": turn_num,
                "user_message": user_msg,
                "agent_response": agent_response,
                "tool_result": tool_result,
                "final_answer": final_answer
            }
            trajectory_turns.append(turn_record)

        return {
            "sample_id": sample_id,
            "turns": trajectory_turns
        }


def load_dataset(dataset_path: str) -> List[Dict[str, Any]]:
    """Load scenarios from JSON or JSONL file."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    if path.suffix == ".jsonl":
        scenarios = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    scenarios.append(json.loads(line))
        return scenarios
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return data.get("scenarios") or data.get("samples") or [data]
            return []


def get_existing_sample_ids(output_path: str) -> set:
    """Read already evaluated sample_ids from existing output JSONL file."""
    p = Path(output_path)
    if not p.exists():
        return set()
    completed = set()
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    obj = json.loads(line)
                    sid = obj.get("sample_id")
                    if sid:
                        completed.add(sid)
                except Exception:
                    pass
    return completed


def save_trajectory_record(out_path: Path, new_trajectory: Dict[str, Any]):
    """Appends or updates a trajectory record in JSONL, and synchronously updates agent_output.json array."""
    sid = new_trajectory.get("sample_id")
    existing_records = []
    found = False

    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        record = json.loads(line_str)
                        if record.get("sample_id") == sid:
                            existing_records.append(new_trajectory)
                            found = True
                        else:
                            existing_records.append(record)
                    except Exception:
                        pass

    if not found:
        existing_records.append(new_trajectory)

    # Write back to JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        for r in existing_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Also sync to formatted .json array
    json_path = out_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(existing_records, f, ensure_ascii=False, indent=2)


async def main_async():
    parser = argparse.ArgumentParser(description="Run Multi-Turn Agent Trajectory Evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default="qwen3.5:0.8b",
        help="Ollama model name (e.g. qwen3.5:0.8b, gemma4:e2b)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="eval/tool_call/datasets/multi_turn/samples.json",
        help="Path to multi-turn benchmark scenarios JSON or JSONL"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for agent_output.jsonl. Defaults to eval/tool_call/results/{model}/agent_output.jsonl"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of scenarios to run"
    )
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Run only a specific sample_id (e.g. sample_001 or sample_011)"
    )
    parser.add_argument(
        "--from-sample",
        type=str,
        default=None,
        help="Run starting from a specific sample_id onwards (e.g. sample_011)"
    )
    parser.add_argument(
        "--to-sample",
        type=str,
        default=None,
        help="Run up to a specific sample_id (e.g. sample_013)"
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=2048,
        help="Context window size for model (default: 2048)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=800,
        help="Max predict tokens per generation (default: 800)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output file instead of resuming"
    )

    args = parser.parse_args()

    WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent

    # Determine output file path
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = (WORKSPACE_ROOT / out_path).resolve()
    else:
        out_path = (WORKSPACE_ROOT / f"eval/tool_call/results/{args.model}/agent_output.jsonl").resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Check resume state
    if args.force and out_path.exists():
        print(f"🔄 Overwrite mode enabled: removing {out_path}")
        out_path.unlink()

    completed_ids = get_existing_sample_ids(str(out_path)) if not args.force else set()
    if completed_ids:
        print(f"⏩ Resuming: {len(completed_ids)} scenarios already present in {out_path}")

    # Load dataset
    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = (WORKSPACE_ROOT / dataset_path).resolve()
    scenarios = load_dataset(str(dataset_path))
    print(f"📂 Loaded {len(scenarios)} total scenarios from {dataset_path}")

    # Filter by sample-id or from-sample/to-sample
    if args.sample_id:
        scenarios = [s for s in scenarios if (s.get("sample_id") or s.get("id")) == args.sample_id]
        print(f"🎯 Filtered for single sample: {args.sample_id} ({len(scenarios)} matched)")
    elif args.from_sample or args.to_sample:
        from_idx = 0
        to_idx = len(scenarios)
        if args.from_sample:
            for i, s in enumerate(scenarios):
                if (s.get("sample_id") or s.get("id")) == args.from_sample:
                    from_idx = i
                    break
            else:
                print(f"⚠️ Sample {args.from_sample} not found in dataset. Starting from beginning.")
        if args.to_sample:
            for i, s in enumerate(scenarios):
                if (s.get("sample_id") or s.get("id")) == args.to_sample:
                    to_idx = i + 1
                    break
            else:
                print(f"⚠️ Sample {args.to_sample} not found in dataset. Running until end.")
        scenarios = scenarios[from_idx:to_idx]
        print(f"🎯 Filtered range: {len(scenarios)} scenarios ({args.from_sample or 'start'} -> {args.to_sample or 'end'})")
    elif completed_ids:
        # Automatic resume: only run scenarios not yet in agent_output.jsonl
        scenarios = [s for s in scenarios if (s.get("sample_id") or s.get("id")) not in completed_ids]
        print(f"📌 Resuming: Remaining {len(scenarios)} scenarios to execute")

    # Apply limit
    if args.limit and args.limit > 0:
        scenarios = scenarios[:args.limit]
        print(f"⚡ Limiting execution to first {len(scenarios)} scenarios")

    if not scenarios:
        print("✅ No scenarios left to run. Exiting.")
        return

    runner = AgentTrajectoryRunner(
        model_name=args.model,
        num_ctx=args.num_ctx,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )

    print(f"\n🚀 Starting Agent Trajectory Runner for model: {args.model}")
    print(f"📝 Output target: {out_path.resolve()}\n")

    await runner.cleanup_other_models()

    t_start = time.time()
    success_count = 0

    for idx, scenario in enumerate(scenarios, start=1):
        sample_id = scenario.get("sample_id") or scenario.get("id") or f"idx_{idx}"
        print(f"\n[{idx}/{len(scenarios)}] Processing {sample_id}...", flush=True)
        t0 = time.time()

        try:
            trajectory = await runner.run_scenario(scenario)
            save_trajectory_record(out_path, trajectory)
            elapsed = time.time() - t0
            success_count += 1
            print(f"  ✅ Finished {sample_id} in {elapsed:.1f}s (Synced to {out_path.name} & {out_path.with_suffix('.json').name})", flush=True)
        except Exception as e:
            print(f"  ❌ Error executing {sample_id}: {e}", flush=True)

    # Immediately release model from RAM
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            await client.post(f"{runner.base_url}/api/generate", json={"model": args.model, "keep_alive": 0})
        except Exception:
            pass

    total_time = time.time() - t_start
    print(f"\n{'='*65}")
    print(f"🏁 Trajectory Generation Complete!")
    print(f"   Model: {args.model}")
    print(f"   Total processed: {success_count}/{len(scenarios)}")
    print(f"   Total time: {total_time:.1f}s")
    print(f"   Trajectory JSONL: {out_path.resolve()}")
    print(f"   Trajectory JSON:  {out_path.with_suffix('.json').resolve()}")
    print(f"{'='*65}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
