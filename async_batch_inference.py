import json
import asyncio
import aiofiles
from pathlib import Path
from tqdm.asyncio import tqdm_asyncio
import argparse

from client import chat_completion_async


def parse_args():
    parser = argparse.ArgumentParser(description="Async LLM Inference on JSONL prompts")
    
    # LLM Config
    parser.add_argument("--base_url",        type=str, default="http://localhost:8288/v1", help="LLM API endpoint")
    parser.add_argument("--api_key",         type=str, default="EMPTY", help="API Key")
    parser.add_argument("--model",           type=str, default="Qwen/Qwen3.5-35B-A3B", help="Model name")
    parser.add_argument("--max_concurrency", type=int, default=32, help="Max parallel requests")
    parser.add_argument("--temperature",     type=float, default=0.7)
    parser.add_argument("--max_tokens",      type=int, default=1024)
    parser.add_argument("--enable_thinking", action="store_true", help="Enable thinking/reasoning mode")
    
    # Input / Output
    parser.add_argument("--input",           type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output",          type=str, required=True, help="Output JSONL file")
    parser.add_argument("--failed",          type=str, default=None, help="Failed rows output (optional)")
    
    # Misc
    parser.add_argument("--system_prompt",   type=str, default=None, help="Optional system prompt")
    
    return parser.parse_args()


ARGS = parse_args()

# Global config
BASE_URL        = ARGS.base_url
API_KEY         = ARGS.api_key
MODEL           = ARGS.model
MAX_CONCURRENCY = ARGS.max_concurrency
TEMPERATURE     = ARGS.temperature
MAX_TOKENS      = ARGS.max_tokens
ENABLE_THINKING = ARGS.enable_thinking


# ── Helpers ─────────────────────────────────────────────────────────────────────

def load_done_indices(output_path: str) -> set[int]:
    done = set()
    if not Path(output_path).exists():
        return done
    with open(output_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    idx = rec.get("idx")
                    done.add(i if idx is None else idx)  # NB: idx 0 is falsy — use is-None check
                except:
                    done.add(i)
    return done


def to_messages(prompt: str, system_prompt: str = None):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


async def process_row(
    idx: int,
    prompt: str,
    extra_data: dict,
    semaphore: asyncio.Semaphore,
    file_lock: asyncio.Lock,
    system_prompt: str = None,
):
    messages = to_messages(prompt, system_prompt)

    async with semaphore:
        try:
            response = await chat_completion_async(
                base_url=BASE_URL,
                api_key=API_KEY,
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                enable_thinking=ENABLE_THINKING,
            )
            
            # Extract content and reasoning
            if isinstance(response, dict):
                content = response.get("content", "")
                reasoning = response.get("reasoning") if ENABLE_THINKING else None
            else:
                content = str(response)
                reasoning = None

        except Exception as e:
            print(f"[row {idx}] Error: {e}")
            content = None
            reasoning = None

    # Build result
    result = {
        "idx": idx,
        "prompt": prompt,
        "response": content,
        **extra_data
    }
    
    # Add reasoning only when enabled
    if ENABLE_THINKING:
        result["reasoning"] = reasoning

    # Write to file
    async with file_lock:
        if content is None:
            failed_path = ARGS.failed or str(Path(ARGS.output).with_name("failed.jsonl"))
            async with aiofiles.open(failed_path, "a", encoding="utf-8") as f:
                await f.write(json.dumps(result, ensure_ascii=False) + "\n")
        else:
            async with aiofiles.open(ARGS.output, "a", encoding="utf-8") as f:
                await f.write(json.dumps(result, ensure_ascii=False) + "\n")


# ── Main ────────────────────────────────────────────────────────────────────────

async def main():
    input_path = Path(ARGS.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    Path(ARGS.output).parent.mkdir(parents=True, exist_ok=True)
    if ARGS.failed:
        Path(ARGS.failed).parent.mkdir(parents=True, exist_ok=True)
    else:
        Path(str(Path(ARGS.output).with_name("failed.jsonl"))).parent.mkdir(parents=True, exist_ok=True)

    done_indices = load_done_indices(ARGS.output)
    print(f"Already processed: {len(done_indices)} rows")

    # Read JSONL
    rows = []
    with open(input_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                prompt = data.get("prompt") or data.get("user_message")
                if not prompt:
                    continue
                    
                if i not in done_indices:
                    extra = {k: v for k, v in data.items() if k not in ["prompt", "user_message", "system_prompt"]}
                    rows.append((i, prompt, extra, data.get("system_prompt")))
            except Exception as e:
                print(f"Error parsing line {i}: {e}")

    print(f"Rows to process: {len(rows)}")

    if not rows:
        print("Nothing to do.")
        return

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    file_lock = asyncio.Lock()

    tasks = [
        process_row(idx, prompt, extra, semaphore, file_lock, row_system or ARGS.system_prompt)
        for idx, prompt, extra, row_system in rows
    ]

    await tqdm_asyncio.gather(*tasks, desc="Inference")

    print(f"\n✅ Done!")
    print(f"   Output  → {ARGS.output}")
    if ARGS.failed:
        print(f"   Failed  → {ARGS.failed}")


if __name__ == "__main__":
    asyncio.run(main())
