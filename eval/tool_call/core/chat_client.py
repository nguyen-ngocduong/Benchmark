# core/chat_client.py
import time
import json
import warnings
import asyncio
import logging
from typing import List, Dict, Optional, Union
try:
    import toon_format
except ImportError:
    toon_format = None

# Suppress httpcore2 async generator cleanup warnings (upstream bug in httpcore2)
logging.getLogger("asyncio").setLevel(logging.WARNING)




async def chat_completion_async(
    messages: Union[str, List[Dict[str, str]]],
    base_url: str,
    model: str,
    api_key: str = "EMPTY",
    tools: Optional[List[Dict]] = None,
    tool_choice: str = "auto",
    system_prompt: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    use_toon_format: bool = False,
    enable_thinking: bool = False,
    **other_kwargs
) -> Dict:
    """
    Unified chat completion interface for tool-calling evaluation.
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    if system_prompt:
        messages = [{"role": "system", "content": system_prompt}] + messages

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,                         # <--- ENABLE STREAMING
        "stream_options": {"include_usage": True},
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking
            }
        }
    }

    if tools:
        kwargs.update({"tools": tools, "tool_choice": tool_choice})
    kwargs.update(other_kwargs)

    # import json
    # json.dump(kwargs, open("kwargs_async.json", "w"), indent=4)
    start_time = time.perf_counter()
    try:
        stream = await client.chat.completions.create(**kwargs)

        # ── Accumulator state ──────────────────────────────────────────────
        first_token_time = None
        content_parts: List[str] = []
        reasoning_parts: List[str] = []
        usage_data = None

        # tool-call accumulators: index → {id, name, arguments_chunks}
        tc_accum: Dict[int, Dict] = {}

        async for chunk in stream:
            # ── TTFT: first chunk that carries real content ─────────────────
            delta = chunk.choices[0].delta if chunk.choices else None

            has_content = delta and (
                (delta.content and delta.content != "")
                or getattr(delta, "reasoning", None)
                or getattr(delta, "tool_calls", None)
            )
            if first_token_time is None and has_content:
                first_token_time = time.perf_counter()

            if delta:
                if delta.content:
                    content_parts.append(delta.content)
                if getattr(delta, "reasoning", None):
                    reasoning_parts.append(delta.reasoning)

                # accumulate streamed tool-call fragments
                for tc_chunk in (delta.tool_calls or []):
                    idx = tc_chunk.index
                    if idx not in tc_accum:
                        tc_accum[idx] = {
                            "id": tc_chunk.id or "",
                            "name": (tc_chunk.function and tc_chunk.function.name) or "",
                            "arguments": ""
                        }
                    else:
                        if tc_chunk.id:
                            tc_accum[idx]["id"] = tc_chunk.id
                        if tc_chunk.function and tc_chunk.function.name:
                            tc_accum[idx]["name"] += tc_chunk.function.name

                    if tc_chunk.function and tc_chunk.function.arguments:
                        tc_accum[idx]["arguments"] += tc_chunk.function.arguments

            # usage is typically sent in the final chunk
            if getattr(chunk, "usage", None):
                usage_data = chunk.usage

        end_time = time.perf_counter()

    except Exception as e:
        end_time = time.perf_counter()
        exe_time = end_time - start_time
        response = {
            "error": str(e),
            "content": None,
            "reasoning": None,
            "tool_calls": [],
            "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
            "throughput": {
                "exe_time": exe_time,
                "ttft": None,
                "output_token_per_seconds": None,
                "total_token_per_second": None
            }
        }
        print(response)
        return response

    # ── Timing ────────────────────────────────────────────────────────────
    exe_time = end_time - start_time
    ttft = (first_token_time - start_time) if first_token_time else None  # <--- TTFT

    # ── Token counts ──────────────────────────────────────────────────────
    prompt_tokens    = getattr(usage_data, "prompt_tokens",    0) or 0
    completion_tokens = getattr(usage_data, "completion_tokens", 0) or 0
    total_tokens     = getattr(usage_data, "total_tokens",     0) or 0

    output_token_per_seconds = round(completion_tokens / (exe_time - ttft), 2) if exe_time > 0 else 0.0

    full_content  = "".join(content_parts)
    full_reasoning = "".join(reasoning_parts) if reasoning_parts else None

    result = {
        "content": full_content,
        "reasoning": full_reasoning if enable_thinking else None,
        "usage": {
            "prompt_tokens":    getattr(usage_data, "prompt_tokens",    None),
            "completion_tokens": getattr(usage_data, "completion_tokens", None),
            "total_tokens":     getattr(usage_data, "total_tokens",     None),
        },
        "tool_calls": [],
        "throughput": {
            "exe_time":                round(exe_time, 2),
            "ttft":                    round(ttft, 4) if ttft is not None else None,  # <--- TTFT
            "output_token_per_seconds": output_token_per_seconds,
        }
    }

    # ── Tool-call parsing (same logic as before) ──────────────────────────
    if use_toon_format and "<tool_call>" in full_content:
        tool_calls_raw = full_content.split('</tool_call>\n<tool_call>')
        for tc in tool_calls_raw:
            try:
                tc = tc.replace("<tool_call>", "").replace("</tool_call>", "").strip()
                args = toon_format.decode(tc)
            except toon_format.ToonDecodeError:
                args = {"_raw": tc}
            result["tool_calls"].append(args)
    elif tc_accum:
        for idx in sorted(tc_accum):
            tc = tc_accum[idx]
            try:
                args = json.loads(tc["arguments"])
            except json.JSONDecodeError:
                # print("❌ Failed to parse tool call arguments:", tc["arguments"])
                args = {"_raw": tc["arguments"]}
            result["tool_calls"].append({
                "name": tc["name"],
                "arguments": args
            })

    return result

    
 
