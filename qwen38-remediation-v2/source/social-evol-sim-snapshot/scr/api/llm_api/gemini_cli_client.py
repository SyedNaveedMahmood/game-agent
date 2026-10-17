"""LLM Client that uses `gemini -p` subprocess with stream-json output."""

import asyncio
import json
import os
import re
import time
from typing import Callable, Dict, List, Optional

from scr.api.llm_api.providers.completion_result import CompletionResult
from scr.utils.logger import get_logger

logger = get_logger(__name__)

SUBPROCESS_TIMEOUT_SECONDS = 110.0

JSON_ENFORCEMENT = (
    "\n\nCRITICAL: You MUST respond with ONLY a raw JSON object. "
    "No markdown, no code fences, no explanation. Just the raw JSON."
)


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ```) from response."""
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    return match.group(1) if match else text


def _format_messages(messages: List[Dict[str, str]]) -> tuple[str, str]:
    """Extract system prompt and format remaining messages as user prompt.

    Returns:
        (system_prompt, user_prompt) tuple
    """
    system_prompt = ""
    conversation = []

    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        else:
            conversation.append(msg)

    if len(conversation) == 1:
        user_prompt = conversation[0]["content"]
    else:
        parts = []
        for msg in conversation:
            if msg["role"] == "assistant":
                parts.append(f"[Previous response]\n{msg['content']}")
            elif msg["role"] == "user":
                parts.append(f"[Current instruction]\n{msg['content']}")
        user_prompt = "\n\n".join(parts)

    return system_prompt, user_prompt


async def _kill_subprocess(proc: asyncio.subprocess.Process) -> None:
    """Kill a subprocess and reap it to avoid zombies."""
    if proc.returncode is not None:
        return
    try:
        proc.kill()
        logger.warning(f"Killed orphaned gemini subprocess (pid={proc.pid})")
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(proc.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        logger.error(f"gemini subprocess (pid={proc.pid}) did not exit after kill")


class GeminiCLIClient:
    """LLM client that uses `gemini -p` subprocess with stream-json output."""

    def __init__(self):
        self.provider = "gemini"
        self.extra_kwargs = {}

    async def async_get_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs,
    ) -> CompletionResult:
        t0 = time.time()
        on_token: Optional[Callable] = kwargs.pop("on_token", None)

        system_prompt, user_prompt = _format_messages(messages)
        # Combine system prompt and JSON enforcement into the -p prompt,
        # send user content via stdin.
        combined_prompt = system_prompt + JSON_ENFORCEMENT

        if not user_prompt.strip():
            raise RuntimeError("Empty prompt — gemini -p would produce no output")

        cmd = [
            "gemini", "-p", combined_prompt,
            "-o", "stream-json",
        ]
        if model:
            cmd.extend(["-m", model])

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="/tmp",
            limit=1024 * 1024,  # 1 MB readline buffer (default 64 KB too small for large JSON events)
        )

        # Write user prompt to stdin and close
        proc.stdin.write(user_prompt.encode())
        await proc.stdin.drain()
        proc.stdin.close()

        # Read stream-json events
        accumulated_text = []

        try:
            async def _read_stream():
                while True:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(),
                        timeout=SUBPROCESS_TIMEOUT_SECONDS,
                    )
                    if not line:
                        break

                    line_str = line.decode().strip()
                    if not line_str:
                        continue

                    try:
                        event = json.loads(line_str)
                    except json.JSONDecodeError:
                        logger.debug(f"Non-JSON line from gemini stream: {line_str[:100]}")
                        continue

                    event_type = event.get("type")

                    if event_type == "message" and event.get("role") == "assistant":
                        content = event.get("content", "")
                        if content and on_token:
                            on_token("text", content)
                        if event.get("delta"):
                            accumulated_text.append(content)
                        else:
                            # Full message (non-delta) — replace accumulated
                            accumulated_text.clear()
                            accumulated_text.append(content)

                    elif event_type == "result":
                        # Final result — keep accumulated text as-is
                        pass

            await _read_stream()

        except (asyncio.TimeoutError, asyncio.CancelledError):
            await _kill_subprocess(proc)
            raise

        # Wait for exit
        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            await _kill_subprocess(proc)

        # Read stderr
        stderr_data = b""
        try:
            stderr_data = await asyncio.wait_for(proc.stderr.read(), timeout=5.0)
        except asyncio.TimeoutError:
            pass

        if proc.returncode != 0:
            raise RuntimeError(
                f"gemini -p failed (exit {proc.returncode}): {stderr_data.decode()[:500]}"
            )

        raw = "".join(accumulated_text).strip()

        if not raw:
            raise RuntimeError("Empty response from gemini -p (stream-json)")

        content = _strip_code_fences(raw)

        logger.info(f"Async LLM response received from gemini/{model}")
        return CompletionResult(
            content=content,
            reasoning="",
            model=f"gemini/{model}",
            duration_s=time.time() - t0,
        )

    def get_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs,
    ) -> CompletionResult:
        return asyncio.run(self.async_get_completion(messages, model, **kwargs))
