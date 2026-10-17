"""LLM Client that uses `claude -p` subprocess with stream-json output."""

import asyncio
import json
import os
import re
import time
from typing import Callable, Dict, List, Optional

from scr.api.llm_api.providers.completion_result import CompletionResult
from scr.utils.logger import get_logger

logger = get_logger(__name__)

# Subprocess timeout should be slightly less than the outer call_timeout (default 120s)
# so the subprocess is killed cleanly before the outer timeout fires.
SUBPROCESS_TIMEOUT_SECONDS = 110.0

JSON_ENFORCEMENT = (
    "\n\nCRITICAL: You MUST respond with ONLY a raw JSON object. "
    "No markdown, no code fences, no explanation. Just the raw JSON."
)


def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences (```json ... ```) from Claude's response."""
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
        return  # already exited
    try:
        proc.kill()
        logger.warning(f"Killed orphaned claude subprocess (pid={proc.pid})")
    except ProcessLookupError:
        pass  # already dead
    try:
        await asyncio.wait_for(proc.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        logger.error(f"claude subprocess (pid={proc.pid}) did not exit after kill")


class ClaudeCLIClient:
    """LLM client that uses `claude -p` subprocess with stream-json output."""

    def __init__(self):
        self.provider = "claude"
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
        system_prompt += JSON_ENFORCEMENT

        if not user_prompt.strip():
            raise RuntimeError("Empty prompt — claude -p would hang on empty stdin")

        env = os.environ.copy()
        env.pop("CLAUDECODE", None)

        cmd = [
            "claude", "-p",
            "--system-prompt", system_prompt,
            "--tools", "",
            "--model", model,
            "--effort", "low",
            "--output-format", "stream-json",
            "--verbose",
            "--include-partial-messages",
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=1024 * 1024,  # 1 MB readline buffer (default 64 KB too small for large JSON events)
            env=env,
            cwd="/tmp",
        )

        # Write prompt to stdin and close it
        proc.stdin.write(user_prompt.encode())
        await proc.stdin.drain()
        proc.stdin.close()

        # Read stdout line-by-line for streaming events
        accumulated_text = []
        accumulated_thinking = []

        try:
            async def _read_stream():
                while True:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(),
                        timeout=SUBPROCESS_TIMEOUT_SECONDS,
                    )
                    if not line:
                        break  # EOF

                    line_str = line.decode().strip()
                    if not line_str:
                        continue

                    try:
                        event = json.loads(line_str)
                    except json.JSONDecodeError:
                        logger.debug(f"Non-JSON line from claude stream: {line_str[:100]}")
                        continue

                    top_type = event.get("type")

                    # Unwrap stream_event envelope:
                    # {"type": "stream_event", "event": {"type": "content_block_delta", ...}}
                    if top_type == "stream_event":
                        inner = event.get("event", {})
                        inner_type = inner.get("type")

                        if inner_type == "content_block_delta":
                            delta = inner.get("delta", {})
                            delta_type = delta.get("type")

                            if delta_type == "thinking_delta":
                                thinking_text = delta.get("thinking", "")
                                accumulated_thinking.append(thinking_text)
                                if on_token and thinking_text:
                                    on_token("thinking", thinking_text)

                            elif delta_type == "text_delta":
                                text = delta.get("text", "")
                                accumulated_text.append(text)
                                if on_token and text:
                                    on_token("text", text)

                    elif top_type == "result":
                        # Final result message — extract content from it
                        result_obj = event.get("result", "")
                        if isinstance(result_obj, str) and result_obj:
                            accumulated_text.clear()
                            accumulated_text.append(result_obj)

            await _read_stream()

        except (asyncio.TimeoutError, asyncio.CancelledError):
            await _kill_subprocess(proc)
            raise

        # Wait for process to exit
        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            await _kill_subprocess(proc)

        # Read any stderr
        stderr_data = b""
        try:
            stderr_data = await asyncio.wait_for(proc.stderr.read(), timeout=5.0)
        except asyncio.TimeoutError:
            pass

        if proc.returncode != 0:
            raise RuntimeError(
                f"claude -p failed (exit {proc.returncode}): {stderr_data.decode()}"
            )

        raw = "".join(accumulated_text).strip()

        if not raw:
            raise RuntimeError("Empty response from claude -p (stream-json)")

        content = _strip_code_fences(raw)
        reasoning = "".join(accumulated_thinking)

        logger.info(f"Async LLM response received from claude/{model}")
        return CompletionResult(
            content=content,
            reasoning=reasoning,
            model=f"claude/{model}",
            duration_s=time.time() - t0,
        )

    def get_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs,
    ) -> CompletionResult:
        return asyncio.run(self.async_get_completion(messages, model, **kwargs))
