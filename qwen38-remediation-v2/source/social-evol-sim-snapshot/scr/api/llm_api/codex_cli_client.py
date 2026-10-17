"""LLM Client that uses `codex exec` subprocess with file-based output."""

import asyncio
import json
import os
import re
import tempfile
import time
from typing import Callable, Dict, List, Optional

from scr.api.llm_api.providers.completion_result import CompletionResult
from scr.utils.logger import get_logger

logger = get_logger(__name__)

SUBPROCESS_TIMEOUT_SECONDS = 150.0

JSON_ENFORCEMENT = (
    "\n\nCRITICAL: You MUST respond with ONLY a raw JSON object. "
    "No markdown, no code fences, no explanation. Just the raw JSON."
)


def _extract_first_json_object(text: str) -> str:
    """Extract the first complete JSON object from text.

    Handles the common Codex failure mode where extra content appears after
    the closing brace of the JSON object.
    """
    depth = 0
    in_string = False
    escape = False
    for i, c in enumerate(text):
        if escape:
            escape = False
            continue
        if c == '\\':
            if in_string:
                escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                extracted = text[:i + 1]
                if i + 1 < len(text):
                    extra = len(text) - i - 1
                    logger.warning(
                        f"Trimmed {extra} chars of extra content after JSON object"
                    )
                return extracted
    return text


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
        logger.warning(f"Killed orphaned codex subprocess (pid={proc.pid})")
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(proc.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        logger.error(f"codex subprocess (pid={proc.pid}) did not exit after kill")


class CodexCLIClient:
    """LLM client that uses `codex exec` subprocess with file-based output."""

    def __init__(self):
        self.provider = "codex"
        self.extra_kwargs = {}

    async def async_get_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs,
    ) -> CompletionResult:
        t0 = time.time()
        # on_token silently ignored (no streaming support)
        kwargs.pop("on_token", None)

        system_prompt, user_prompt = _format_messages(messages)
        combined_prompt = system_prompt + JSON_ENFORCEMENT + "\n\n" + user_prompt

        if not combined_prompt.strip():
            raise RuntimeError("Empty prompt — codex exec would produce no output")

        # Create temp file for output
        fd, temp_path = tempfile.mkstemp(suffix=".txt", prefix="codex_out_")
        os.close(fd)

        try:
            cmd = [
                "codex", "exec", "-",       # read prompt from stdin
                "-o", temp_path,             # output to file
                "-m", model,
                "-s", "read-only",
                "--skip-git-repo-check",
                "--ephemeral",
                "--color", "never",
            ]

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                cwd="/tmp",
            )

            # Write combined prompt to stdin and close
            proc.stdin.write(combined_prompt.encode())
            await proc.stdin.drain()
            proc.stdin.close()

            # Wait for completion with timeout
            try:
                await asyncio.wait_for(proc.wait(), timeout=SUBPROCESS_TIMEOUT_SECONDS)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                await _kill_subprocess(proc)
                raise

            # Read stderr
            stderr_data = b""
            try:
                stderr_data = await asyncio.wait_for(proc.stderr.read(), timeout=5.0)
            except asyncio.TimeoutError:
                pass

            if proc.returncode != 0:
                raise RuntimeError(
                    f"codex exec failed (exit {proc.returncode}): {stderr_data.decode()[:500]}"
                )

            # Read output from temp file
            with open(temp_path, "r") as f:
                raw = f.read().strip()

        finally:
            # Cleanup temp file
            try:
                os.unlink(temp_path)
            except OSError:
                pass

        if not raw:
            raise RuntimeError("Empty response from codex exec (output file empty)")

        content = _strip_code_fences(raw)

        logger.info(f"Async LLM response received from codex/{model}")
        return CompletionResult(
            content=content,
            reasoning="",
            model=f"codex/{model}",
            duration_s=time.time() - t0,
        )

    def get_completion(
        self,
        messages: List[Dict[str, str]],
        model: str,
        **kwargs,
    ) -> CompletionResult:
        return asyncio.run(self.async_get_completion(messages, model, **kwargs))
