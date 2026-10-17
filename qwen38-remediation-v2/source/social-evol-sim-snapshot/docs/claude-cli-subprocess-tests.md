# claude -p Subprocess LLM Backend — Manual Test Results

**Date**: 2026-02-20
**Claude Code version**: 2.1.50
**Environment**: macOS (Darwin 25.3.0)

## Summary

| # | Test | Result | Key Finding |
|---|------|--------|-------------|
| 1 | Smoke test | PASS | Basic `claude -p` works, exit code 0 |
| 2 | JSON output format | PASS | `--output-format json` returns JSON array with metadata wrapper |
| 3 | System prompt | PASS | `--system-prompt` correctly influences behavior |
| 4 | JSON schema | PARTIAL | `--json-schema` alone doesn't force JSON output; prompt engineering required |
| 5 | Long system prompt | PASS | ~29KB system prompt works, but response has markdown fences |
| 6 | Full Response schema | PASS | Pydantic `Response.model_validate_json()` succeeds |
| 7 | Concurrent subprocesses | PASS | 3 parallel processes work without conflicts |
| 8 | Error recovery | PARTIAL | Empty input hangs; long input and special chars work fine |
| 9 | Multi-round (--resume) | PASS | `--session-id` + `--resume` preserves context across rounds |
| 10 | End-to-end integration | PASS | Real 48KB prompt -> valid Response through Pydantic validation |

## Critical Findings for Implementation

### 1. MUST unset CLAUDECODE env var
`claude -p` refuses to run inside another Claude Code session. The subprocess launcher must:
```python
env = os.environ.copy()
env.pop("CLAUDECODE", None)
```

### 2. --json-schema is unreliable for forcing JSON output
- With simple schemas: sometimes works, sometimes returns text
- With complex schemas ($ref, anyOf): returns empty output
- **Solution**: Use system prompt instruction to enforce JSON output:
  ```
  CRITICAL: You MUST respond with ONLY a raw JSON object. No markdown, no code fences, no explanation.
  ```

### 3. Response often wrapped in markdown code fences
Even with explicit "no code fences" instructions, Claude sometimes wraps output in ` ```json ... ``` `. The parser must strip these:
```python
import re
match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw, re.DOTALL)
json_str = match.group(1) if match else raw
```

### 4. Use --tools "" to disable tool use
Without `--tools ""`, `claude -p` loads Claude Code's full toolset (file editing, bash, etc.), which:
- Increases latency (MCP server startup)
- May cause Claude to use tools instead of responding directly
- Wastes tokens on tool descriptions in context

### 5. Use --output-format text (default) for raw response
- `--output-format text`: Returns Claude's response as plain text (what we want)
- `--output-format json`: Returns a JSON array with metadata wrapper (type, session_id, cost, etc.)
  - Need to extract `.result` from the last object where `type == "result"`
  - The `result` field itself is text, not structured JSON

**Recommendation**: Use `--output-format text` and parse the raw text output.

### 6. Process startup overhead is significant
Each `claude -p` invocation has ~1.3-1.8s of Node.js/CLI startup overhead (measured from `user` time in concurrent test). For 8 agents per step, this adds ~12-15s of cumulative process startup time, though parallelism mitigates wall-clock impact.

### 7. Empty stdin causes hang
`echo "" | claude -p` hangs indefinitely. Ensure prompts are never empty.

### 8. cd out of project directory to avoid project context
When running from the project directory, Claude Code loads `CLAUDE.md` and project context, which can override system prompt behavior. Run subprocess from `/tmp` or another neutral directory.

---

## Detailed Test Results

### Test 1: Smoke Test

**Command**:
```bash
unset CLAUDECODE && echo "Say hello" | claude -p
```

**Output**:
```
Hello! How can I help you today?
```

**Exit code**: 0
**Result**: PASS

---

### Test 2: JSON Output Format

**Command**:
```bash
echo "Return a JSON object with keys: name, age" | claude -p --output-format json > /tmp/test2.json
```

**Output structure** (JSON array):
```json
[
  {"type":"system", "subtype":"init", "session_id":"...", "tools":[...], ...},
  {"type":"assistant", "message":{"content":[{"type":"thinking",...}, {"type":"text","text":"```json\n{...}\n```"}],...}},
  {"type":"result", "subtype":"success", "result":"```json\n{\"name\":\"Alice\",\"age\":30}\n```", "total_cost_usd":0.047, ...}
]
```

**Key finding**: `--output-format json` wraps everything in metadata. The `result` field in the `type: "result"` entry contains Claude's text response (with markdown fences, not raw JSON).

**Result**: PASS

---

### Test 3: System Prompt

**Command**:
```bash
echo "What is your role?" | claude -p --system-prompt "You are a pirate captain named Blackbeard. Always respond in pirate speak."
```

**Output** (excerpt):
```
Ahoy there, ye scallywag! I be **Captain Blackbeard**, the most fearsome pirate captain...
```

**Result**: PASS — System prompt clearly influences behavior.

---

### Test 4: JSON Schema

**Attempts**:

| Variant | Flags | Result |
|---------|-------|--------|
| 4a | `--output-format json --json-schema '{...}'` | Text response, no JSON |
| 4b | `--output-format text --json-schema '{...}'` (from /tmp) | Empty output |
| 4c | `--json-schema '{...}' --tools ""` (from /tmp) | Text response, no JSON |
| 4d | `--output-format json --json-schema '{...}' --tools ""` | Text response, no JSON |
| 4e | `--json-schema '{...}' --tools "" --system-prompt "raw JSON only"` | **Raw JSON output** |

**Winning command**:
```bash
echo "..." | claude -p \
  --system-prompt "You are a JSON-only responder. You MUST respond with raw JSON only. No markdown, no explanation, no code fences." \
  --json-schema '{"type":"object",...}' \
  --tools ""
```

**Output**:
```json
{"agent_id":"agent_default","thinking":"...","action":{"action_type":"do_nothing","reason":"..."}}
```

**Result**: PARTIAL — `--json-schema` alone insufficient; system prompt instruction is the actual forcing mechanism.

---

### Test 5: Long System Prompt

**Command**: Combined morality (685 chars) + rules (21654 chars) + strategies (6560 chars) = ~29KB system prompt.

**Output**: Valid agent decision JSON wrapped in markdown fences, plus commentary after the JSON block.

**Result**: PASS — Long prompts work, but fences must be stripped.

---

### Test 6: Full Response Schema with Pydantic Validation

**Command**:
```bash
echo "<agent scenario prompt>" | claude -p \
  --system-prompt "...schema in prompt... MUST respond with raw JSON only" \
  --tools "" --model sonnet
```

**Output**:
```json
{"agent_id":"agent_1","thinking":"...","short_term_plan":{...},"long_term_memory":{...},"action":{"action_type":"hunt","reason":"...","prey_id":"prey_1"}}
```

**Validation**:
```python
response = Response.model_validate_json(raw)
# PASS: agent_id="agent_1", action_type="hunt"
```

**Result**: PASS — This is the critical test. `claude -p` can produce Pydantic-valid Response objects.

---

### Test 7: Concurrent Subprocesses

**Command**: 3 parallel `claude -p` processes via background jobs.

**Results**:
| Agent | Output | Time |
|-------|--------|------|
| 1 | "agent_1" | 9.0s |
| 2 | "agent_2" | 5.1s |
| 3 | "agent_3" | 3.9s |

**Result**: PASS — No conflicts, no auth errors, no output mixing.

---

### Test 8: Error Recovery

| Scenario | Input | Result | Exit Code |
|----------|-------|--------|-----------|
| Empty string | `echo "" \| claude -p` | **HANGS** | N/A |
| Long input (10K chars) | `python -c "print('x'*10000)"` | "Gibberish." | 0 |
| JSON special chars | `{"key": "value with \"quotes\""}` | "VALID" | 0 |

**Result**: PARTIAL — Empty input causes hang. Must guard against empty prompts.

---

### Test 9: Multi-round Conversation

**Round 1**:
```bash
echo "...Return JSON with agent_id and action_type: do_nothing..." | claude -p --session-id "$UUID" --model haiku
```
Output: `{"agent_id":"agent_1","action_type":"do_nothing"}` (with fences)

**Round 2**:
```bash
echo "Reflect on your previous response. What was the action_type?" | claude -p --resume "$UUID" --model haiku
```
Output: `do_nothing`

**Result**: PASS — Context preserved across rounds. Enables two-stage (decision + reflection) workflows.

---

### Test 10: End-to-End Integration

**Input**: Real simulation prompts extracted from `data/messages/0219-075659/step-base/step_1/messages_t1_agent_1.json`
- System prompt: 38,906 chars
- User prompt: 8,738 chars
- Total: ~48KB

**Command**:
```bash
cat /tmp/test10_user.txt | claude -p \
  --system-prompt "$(cat /tmp/test10_system.txt) CRITICAL: respond with raw JSON only" \
  --tools "" --model sonnet
```

**Output**: Valid JSON response (with markdown fences), containing:
- `agent_id`: "agent_1"
- `action`: communicate with agents 2,3,4
- `long_term_memory`: properly structured with all 5 fields
- `short_term_plan`: reasonable multi-step plan

**Validation**:
```python
response = Response.model_validate_json(json_str)  # After stripping fences
# PASS: agent_id="agent_1", action_type="communicate"
```

**Result**: PASS — Full end-to-end validation confirms `claude -p` can replace litellm.

---

## Recommended Implementation Pattern

```python
import asyncio
import json
import os
import re

async def claude_subprocess_completion(
    system_prompt: str,
    user_prompt: str,
    model: str = "sonnet",
) -> str:
    """Call claude -p as subprocess, return raw JSON string."""

    # Add JSON enforcement to system prompt
    system_prompt += "\n\nCRITICAL: Respond with ONLY a raw JSON object. No markdown, no code fences, no explanation."

    env = os.environ.copy()
    env.pop("CLAUDECODE", None)  # Prevent nested session error

    proc = await asyncio.create_subprocess_exec(
        "claude", "-p",
        "--system-prompt", system_prompt,
        "--tools", "",
        "--model", model,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd="/tmp",  # Avoid project context
    )

    stdout, stderr = await proc.communicate(user_prompt.encode())
    raw = stdout.decode().strip()

    if proc.returncode != 0:
        raise RuntimeError(f"claude -p failed: {stderr.decode()}")

    # Strip markdown code fences if present
    match = re.search(r'```(?:json)?\s*(.*?)\s*```', raw, re.DOTALL)
    return match.group(1) if match else raw
```

## Performance Considerations

- **Process startup**: ~1.5s overhead per invocation (Node.js CLI startup)
- **Concurrent processes**: Works well, no auth/rate-limit issues with 3 concurrent
- **Large prompts**: 48KB total prompt size works fine
- **Model selection**: Use `--model haiku` for lower cost/latency, `--model sonnet` for quality
