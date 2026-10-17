"""Exactly two original simulation steps, with local inference and evidence files."""
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
os.environ['DO_NOT_TRACK'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

from scr.api.db_api.connection import disable_db
from scr.models.simulation.checkpoint import Checkpoint
from scr.simulation.event_bus import SimulationEventBus
from scr.simulation.runner.simulation_step import step
from scr.simulation.act_manager.validator.validator import validate_llm_response
from scr.utils import sim_logger
from scr.utils.logger import init_run_logger

BASE = 'http://127.0.0.1:8080'


def metrics():
    with urlopen(BASE + '/metrics', timeout=10) as response:
        raw = response.read().decode()
    return {line.split()[0]: float(line.split()[1])
            for line in raw.splitlines() if line and not line.startswith('#')}


def sample_gpu(stop, samples):
    while not stop.is_set():
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu',
             '--format=csv,noheader,nounits', '--id=0'],
            capture_output=True, text=True, check=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        used, total, utilization = map(int, result.stdout.strip().split(','))
        samples.append(dict(time=time.time(), used_mib=used, total_mib=total,
                            utilization_percent=utilization))
        stop.wait(1)


async def main():
    disable_db()
    with urlopen(BASE + '/health', timeout=10) as response:
        assert json.load(response)['status'] == 'ok'
    checkpoint = Checkpoint.initialize_from_config('configZ_major_v2', {
        'llm.provider': 'llama_cpp',
        'llm.chat_model': 'Qwen3.8-27B-Q4_K_M',
        'llm.two_stage_model': True,
        'llm.async_config.max_concurrent_calls': 1,
        # The existing timeout surrounds BOTH model calls in an attempt.
        'llm.async_config.call_timeout_seconds': 3600.0,
    })
    agents = checkpoint.social_environment.agents
    assert len(agents) == 8
    assert Counter(a.type for a in agents) == {
        'universal_group_focused_moral': 2, 'reciprocal_group_focused_moral': 2,
        'kin_focused_moral': 2, 'reproductive_selfish': 2,
    }
    assert len({id(a.memory) for a in agents}) == 8
    assert len({id(a.memory.long_term_memory) for a in agents}) == 8
    output = ROOT / 'data' / checkpoint.metadata.run_id
    output.mkdir(parents=True, exist_ok=True)
    init_run_logger(output)
    sim_logger.bind(run_id=checkpoint.metadata.run_id)
    (output / 'initial.json').write_text(checkpoint.model_dump_json(indent=2), encoding='utf-8')
    bus = SimulationEventBus()
    bus.subscribe(sim_logger.jsonl_event_bus_subscriber)
    await bus.start()
    samples, checks = [], []
    stop = threading.Event()
    monitor = threading.Thread(target=sample_gpu, args=(stop, samples), daemon=True)
    before = metrics()
    monitor.start()
    started = time.perf_counter()
    try:
        for _ in range(2):
            current = checkpoint.metadata.current_time_step
            # Preserve the frozen state for independent contextual revalidation.
            frozen = checkpoint.model_copy(deep=True)
            previous = {a.id: a.memory.model_dump() for a in agents}
            messages = await step(checkpoint, str(ROOT / 'data'), max_concurrent=1, event_bus=bus)
            (output / f'messages_step_{current}.json').write_text(
                json.dumps({key: value.messages for key, value in messages.items()}, indent=2),
                encoding='utf-8')
            assert set(messages) == {a.id for a in agents}
            for agent in agents:
                conversation = messages[agent.id].messages
                assistant = [m['content'] for m in conversation if m['role'] == 'assistant']
                assert len(assistant) >= 2 and len(assistant) % 2 == 0, agent.id
                # JSON grammar applies to every draft and every reflection, including retries.
                for content in assistant:
                    assert isinstance(json.loads(content), dict), agent.id
                assert any(m['role'] == 'user' and 'Reflect:' in m['content'] for m in conversation)
                validated = validate_llm_response(assistant[-1], frozen, agent_id=agent.id)
                assert validated.success, (agent.id, validated.error_message)
                assert len(agent.response_history) == len(checks) // 8 + 1, agent.id
                assert agent.response_history[-1] == validated.response, agent.id
                assert agent.memory.long_term_memory != previous[agent.id]['long_term_memory'], agent.id
                for key, value in validated.response.long_term_memory.items():
                    assert agent.memory.long_term_memory[key] == value, (agent.id, key)
                assert agent.memory.short_term_plan == validated.response.short_term_plan
                path = output / 'checkpoints' / 'step-base' / f'step_{current}' / f'checkpoint_t{current}_{agent.id}.json'
                restored = Checkpoint.model_validate_json(path.read_text())
                saved = restored.social_environment.get_agent_by_id(agent.id)
                assert saved.memory.long_term_memory == agent.memory.long_term_memory, agent.id
                assert saved.memory.short_term_plan == agent.memory.short_term_plan, agent.id
                checks.append(dict(step=current, agent_id=agent.id, moral_type=agent.type,
                                   model_calls=len(assistant), reflection=True,
                                   valid_action=validated.response.action.root.action_type,
                                   memory_updated=True, checkpoint_verified=True))
            assert len({id(a.memory) for a in agents}) == 8
            assert len({id(a.memory.long_term_memory) for a in agents}) == 8
            (output / f'after_step_{current}.json').write_text(
                checkpoint.model_dump_json(indent=2), encoding='utf-8')
            print(f'Verified step {current}: eight validated decisions and independent memories', flush=True)
    finally:
        stop.set()
        monitor.join(timeout=5)
        await bus.stop()
        sim_logger.close()
        (output / 'gpu_samples.json').write_text(json.dumps(samples, indent=2), encoding='utf-8')
    after = metrics()
    delta = {key: after[key] - value for key, value in before.items() if key in after}
    report = dict(
        passed=True, run_id=checkpoint.metadata.run_id, simulation_steps=2, agents=8,
        wall_seconds=time.perf_counter() - started,
        gpu_memory_min_mib=min(s['used_mib'] for s in samples),
        gpu_memory_peak_mib=max(s['used_mib'] for s in samples),
        metrics_before=before, metrics_after=after, metrics_delta=delta, checks=checks,
    )
    for label, tokens, seconds in [
        ('generation_tokens_per_second', 'llamacpp:tokens_predicted_total', 'llamacpp:tokens_predicted_seconds_total'),
        ('prompt_tokens_per_second', 'llamacpp:prompt_tokens_total', 'llamacpp:prompt_seconds_total'),
    ]:
        if delta.get(seconds, 0) > 0:
            report[label] = delta[tokens] / delta[seconds]
    (output / 'smoke_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k not in {'checks', 'metrics_before', 'metrics_after', 'metrics_delta'}}, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
