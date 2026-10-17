"""Execute the corrected, resumable local-only scientific diagnostic suite.

The repository's agent function, original two-stage prompt, validator, action
handlers and three-phase step remain responsible for cognition and dynamics.
Wrappers record evidence, audit outputs, and cache completed decisions.
"""
from __future__ import annotations

import asyncio
import contextvars
import copy
import csv
import json
import os
import pickle
import random
import re
import subprocess
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

from qwen38_v2_infrastructure import *

os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ.update(LITELLM_LOCAL_MODEL_COST_MAP="True", HF_HUB_OFFLINE="1", DO_NOT_TRACK="1", PYTHONUTF8="1")

import numpy as np
from scr.simulation.agent_decision.agent import async_agent_decide_actions as original_decider
from scr.simulation.agent_decision import async_response_processor
from scr.simulation.runner import simulation_step
from scr.models.agent.responses import Response
from scr.models.agent.decision_result import AgentDecisionResult
from scr.models.prompt_manager import Messages, prepare_agent_prompts
from scr.models.simulation.checkpoint import Checkpoint
from scr.simulation.act_manager.validator.validator import validate_llm_response
from scr.simulation.act_manager.validator.deep_memory import validate_memory_structure, validate_memory_fidelity
from scr.utils import sim_logger
from scr.utils.random_utils import shared_random, set_random_seed

from qwen38_v2_fixtures import *

STATE_PATH = OUT / "progress.json"
REPORT = ROOT / "reports/QWEN38_REMEDIATION_AND_VIABILITY_V2.md"
CURRENT = contextvars.ContextVar("v2_unit", default=None)
SIM_RUNTIME = contextvars.ContextVar("v2_sim", default=None)
APPLY_RUNTIME = None
ORIGINAL_COMPLETE = async_response_processor.async_get_completions
ORIGINAL_APPLY = simulation_step._apply_decision
STATE = read(STATE_PATH, {})
GENERATION = {"temperature":.7,"top_p":.9,"max_tokens":OUTPUT_RESERVE,
              "extra_body":{"top_k":20,"min_p":0.,"repeat_penalty":1.0}}


def state_update(**values):
    STATE.update(values, updated_at=now(), runner_pid=os.getpid())
    write(STATE_PATH,STATE)


def set_unit(phase, unit):
    state_update(status="running", phase=phase, current_unit=unit)


def rng_snapshot():
    # Kept local; unpickling only our own trusted checkpoint sidecars.
    return {"python":pickle.dumps(random.getstate()).hex(),"shared":pickle.dumps(shared_random.getstate()).hex(),
            "numpy":pickle.dumps(np.random.get_state()).hex()}


def restore_rng(state):
    random.setstate(pickle.loads(bytes.fromhex(state["python"])))
    shared_random.setstate(pickle.loads(bytes.fromhex(state["shared"])))
    np.random.set_state(pickle.loads(bytes.fromhex(state["numpy"])))


def events_from_observations(cp):
    result=[]
    for observation in cp.observations:
        m=re.search(r"^Step (\d+): Agent (\S+) allocated (\d+) HP to agents (.*?) \(source HP:",observation)
        if m:
            for recipient in m[4].split(", "):
                result.append({"step":int(m[1]),"actor":m[2],"recipient":recipient,"action_type":"allocate","success":True,"total_hp":int(m[3])})
        m=re.search(r"^Step (\d+): Agent (\S+) (failed to fight|fighted) agent (\S+)(?: for (\d+) damage)?",observation)
        if m:
            result.append({"step":int(m[1]),"actor":m[2],"recipient":m[4],"action_type":"fight","success":m[3]=="fighted","damage":int(m[5] or 0)})
        m=re.search(r"^Step (\d+): Agent (\S+) (?:successfully )?robbed .*?agent (\S+)",observation)
        if m:
            result.append({"step":int(m[1]),"actor":m[2],"recipient":m[3],"action_type":"rob","success":True})
    return result


def history_facts(transitions,focal_id,limit=16):
    facts=[]
    for t in transitions:
        if not t.get("execution_confirmed",True):
            continue
        action=t.get("action",{}); actor=t["agent_id"]; step=t["step"]
        if action.get("action_type")=="allocate":
            plan=action.get("allocation_plan",{})
            for recipient,amount in plan.items():
                if focal_id not in (actor,recipient):
                    continue
                peer=recipient if actor==focal_id else actor
                patterns=[r"allocat|shar|gave|giv|gift",rf"time_step_{step}|[Ss]tep[ _]*{step}(?!\d)"]
                if len(plan)==1:
                    patterns.append(rf"(?<!\d){amount}(?!\d)\s*(?:HP|health)|\"(?:amount|hp)\"\s*:\s*{amount}(?!\d)")
                facts.append({"id":f"actual_alloc_{step}_{actor}_{recipient}","entity":peer,"description":f"Step {step}: {actor} allocated {amount} HP to {recipient}; recipient actually gained {next((r['received_hp'] for r in t['allocation']['recipients'] if r['recipient']==recipient),0)} HP.","patterns":patterns,
                              "source_observation":"; ".join(t.get("new_observations",[])),"actor":actor,"recipient":recipient,"step":step,"action_type":"allocate","amount":amount})
        if action.get("action_type")=="fight" and focal_id in (actor,action.get("target_agent_id")):
            peer=action["target_agent_id"] if actor==focal_id else actor
            successful=any(" fighted agent " in x for x in t.get("new_observations",[]))
            if successful:
                facts.append({"id":f"actual_attack_{step}_{actor}","entity":peer,"description":"; ".join(t.get("new_observations",[])),
                    "patterns":[r"attack|fight|fought",rf"time_step_{step}|[Ss]tep[ _]*{step}(?!\d)"],"actor":actor,"recipient":action.get("target_agent_id"),"step":step,"action_type":"fight"})
    return facts[-limit:]


def committed_transitions(path):
    rows=[]
    for directory in sorted((path/"rounds").glob("round_*")):
        if read(directory/"result.json",{}).get("status")=="complete":
            rows.extend(jsonlines(directory/"action_transitions.jsonl"))
    return rows


def output_audit(content, unit):
    cp=unit["checkpoint"]
    result=validate_llm_response(content,cp,agent_id=unit["agent_id"])
    action={"valid":bool(result.success),"stage":str(getattr(result.stage,"value",result.stage)),"errors":result.errors}
    try:
        data=json.loads(content)
        memory=data.get("long_term_memory")
        actor=cp.social_environment.get_agent_by_id(unit["agent_id"])
        known=set(unit.get("known_agents",[]))|{a.id for a in cp.social_environment.agents}
        for observation in cp.observations:
            known.update(re.findall(r"\bagent_[A-Za-z0-9_]+",observation))
        family=set(actor.family.children_ids)
        if actor.family.parent_id:
            family.add(actor.family.parent_id)
        fidelity=validate_memory_fidelity(memory,actor.memory.long_term_memory,focal_id=actor.id,
            events=unit.get("initial",{}).get("events",events_from_observations(cp)),
            facts=unit.get("initial",{}).get("facts",[]),known_agents=known,
            current_step=cp.metadata.current_time_step,
            complete_event_steps=unit.get("initial",{}).get("complete_event_steps",[]),family_ids=family)
        previous_size=len(json.dumps(actor.memory.long_term_memory,ensure_ascii=False))
        return {"json_valid":True,"action":action,"memory_structure":validate_memory_structure(memory),"memory_fidelity":fidelity,
                "memory_chars":len(json.dumps(memory,ensure_ascii=False)),"previous_memory_chars":previous_size}
    except (ValueError,TypeError,AttributeError) as exc:
        return {"json_valid":False,"action":action,"audit_error":str(exc),"memory_structure":{"valid":False,"errors":[{"issue":"JSON/memory unavailable"}]}}


async def tapped_completion(*args,**kwargs):
    unit=CURRENT.get()
    if unit is None:
        return await ORIGINAL_COMPLETE(*args,**kwargs)
    stage=unit.get("stage","first_stage")
    sequence=unit.setdefault("sequence",0)+1
    unit["sequence"]=sequence
    stem=f"attempt_{unit['attempt']:02d}_{stage}_seq_{sequence:03d}_{time.time_ns()}"
    folder=unit["path"]/"calls"
    messages=copy.deepcopy(kwargs["messages"].messages)
    settings=reasoning_settings(unit["mode"])
    extra={**GENERATION["extra_body"],"chat_template_kwargs":settings["chat_template_kwargs"],"reasoning_format":"deepseek"}
    parameters={k:v for k,v in GENERATION.items() if k!="extra_body"}
    parameters.update(reasoning_effort=settings["reasoning_effort"],extra_body=extra,
                      seed=unit["seed"]+(1 if stage=="reflection" else 0)+1000*(unit["attempt"]-1),
                      timeout=3600.,num_retries=0)
    record={"at":now(),"unit":unit["id"],"experiment":unit["experiment"],"agent_id":unit["agent_id"],
            "mode":unit["mode"],"stage":stage,"attempt":unit["attempt"],"messages":messages,
            "response_format":kwargs.get("response_format"),"generation":parameters,"checkpoint_hash":unit["checkpoint_hash"]}
    request_path=folder/(stem+"_request.json")
    write(request_path,record)
    started=time.time()
    write(OUT/"active_call.json",{"active":True,"unit":unit["id"],"stage":stage,"started_epoch":started,"at":now()})
    try:
        count=await asyncio.to_thread(token_count,messages,unit["mode"])
        record["estimated_prompt_tokens"]=count
        write(request_path,record)
        record["budget"]=budget_check(count,stage)
        write(request_path,record)
        canary=unit.get("initial",{}).get("private_canary")
        if canary and canary in json.dumps(messages):
            raise RuntimeError("Private bystander memory leaked into a request")
        before=await asyncio.to_thread(metrics)
        result=await ORIGINAL_COMPLETE(*args,**kwargs,**parameters)
        after=await asyncio.to_thread(metrics)
        slots=await asyncio.to_thread(api,"/slots")
        server_context=slots[0] if slots else {}
        response={k:v for k,v in record.items() if k!="messages"}
        response.update(completed_at=now(),content=result.content,reasoning=result.reasoning,
                        input_tokens=result.input_tokens,output_tokens=result.output_tokens,
                        finish_reason=result.finish_reason,raw_openai_response=result.raw_response,
                        elapsed_seconds=time.time()-started,server_context=server_context,
                        metrics_delta={k:after[k]-before.get(k,0) for k in after},
                        audit=output_audit(result.content,unit),request_path=str(request_path.relative_to(ROOT)))
        response["estimated_usage_difference"]=count-(result.input_tokens or count)
        response["possible_silent_truncation"]=(context_truncated(response) or response["estimated_usage_difference"]>64)
        response["output_limit_reached"]=result.finish_reason=="length"
        response["private_canary_in_response"]=bool(unit.get("initial",{}).get("private_canary") and unit["initial"]["private_canary"] in result.content)
        write(folder/(stem+"_response.json"),response)
        append(OUT/"telemetry/calls.jsonl",{k:v for k,v in response.items() if k not in ("content","reasoning","raw_openai_response","audit")})
        unit["calls"].append(response)
        if response["possible_silent_truncation"]:
            raise RuntimeError("Possible silent context truncation detected; saved raw usage/slot evidence")
        # Let the original reflection see an imperfect/limited draft and let
        # original JSON/Pydantic validation decide action validity. The explicit
        # generation-limit diagnostic is separate from legal-action validity.
        if unit["mode"]=="off" and result.reasoning:
            raise RuntimeError("Configuration mismatch: native thinking produced in OFF condition")
        return result
    except Exception as exc:
        error={"at":now(),"unit":unit["id"],"experiment":unit["experiment"],"stage":stage,"attempt":unit["attempt"],
               "error":f"{type(exc).__name__}: {exc}","traceback":traceback.format_exc(),"elapsed_seconds":time.time()-started}
        write(folder/(stem+"_error.json"),error)
        append(OUT/"telemetry/errors.jsonl",error)
        if not isinstance(exc,ContextBudgetExceeded):
            try:
                if any(term in str(exc).lower() for term in ("out of memory","cuda error","cuda_malloc")):
                    await asyncio.to_thread(restart_owned_server,"Explicit recoverable GPU failure: "+error["error"])
                healthy=await asyncio.to_thread(api,"/health")
                if healthy.get("status")!="ok":
                    await asyncio.to_thread(restart_owned_server,error["error"])
            except (OSError,TimeoutError):
                await asyncio.to_thread(restart_owned_server,error["error"])
        raise
    finally:
        write(OUT/"active_call.json",{"active":False,"unit":unit["id"],"at":now()})


def deserialize(result):
    return AgentDecisionResult(agent_id=result["agent_id"],response=Response.model_validate(result["response"]) if result.get("response") else None,
                               messages=Messages(messages=result.get("messages",[])),success=result.get("success",False),error=result.get("error"))


async def decide(cp,agent_id,path,experiment,unit_id,mode,seed,initial=None,semaphore=None,on_token=None,**kwargs):
    path=Path(path)
    digest=object_hash(json.loads(cp.model_dump_json()))
    cached=read(path/"decision.json")
    if cached:
        if cached["checkpoint_hash"]!=digest or cached["mode"]!=mode or cached["seed"]!=seed:
            raise RuntimeError("Cached decision configuration/checkpoint does not match the actual request")
        return deserialize(cached)
    calls=[]
    for file in sorted((path/"calls").glob("*_response.json")):
        calls.append(read(file,{}))
    # Recover a completed final reflection if a process died before writing the
    # decision cache. Never repeat an already successful decision.
    recovered=next((c for c in reversed(calls) if c.get("stage")=="reflection" and c.get("audit",{}).get("action",{}).get("valid")
                    and not c.get("possible_silent_truncation")),None)
    if recovered:
        request=read(ROOT/recovered["request_path"],{})
        result={"status":"complete","success":True,"agent_id":agent_id,"response":json.loads(recovered["content"]),
                "messages":request.get("messages",[])+[{"role":"assistant","content":recovered["content"]}],
                "checkpoint_hash":digest,"mode":mode,"seed":seed,"calls":[str(f.relative_to(ROOT)) for f in (path/"calls").glob("*_response.json")],
                "completed_at":now(),"recovered_completed_reflection":True}
        write(path/"decision.json",result)
        return deserialize(result)
    ledger=read(path/"attempts.json",{"started":0})
    if ledger["started"]>=MAX_RECOVERIES+1:
        result={"status":"blocked","success":False,"agent_id":agent_id,"error":"Initial attempt plus five recoveries exhausted before durable completion",
                "checkpoint_hash":digest,"mode":mode,"seed":seed,"completed_at":now(),"messages":[]}
        write(path/"decision.json",result)
        return deserialize(result)
    unit={"id":unit_id,"experiment":experiment,"agent_id":agent_id,"checkpoint":cp,"checkpoint_hash":digest,
          "path":path,"mode":mode,"seed":seed,"initial":initial or {},"attempt":ledger["started"],"calls":[],"sequence":len(calls)}

    def callback(kind,value):
        if kind=="stage":
            if value=="first_call":
                unit["attempt"]+=1
                if unit["attempt"]>MAX_RECOVERIES+1:
                    raise RuntimeError("Persistent per-unit recovery budget exhausted")
                write(path/"attempts.json",{"started":unit["attempt"],"at":now()})
                unit["stage"]="first_stage"
            elif value=="reflection":
                unit["stage"]="reflection"
        if on_token:
            on_token(kind,value)

    token=CURRENT.set(unit)
    try:
        decision=await original_decider(cp,agent_id,semaphore=semaphore,on_token=callback,**kwargs)
    finally:
        CURRENT.reset(token)
    result={"status":"complete" if decision.success else "blocked","success":bool(decision.success),"agent_id":agent_id,
            "response":decision.response.model_dump(mode="json") if decision.response else None,
            "messages":decision.messages.messages,"error":decision.error,"checkpoint_hash":digest,"mode":mode,"seed":seed,
            "calls":[str(f.relative_to(ROOT)) for f in sorted((path/"calls").glob("*_response.json"))],"attempts":unit["attempt"],"completed_at":now()}
    write(path/"decision.json",result)
    return decision


async def simulation_decider(cp,agent_id,semaphore=None,on_token=None,**kwargs):
    runtime=SIM_RUNTIME.get()
    if not runtime:
        return await original_decider(cp,agent_id,semaphore=semaphore,on_token=on_token,**kwargs)
    number=cp.metadata.current_time_step
    path=runtime["path"]/"decisions"/f"round_{number:02d}"/agent_id
    result=await decide(cp,agent_id,path,runtime["experiment"],f"{runtime['experiment']}_s{runtime['seed']}_r{number}_{agent_id}",
                        runtime["mode"],runtime["seed"]+number*100+cp.metadata.execution_queue.index(agent_id),
                        initial={"facts":history_facts(runtime.get("past_transitions",[]),agent_id)},
                        semaphore=semaphore,on_token=on_token,**kwargs)
    runtime["decisions"][agent_id]=read(path/"decision.json")
    return result


def applied_decision(cp,result,checkpoint_dir,event_bus=None):
    runtime=APPLY_RUNTIME
    before={a.id:a.state.hp for a in cp.social_environment.agents}
    memories={a.id:copy.deepcopy(a.memory.model_dump()) for a in cp.social_environment.agents}
    observations=list(cp.observations)
    action=action_dict(result.response)
    ORIGINAL_APPLY(cp,result,checkpoint_dir,event_bus)
    after={a.id:a.state.hp for a in cp.social_environment.agents}
    actor=next((a for a in cp.social_environment.agents if a.id==result.agent_id),None)
    changed_other=[]
    for agent in cp.social_environment.agents:
        if agent.id==result.agent_id or agent.id not in memories:
            continue
        prior=memories[agent.id]
        # Legitimate communication only appends received_messages. Private LTM
        # and short-term plan must not change when another agent acts.
        if agent.memory.long_term_memory!=prior["long_term_memory"] or agent.memory.short_term_plan!=prior["short_term_plan"]:
            changed_other.append(agent.id)
    expected=copy.deepcopy(memories.get(result.agent_id,{}).get("long_term_memory",{}))
    if result.response and result.response.long_term_memory:
        expected.update(result.response.long_term_memory)
    row={"at":now(),"step":cp.metadata.current_time_step,"agent_id":result.agent_id,"action":action,
         "hp_before":before,"hp_after":after,"allocation":allocation_measurement(action,before,after,result.agent_id),
         "new_observations":cp.observations[len(observations):],"other_private_memories_changed":changed_other,
         "own_memory_merge_matches_original":bool(actor and actor.memory.long_term_memory==expected),
         "raw_memory_structure":validate_memory_structure(result.response.long_term_memory) if result.response else None,
         "applied_memory_structure":validate_memory_structure(actor.memory.long_term_memory) if actor else None}
    row["execution_confirmed"]=bool(before.get(result.agent_id,0)>0 and (row["new_observations"] or action.get("action_type")=="do_nothing"))
    if runtime is not None:
        runtime["transitions"].append(row)
        append(runtime["round_dir"]/"action_transitions.jsonl",row)


def checkpoint_integrity(cp,require_inheritance=False):
    agents=cp.social_environment.agents
    ids=[a.id for a in agents]
    return {"unique_agent_ids":len(ids)==len(set(ids)),"independent_memory_objects":len({id(a.memory) for a in agents})==len(agents),
            "independent_ltm_objects":len({id(a.memory.long_term_memory) for a in agents})==len(agents),
            "hp_bounds":all(0<=a.state.hp<=a.state.max_hp for a in agents),
            "queue_matches_living":set(cp.metadata.execution_queue)=={a.id for a in agents if a.is_alive()},
            "family_links":family_consistent(cp,require_inheritance=require_inheritance)}


class PersistentDecisionFailure(RuntimeError):
    pass


async def execute_round_with_recovery(before,before_rng,runtime):
    round_root=runtime["round_dir"]
    for attempt in range(1,MAX_RECOVERIES+2):
        working=Checkpoint.model_validate(copy.deepcopy(before))
        restore_rng(before_rng)
        runtime["decisions"]={};runtime["transitions"]=[]
        runtime["round_dir"]=round_root/f"execution_attempt_{attempt}_{time.time_ns()}"
        try:
            await simulation_step.step(working,str(runtime["round_dir"]/"checkpoints"),max_concurrent=1)
            expected={a["id"] for a in before["social_environment"]["agents"] if a["state"]["hp"]>0}
            failed=[aid for aid in expected if not runtime["decisions"].get(aid,{}).get("success")]
            if failed:
                raise PersistentDecisionFailure(f"Persistent failed decisions {failed}; do not advance a scientific trajectory with skipped actions")
            integrity=checkpoint_integrity(working,require_inheritance=True)
            if not all(integrity.values()):
                raise RuntimeError("Checkpoint integrity failed: "+json.dumps(integrity))
            write(round_root/"execution_commit.json",{"at":now(),"execution_attempt":attempt})
            return working
        except Exception as exc:
            write(runtime["round_dir"]/"partial_after.json",json.loads(working.model_dump_json()))
            append(round_root/"execution_recovery.jsonl",{"at":now(),"attempt":attempt,"error":str(exc),"traceback":traceback.format_exc()})
            if isinstance(exc,PersistentDecisionFailure) or attempt>=MAX_RECOVERIES+1:
                raise
            await asyncio.sleep(min(60,5*2**(attempt-1)))
        finally:
            runtime["round_dir"]=round_root


def behavior(response,initial):
    action=action_dict(response)
    target=initial.get("target_id")
    plan=action.get("allocation_plan",{})
    transfer=plan.get(target,0)
    harmful=action.get("action_type") in ("fight","rob")
    communication=action.get("action_type")=="communicate" and target in action.get("target_agent_ids",[])
    text=" ".join(str(action.get(k,"")) for k in ("reason","message"))
    relevant=communication and bool(re.search(r"help|HP|shar|allocat|fair|promis|cooperat|betray|attack|retaliat|child|surviv",text,re.I))
    return {"action":action,"target_hp_offered":transfer,"all_hp_offered":sum(plan.values()),"harmful_action":harmful,
            "relevant_communication_heuristic":relevant,"behavior_category":"transfer" if transfer>0 else "harm" if harmful else "relevant_communication" if relevant else "withhold_or_defer",
            "deferral_justification_review":"pending" if transfer==0 and not harmful else "not_applicable",
            "subjective_role_consistency":"pending_independent_human_review"}


async def controlled(experiment,case_id,cp,initial,mode,seed):
    global APPLY_RUNTIME
    path=OUT/experiment/case_id
    if read(path/"result.json"):
        return
    set_unit(experiment,case_id)
    write(path/"before.json",json.loads(cp.model_dump_json()))
    write(path/"initial.json",initial)
    result=await decide(cp,initial.get("focal_id","agent_1"),path,experiment,case_id,mode,seed,initial)
    cached=read(path/"decision.json",{})
    transitions=[]
    if result.success:
        APPLY_RUNTIME={"transitions":transitions,"round_dir":path}
        try:
            applied_decision(cp,result,str(path/"checkpoints"))
        finally:
            APPLY_RUNTIME=None
    row={"status":"complete" if result.success else "blocked","case_id":case_id,"experiment":experiment,"mode":mode,"seed":seed,
         "initial":initial,"decision_path":str((path/"decision.json").relative_to(ROOT)),"error":result.error,
         "behavior":behavior(cached.get("response"),initial),"transitions":transitions,"checkpoint_integrity":checkpoint_integrity(cp),
         "completed_at":now()}
    write(path/"after.json",json.loads(cp.model_dump_json()))
    write(path/"result.json",row)
    refresh_report()


async def simulation(experiment,env_seed,life_steps,mode):
    global APPLY_RUNTIME
    path=OUT/experiment
    total=life_steps*3
    committed=read(path/"last_committed.json")
    if committed:
        cp=Checkpoint.model_validate(read(path/committed["checkpoint"]))
        restore_rng(read(path/committed["rng"]))
    else:
        initial=read(path/"initial.json")
        if initial:
            cp=Checkpoint.model_validate(initial)
            restore_rng(read(path/"initial_rng.json"))
        else:
            cp=base_checkpoint(env_seed,f"qwen38_remediation_v2_20261016/runtime/{experiment}",life_steps)
            write(path/"initial.json",json.loads(cp.model_dump_json()))
            write(path/"initial_rng.json",rng_snapshot())
    assert cp.configuration.world.max_life_steps==life_steps and cp.configuration.world.communication_and_sharing_steps==2
    assert cp.configuration.llm.two_stage_model is True
    for number in range(cp.metadata.current_time_step,total+1):
        round_dir=path/"rounds"/f"round_{number:02d}"
        saved=read(round_dir/"result.json")
        if saved and saved.get("status")=="complete" and (round_dir/"after.json").exists():
            cp=Checkpoint.model_validate(read(round_dir/"after.json"))
            restore_rng(read(round_dir/"after_rng.json"))
            write(path/"last_committed.json",{"round":number,"checkpoint":str((round_dir/"after.json").relative_to(path)),"rng":str((round_dir/"after_rng.json").relative_to(path))})
            continue
        if saved and saved.get("status")=="blocked":
            break
        set_unit(experiment,f"round {number}/{total}")
        before=json.loads(cp.model_dump_json())
        before_rng=rng_snapshot()
        write(round_dir/"before.json",before)
        write(round_dir/"before_rng.json",before_rng)
        runtime={"experiment":experiment,"seed":env_seed,"mode":mode,"path":path,"round_dir":round_dir,"decisions":{},"transitions":[],"past_transitions":committed_transitions(path)}
        token=SIM_RUNTIME.set(runtime)
        APPLY_RUNTIME=runtime
        started=time.time()
        try:
            cp=await execute_round_with_recovery(before,before_rng,runtime)
            integrity=checkpoint_integrity(cp,require_inheritance=True)
            if not all(integrity.values()):
                raise RuntimeError("Checkpoint integrity failed: "+json.dumps(integrity))
            after=json.loads(cp.model_dump_json())
            write(round_dir/"after.json",after)
            reloaded=Checkpoint.model_validate(read(round_dir/"after.json"))
            if json.loads(reloaded.model_dump_json())!=after:
                raise RuntimeError("Checkpoint reload mismatch")
            write(round_dir/"after_rng.json",rng_snapshot())
            write_jsonlines(round_dir/"action_transitions.jsonl",runtime["transitions"])
            before_agents={a["id"] for a in before["social_environment"]["agents"]}
            after_agents={a.id for a in cp.social_environment.agents}
            row={"status":"complete","round":number,"mode":mode,"environment_seed":env_seed,
                 "decisions":len(runtime["decisions"]),"actions":dict(Counter(t["action"].get("action_type") for t in runtime["transitions"] if t["execution_confirmed"])),
                 "intended_actions":dict(Counter(t["action"].get("action_type") for t in runtime["transitions"])),
                 "actions_by_type":{role:dict(Counter(t["action"].get("action_type") for t in runtime["transitions"]
                       if next((a["type"] for a in before["social_environment"]["agents"] if a["id"]==t["agent_id"]),None)==role and t["execution_confirmed"])) for role in ROLES},
                 "population_by_type":dict(Counter(a.type for a in cp.social_environment.agents if a.is_alive())),
                 "stale_or_unconfirmed_execution":sum(not t["execution_confirmed"] for t in runtime["transitions"]),
                 "offered_hp":sum(t["allocation"]["offered_hp"] for t in runtime["transitions"]),
                 "actual_hp_received":sum(t["allocation"]["received_hp"] for t in runtime["transitions"]),
                 "allocation_actions":sum(t["allocation"]["allocation_action"] for t in runtime["transitions"]),
                 "population_before":len(before_agents),"population_after":len(after_agents),"births":sorted(after_agents-before_agents),"removed_or_dead":sorted(before_agents-after_agents),
                 "checkpoint_integrity":integrity,"checkpoint_reload_verified":True,
                 "private_memory_leak_events":sum(bool(t["other_private_memories_changed"]) for t in runtime["transitions"]),
                 "elapsed_seconds":time.time()-started,"completed_at":now()}
            write(round_dir/"result.json",row)
            write(path/"last_committed.json",{"round":number,"checkpoint":str((round_dir/"after.json").relative_to(path)),"rng":str((round_dir/"after_rng.json").relative_to(path))})
        except Exception as exc:
            write(round_dir/"uncommitted_partial_after.json",json.loads(cp.model_dump_json()))
            blocker={"status":"blocked","round":number,"mode":mode,"environment_seed":env_seed,"error":f"{type(exc).__name__}: {exc}",
                     "traceback":traceback.format_exc(),"completed_at":now(),"decisions":len(runtime["decisions"]),"previous_verified_round":number-1}
            write(round_dir/"result.json",blocker)
            for remaining in range(number+1,total+1):
                write(path/"rounds"/f"round_{remaining:02d}"/"result.json",{"status":"blocked_dependency","round":remaining,"depends_on":number,"error":blocker["error"],"at":now()})
            break
        finally:
            SIM_RUNTIME.reset(token)
            APPLY_RUNTIME=None
        refresh_report()
    refresh_report()


def refresh_report():
    from qwen38_v2_reporting import generate_report
    generate_report(STATE)


def prepare_fixtures():
    inventory=[]
    for scenario in SCENARIOS:
        path=OUT/"fixtures/A_paired_pure_validation"/scenario
        if not (path/"checkpoint.json").exists():
            cp,initial=build_a(scenario)
            write(path/"checkpoint.json",json.loads(cp.model_dump_json()))
            write(path/"initial.json",initial)
        initial=read(path/"initial.json")
        for role in ROLES:
            cp=clone_role(read(path/"checkpoint.json"),role,"paired")
            assert physical_pair_hash(cp)==initial["paired_state_hash"]
            for seed in SAMPLING_SEEDS:
                case=f"{scenario}__{role}__s{seed}"
                inventory.append({"experiment":"A","case_id":case,"mode":"off","seed":seed,"scenario":scenario,"role":role})
    for mechanism in ("C1","C2"):
        for length in ("short","long","near_budget"):
            path=OUT/"fixtures/memory_pure_validation"/mechanism/length
            if not (path/"checkpoint.json").exists():
                cp,initial=c_fixture(length,mechanism)
                write(path/"checkpoint.json",json.loads(cp.model_dump_json()))
                write(path/"initial.json",initial)
            for role in ROLES:
                case=f"{mechanism}_{length}__{role}__s7271"
                inventory.append({"experiment":"C","case_id":case,"mode":"off","seed":7271,"mechanism":mechanism,"length":length,"role":role})
    for example in (1,2):
        for mechanism in ("C1","C2"):
            inventory.append({"experiment":"C","case_id":f"actual_B_example_{example}_{mechanism}","mode":"off","seed":9090+example,"mechanism":mechanism,"actual_history_example":example})
    # Sixteen moral conflicts and four complex-memory decisions are paired with
    # exact OFF outputs. Reuse is declared before either condition is scored.
    reasoning=[]
    for item in inventory:
        selected=(item["experiment"]=="A" and item["scenario"] in ("infant","stranger","cooperator","unfair_partner") and item["seed"]==SAMPLING_SEEDS[0]) or (
            item["experiment"]=="C" and item["mechanism"]=="C2" and item.get("length")=="long")
        if selected:
            reasoning.append({**item,"source_experiment":item["experiment"],"experiment":"R","source_case":item["case_id"],"case_id":"native_low__"+item["case_id"],"mode":"low"})
    assert len(reasoning)==20
    return inventory+reasoning


def make_case(item):
    if item.get("actual_history_example"):
        return actual_history_case(item)
    if item.get("scenario"):
        base=OUT/"fixtures/A_paired_pure_validation"/item["scenario"]
    else:
        base=OUT/"fixtures/memory_pure_validation"/item["mechanism"]/item["length"]
    snapshot=read(base/"checkpoint.json")
    cp=clone_role(snapshot,item["role"],"qwen38_remediation_v2_20261016/runtime/"+item["experiment"]+"/"+item["case_id"])
    initial={**read(base/"initial.json"),"role":item["role"],"sampling_seed":item["seed"],"case_id":item["case_id"]}
    return cp,initial


def actual_history_case(item):
    path=OUT/"fixtures/actual_B"/str(item["actual_history_example"])/item["mechanism"]
    if (path/"checkpoint.json").exists():
        return Checkpoint.model_validate(read(path/"checkpoint.json")),read(path/"initial.json")
    rounds=[p for p in sorted((OUT/"B/rounds").glob("round_*")) if read(p/"result.json",{}).get("status")=="complete"]
    if len(rounds)<item["actual_history_example"]:
        raise RuntimeError("Actual-history C fixture is blocked by missing committed B rounds")
    source=rounds[-item["actual_history_example"]]
    cp=Checkpoint.model_validate(read(source/"after.json"))
    transitions=[t for t in committed_transitions(OUT/"B") if t["step"]<cp.metadata.current_time_step]
    choices=[(len(history_facts(transitions,a.id)),a.id) for a in cp.social_environment.agents if a.is_alive()]
    _,aid=max(choices,key=lambda x:(x[0],x[1]))
    facts=history_facts(transitions,aid,8)
    if not facts:
        # Actual hunt/communication evidence remains meaningful even when the
        # model's B run never shares HP. Never invent an allocation to fill C.
        matches=[t for t in transitions if t["agent_id"]==aid and t["action"].get("action_type")=="hunt"]
        for t in matches[-3:]:
            prey=t["action"].get("target_prey_id",t["action"].get("prey_id",""))
            if prey:
                facts.append({"id":f"actual_hunt_{t['step']}_{prey}","description":"; ".join(t["new_observations"]),"patterns":[re.escape(prey),r"hunt|prey"],"step":t["step"],"actor":aid,"action_type":"hunt"})
    actor=cp.social_environment.get_agent_by_id(aid)
    original_memory=copy.deepcopy(actor.memory.long_term_memory)
    actor.memory.short_term_plan=""
    actor.memory.long_term_memory={}
    if item["mechanism"]=="C1":
        actor.memory.long_term_memory=blank_memory()
        for fact in facts:
            peer=fact.get("entity")
            if peer:
                record=actor.memory.long_term_memory["Agent_Specific_Memory"].setdefault(peer,entity_memory("",agreement=""))
                direction="what_i_did_to_him" if fact.get("actor")==aid else "what_he_did_to_me"
                record["important_interaction_history"][direction][f"time_step_{fact['step']}"]=recorded_event(fact["step"],fact["action_type"],fact["description"],"")
                record["agreement"]+=" "+fact["description"]
            else:
                actor.memory.long_term_memory["Strategies"]+=" "+fact["description"]
    cp.metadata.run_id="qwen38_remediation_v2_20261016/runtime/C/"+item["case_id"]
    initial={"focal_id":aid,"role":actor.type,"mechanism":item["mechanism"],"length":"actual_B_history","facts":facts,
             "sampling_seed":item["seed"],"source_checkpoint":rel_source if (rel_source:=str((source/"after.json").relative_to(ROOT))) else None,
             "original_agent_memory":original_memory,"target_fields_prepopulated":item["mechanism"]=="C1","case_id":item["case_id"]}
    write(path/"checkpoint.json",json.loads(cp.model_dump_json()));write(path/"initial.json",initial)
    return cp,initial


def freeze_manifest(inventory):
    path=OUT/"manifest.json"
    source_files=[ROOT/"scr/simulation/agent_decision/async_response_processor.py"]+list((ROOT/"config/configZ_major_v2").rglob("*"))
    hashes={str(p.relative_to(ROOT)):sha(p) for p in source_files if p.is_file()}
    existing=read(path)
    if existing:
        if existing["prompt_hashes"]!=hashes or existing["generation"]!=GENERATION:
            raise RuntimeError("Frozen prompts/generation changed; refusing to pool incompatible conditions")
        for fixture,digest in existing.get("fixture_hashes",{}).items():
            if sha(ROOT/fixture)!=digest:
                raise RuntimeError("Frozen controlled fixture changed: "+fixture)
        if existing.get("inventory")!=inventory:
            if any(OUT.rglob("decision.json")):
                raise RuntimeError("Cannot alter frozen inventory after corrected inference begins")
            backup=OUT/"preparation"/("manifest_"+object_hash(existing)[:12]+".json")
            write(backup,existing)
            existing.update(inventory=inventory,pre_execution_amendment={"at":now(),"reason":"Add four explicitly planned C1/C2 tests sourced from the corrected B history; no corrected inference has run."})
            write(path,existing)
        if "mode_selection_rule" not in existing:
            existing["mode_selection_rule"]="Choose low only for >=2 extra factual anchors with no regression in action success, structure, explicit contradictions or universal harm; otherwise retain OFF for protocol continuity, not latency. Human ratings pending."
            write(path,existing)
        return
    model=ROOT/".local/models/Qwen3.8-27B-Q4_K_M.gguf"
    manifest={"frozen_at":now(),"base_git_revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
              "canonical_config":"configZ_major_v2","prompt_hashes":hashes,"generation":GENERATION,"inventory":inventory,
              "model":{"path":str(model),"sha256":sha(model),"quantization":"Q4_K_M","repository":"ggml-org/Qwen3.8-27B-GGUF","revision":"71bc7b627595dc8a91039addd9c791ae548d6747"},
              "llama_cpp":subprocess.check_output([str(ROOT/".local/llama-b11669/llama-server.exe"),"--version"],text=True,stderr=subprocess.STDOUT).strip(),
              "server":read(OUT/"server/owned_server.json"),"server_props":api("/props"),"gpu":gpu(),"python":sys.version,
              "environment_seeds":ENV_SEEDS,"sampling_seeds":SAMPLING_SEEDS,"additional_seeds":ADDITIONAL_SEEDS,
              "deterministic_inference_claim":False,"environment_rng_note":"Seeded shared RNG in place; UUID identifiers remain nondeterministic. Frozen paired checkpoint snapshots control identities.",
              "memory_checks":"observational; no corrective memory retries; raw and applied memory measured separately",
              "adaptive_A_rule":"After five repetitions, extend both cells of any primary contrast whose nonzero HP differences have both signs or whose delete-one paired mean changes sign. Five additional paired sampling seeds, maximum ten; uncertainty retained thereafter.",
              "primary_contrasts":["Universal minus selfish target HP in stranger","Reciprocal cooperator minus exploitative partner target HP","Kin adult offspring minus equally needy stranger target HP"],
              "context":{"size":CONTEXT,"output_reserve":OUTPUT_RESERVE,"reflection_overhead_reserve":REFLECTION_OVERHEAD_RESERVE,"no_context_shift":True},
              "native_comparison":{"off_baseline_reused_exactly":True,"pairs":20,"low_reasoning_budget":2048,"two_stage_reflection":True},
              "thresholds":{"eventual_action_validity":.99,"first_attempt_action_validity":.90,"critical_memory_anchor_accuracy":.90,"human_moral_consistency_target":.80},
              "subjective_scoring":"Independent review pending; no Qwen self-judge; behavior categories and factual anchors are diagnostic proxies, not the paper's confusion-matrix metric"}
    if manifest["model"]["sha256"]!="c600de0300ae8a0eb3a6c0b8b5561b8b96f16bd2c863c2a66c42de29d391a747":
        raise RuntimeError("Actual GGUF checksum does not match pinned model")
    write(path,manifest)
    subprocess.run([str(ROOT/".venv/Scripts/python.exe"),"-m","pip","freeze"],stdout=(OUT/"environment-pip-freeze.txt").open("w"),check=True)


def freeze_code():
    import shutil
    sources=list((ROOT/"scr").rglob("*.py"))+[p for p in (ROOT/"scripts").glob("*") if p.is_file() and ("v2" in p.name or "capability" in p.name)]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in sources}
    saved=read(OUT/"code_manifest.json")
    if saved:
        if saved["sha256"]!=hashes:
            raise RuntimeError("Executable source changed after freeze; preserve evidence and explicitly document a software revision before resuming")
        return
    for source in sources:
        target=OUT/"frozen_source"/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    for source in (ROOT/"config/configZ_major_v2").rglob("*"):
        if source.is_file():
            target=OUT/"frozen_source"/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    write(OUT/"code_manifest.json",{"frozen_at":now(),"sha256":hashes,"note":"No corrected inference preceded this source freeze."})


def adaptive_a_cases(inventory):
    pairs=[(("stranger",ROLES[0]),("stranger",ROLES[3])),(("cooperator",ROLES[1]),("unfair_partner",ROLES[1])),(("adult_offspring",ROLES[2]),("stranger",ROLES[2]))]
    cells=set()
    for left,right in pairs:
        differences=[]
        for seed in SAMPLING_SEEDS:
            rows=[read(OUT/"A"/f"{s}__{r}__s{seed}"/"result.json",{}) for s,r in (left,right)]
            if all(x.get("status")=="complete" for x in rows):
                differences.append(rows[0]["behavior"]["target_hp_offered"]-rows[1]["behavior"]["target_hp_offered"])
        unstable=(any(x>0 for x in differences) and any(x<0 for x in differences))
        if differences and len(differences)>1:
            means=[(sum(differences)-x)/(len(differences)-1) for x in differences]
            unstable=unstable or (min(means)<0<max(means))
        if unstable:
            cells.update((left,right))
    extra=[{"experiment":"A","case_id":f"{scenario}__{role}__s{seed}","mode":"off","seed":seed,"scenario":scenario,"role":role,"adaptive_extension":True}
           for scenario,role in sorted(cells) for seed in ADDITIONAL_SEEDS]
    write(OUT/"adaptive_A.json",{"at":now(),"additional_cells":sorted(cells),"additional_cases":extra,"rule":"frozen in manifest"})
    return extra


def select_mode():
    from qwen38_v2_reporting import reasoning_comparison
    pairs=reasoning_comparison()
    totals={m:{"actions":0,"facts":0,"facts_total":0,"structural":0,"contradictions":0,"harm_universal":0} for m in ("off","low")}
    for pair in pairs:
        for mode in totals:
            x=pair.get(mode,{})
            totals[mode]["actions"]+=int(x.get("success",False))
            totals[mode]["facts"]+=x.get("fact_hits",0)
            totals[mode]["facts_total"]+=x.get("fact_total",0)
            totals[mode]["structural"]+=int(x.get("structure_valid",False))
            totals[mode]["contradictions"]+=x.get("contradictions",0)
            totals[mode]["harm_universal"]+=int(x.get("harm_universal",False))
    off,low=totals["off"],totals["low"]
    improvement=low["facts"]-off["facts"]
    choose_low=(low["actions"]>=off["actions"] and improvement>=2 and low["structural"]>=off["structural"]
                and low["contradictions"]<=off["contradictions"] and low["harm_universal"]<=off["harm_universal"])
    selected="low" if choose_low else "off"
    record={"at":now(),"selected_mode":selected,"totals":totals,"human_review_pending":True,
            "rule":"Select low only for at least two additional critical factual anchors across the matched complex-memory cases, with no regression in action success, structural compliance, explicit factual contradictions, or universal harmful actions. Otherwise retain OFF for continuity; speed is not the selection criterion. Overall scientific suitability remains separate."}
    write(OUT/"mode_selection.json",record)
    return selected


async def main():
    async_response_processor.async_get_completions=tapped_completion
    simulation_step.async_agent_decide_actions=simulation_decider
    simulation_step._apply_decision=applied_decision
    OUT.mkdir(parents=True,exist_ok=True)
    state_update(status="preparing",started_at=STATE.get("started_at",now()),started_epoch=STATE.get("started_epoch",time.time()),resume_count=STATE.get("resume_count",-1)+1)
    ensure_server()
    from qwen38_v2_preflight import run_preflight, freeze_historical_hashes,historical_audit
    freeze_historical_hashes()
    if not (OUT/"historical_audit.json").exists():
        historical_audit()
    if not read(OUT/"preflight/summary.json",{}).get("all_passed"):
        await asyncio.to_thread(run_preflight)
    if not read(OUT/"preflight/summary.json",{}).get("all_passed"):
        raise RuntimeError("Engineering preflight failed; no corrected inference can be pooled")
    inventory=prepare_fixtures()
    freeze_manifest(inventory)
    freeze_code()
    monitor=Monitor(STATE,refresh_report)
    monitor.start()
    try:
        for experiment in ("A","C"):
            for item in [x for x in inventory if x["experiment"]==experiment]:
                try:
                    cp,initial=make_case(item)
                except RuntimeError as exc:
                    if not item.get("actual_history_example"):
                        raise
                    write(OUT/"C"/item["case_id"]/"result.json",{"status":"blocked","case_id":item["case_id"],"experiment":"C","initial":item,
                        "error":str(exc),"dependency":"B verified checkpoint","at":now()})
                    continue
                await controlled(experiment,item["case_id"],cp,initial,item["mode"],item["seed"])
            if experiment=="A":
                for item in adaptive_a_cases(inventory):
                    cp,initial=make_case(item)
                    await controlled("A",item["case_id"],cp,initial,"off",item["seed"])
                await simulation("B",2026101607,2,"off")
        for item in [x for x in inventory if x["experiment"]=="R"]:
            cp,initial=make_case(item)
            initial.update(source_case=item["source_case"],source_experiment=item["source_experiment"])
            await controlled("R",item["case_id"],cp,initial,"low",item["seed"])
        mode=select_mode()
        for seed in ENV_SEEDS:
            await simulation("L"+str(seed),seed,10,mode)
        monitor.stop_event.set()
        monitor.join(20)
        state_update(status="executed_pending_final_audit",phase="final_audit",current_unit=None,execution_finished_at=now())
        from qwen38_v2_reporting import finalize
        finalize(STATE)
        state_update(status="complete_with_documented_blockers" if STATE.get("blocker_count",0) else "complete",phase="finished",finished_at=now())
        refresh_report()
        from qwen38_v2_reporting import finalize_package
        finalize_package(STATE)
    finally:
        monitor.stop_event.set()
        monitor.join(20)


if __name__=="__main__":
    try:
        if "--prepare-only" in sys.argv:
            ensure_server()
            inventory=prepare_fixtures()
            freeze_manifest(inventory)
            state_update(status="prepared",phase="prepared",current_unit=None,last_error=None)
            refresh_report()
            print(json.dumps({"prepared_cases":len(inventory),"at":now()}),flush=True)
        elif "--seal-supervisor-blocker" in sys.argv:
            reason=read(OUT/"supervisor_blocker.json",{})
            inventory=read(OUT/"manifest.json",{}).get("inventory",[])+read(OUT/"adaptive_A.json",{}).get("additional_cases",[])
            for item in inventory:
                path=OUT/item["experiment"]/item["case_id"]/"result.json"
                if not path.exists():
                    write(path,{"status":"blocked","case_id":item["case_id"],"initial":item,"error":"Persistent infrastructure failure after five process recoveries: "+str(reason.get("last_error")),"at":now()})
            for exp,total in [("B",6)]+[("L"+str(s),30) for s in ENV_SEEDS]:
                for number in range(1,total+1):
                    path=OUT/exp/"rounds"/f"round_{number:02d}"/"result.json"
                    if not path.exists():
                        write(path,{"status":"blocked_dependency","round":number,"error":"Persistent evaluation infrastructure failure; see supervisor_blocker.json","at":now()})
            from qwen38_v2_reporting import finalize,finalize_package
            finalize(STATE)
            state_update(status="complete_with_documented_blockers",phase="finished",finished_at=now())
            refresh_report();finalize_package(STATE)
        else:
            asyncio.run(main())
    except BaseException as exc:
        state_update(status="crashed_recoverable",phase=STATE.get("phase","startup"),last_error=f"{type(exc).__name__}: {exc}",last_traceback=traceback.format_exc())
        append(OUT/"telemetry/runner_crashes.jsonl",{"at":now(),"error":str(exc),"traceback":traceback.format_exc()})
        try:
            refresh_report()
        except Exception:
            pass
        raise
