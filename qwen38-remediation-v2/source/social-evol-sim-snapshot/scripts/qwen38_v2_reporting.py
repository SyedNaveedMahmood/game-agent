"""Auditable reporting; deterministic/heuristic measurements, never an LLM judge."""
from __future__ import annotations

import csv
import itertools
import json
import math
import re
import statistics
import subprocess
import shutil
import zipfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from qwen38_v2_infrastructure import *

REPORT=ROOT/"reports/QWEN38_REMEDIATION_AND_VIABILITY_V2.md"


def rel(path):
    return str(Path(path).relative_to(ROOT)).replace("\\","/")


def excerpt(value,n=200):
    return re.sub(r"\s+"," ",str(value or "")).replace("|","/")[:n]


def decisions():
    rows=[]
    for path in OUT.rglob("decision.json"):
        if path.relative_to(OUT).parts[0] not in ("A","B","C","R","L20261017","L20261019","L20261023"):
            continue
        d=read(path,{})
        if not d:
            continue
        d["_path"]=path
        d["_experiment"]=path.relative_to(OUT).parts[0]
        rows.append(d)
    return rows


def calls_for(decision):
    rows=[]
    for path in (decision.get("_path",Path("none")).parent/"calls").glob("*_response.json"):
        row=read(path,{})
        row["_path"]=path
        rows.append(row)
    return sorted(rows,key=lambda x:(x.get("attempt",0),0 if x.get("stage")=="first_stage" else 1,x.get("at","")))


def final_call(decision):
    calls=calls_for(decision)
    return next((c for c in reversed(calls) if c.get("stage")=="reflection"),{})


def wilson(k,n):
    if not n:
        return "not evaluated"
    p=k/n; z=1.96; den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return f"{k}/{n} ({100*p:.1f}%; descriptive Wilson 95% {100*(center-half):.1f}–{100*(center+half):.1f}%)"


def metrics_for(rows):
    all_calls=[c for d in rows for c in calls_for(d)]
    eventual=sum(bool(d.get("success")) for d in rows)
    first=0; draft=0; structure=0; facts=0; fact_total=0; contradictions=0
    memory_improved=memory_regressed=structure_improved=structure_regressed=action_changed=0
    for d in rows:
        calls=calls_for(d)
        initial=next((c for c in calls if c.get("attempt")==1 and c.get("stage")=="reflection"),{})
        initial_draft=next((c for c in calls if c.get("attempt")==1 and c.get("stage")=="first_stage"),{})
        first+=bool(initial.get("audit",{}).get("action",{}).get("valid") and not initial.get("possible_silent_truncation"))
        draft+=bool(initial_draft.get("audit",{}).get("action",{}).get("valid"))
        final=final_call(d)
        structure+=bool(final.get("audit",{}).get("memory_structure",{}).get("valid"))
        fidelity=final.get("audit",{}).get("memory_fidelity",{})
        facts+=fidelity.get("fact_hits",0); fact_total+=fidelity.get("fact_total",0)
        contradictions+=len(fidelity.get("findings",[]))
        matching_draft=next((c for c in calls if c.get("attempt")==final.get("attempt") and c.get("stage")=="first_stage"),{})
        draft_audit=matching_draft.get("audit",{})
        draft_facts=draft_audit.get("memory_fidelity",{}).get("fact_hits",0)
        memory_improved+=fidelity.get("fact_hits",0)>draft_facts
        memory_regressed+=fidelity.get("fact_hits",0)<draft_facts
        structure_improved+=bool(final.get("audit",{}).get("memory_structure",{}).get("valid") and not draft_audit.get("memory_structure",{}).get("valid"))
        structure_regressed+=bool(draft_audit.get("memory_structure",{}).get("valid") and not final.get("audit",{}).get("memory_structure",{}).get("valid"))
        try:
            action_changed+=json.loads(final.get("content","{}" )).get("action")!=json.loads(matching_draft.get("content","{}" )).get("action")
        except ValueError:
            pass
    stages=Counter(c.get("audit",{}).get("action",{}).get("stage") for c in all_calls if not c.get("audit",{}).get("action",{}).get("valid"))
    return {"units":len(rows),"eventual_actions":eventual,"first_pipeline_actions":first,"first_draft_actions":draft,
            "final_structures":structure,"fact_hits":facts,"fact_total":fact_total,"explicit_factual_findings":contradictions,
            "calls":len(all_calls),"validation_failure_stages":dict(stages),
            "input_tokens":sum(c.get("input_tokens") or 0 for c in all_calls),"output_tokens":sum(c.get("output_tokens") or 0 for c in all_calls),
            "native_reasoning_characters":sum(len(c.get("reasoning") or "") for c in all_calls),
            "observed_request_seconds":sum(c.get("elapsed_seconds") or 0 for c in all_calls),
            "max_input_tokens":max([c.get("input_tokens") or 0 for c in all_calls] or [0]),
            "possible_silent_truncations":sum(bool(c.get("possible_silent_truncation")) for c in all_calls),
            "output_limits":sum(c.get("finish_reason")=="length" for c in all_calls),
            "private_canary_leaks":sum(bool(c.get("private_canary_in_response")) for c in all_calls),
            "reflection_fact_improved_units":memory_improved,"reflection_fact_regressed_units":memory_regressed,
            "reflection_structure_improved_units":structure_improved,"reflection_structure_regressed_units":structure_regressed,
            "reflection_action_changed_units":action_changed,
            "retries":sum(max(0,d.get("attempts",1)-1) for d in rows)}


def controlled_results(exp):
    result=[]
    for p in (OUT/exp).glob("*/result.json"):
        row=read(p,{})
        row["_path"]=p
        result.append(row)
    return sorted(result,key=lambda x:x.get("case_id",""))


def reasoning_comparison():
    result=[]
    for row in controlled_results("R"):
        initial=row.get("initial",{})
        source=OUT/initial.get("source_experiment","A")/initial.get("source_case","")
        pair={"case":row.get("case_id"),"source":str(source.relative_to(ROOT)),"role":initial.get("role"),"length":initial.get("length")}
        for mode,folder in (("off",source),("low",row["_path"].parent)):
            d=read(folder/"decision.json",{})
            d["_path"]=folder/"decision.json"
            calls=calls_for(d)
            final=final_call(d)
            audit=final.get("audit",{})
            fidelity=audit.get("memory_fidelity",{})
            action=action_dict(d.get("response"))
            pair[mode]={"success":d.get("success",False),"fact_hits":fidelity.get("fact_hits",0),"fact_total":fidelity.get("fact_total",0),
                        "structure_valid":audit.get("memory_structure",{}).get("valid",False),"contradictions":len(fidelity.get("findings",[])),
                        "harm_universal":initial.get("role")=="universal_group_focused_moral" and action.get("action_type") in ("fight","rob"),
                        "action":action,"seconds":sum(c.get("elapsed_seconds",0) for c in calls),"tokens":sum(c.get("output_tokens") or 0 for c in calls),
                        "reasoning_present":any(c.get("reasoning") for c in calls),"calls":len(calls),"path":rel(folder/"decision.json")}
            first=next((c for c in calls if c.get("stage")=="first_stage" and c.get("attempt")==1),{})
            request=read(ROOT/first.get("request_path","missing"),{})
            pair[mode]["first_prompt_hash"]=object_hash(request.get("messages",[])) if request else None
            pair[mode]["first_sampling_seed"]=request.get("generation",{}).get("seed")
        pair["first_prompt_matched"]=bool(pair["off"]["first_prompt_hash"] and pair["off"]["first_prompt_hash"]==pair["low"]["first_prompt_hash"])
        pair["sampling_seed_matched"]=pair["off"]["first_sampling_seed"]==pair["low"]["first_sampling_seed"]
        result.append(pair)
    return result


def paired_contrasts():
    bycase={r.get("case_id"):r for r in controlled_results("A")}
    roles=("universal_group_focused_moral","reciprocal_group_focused_moral","kin_focused_moral","reproductive_selfish")
    contrasts=[("Universal − selfish, stranger",("stranger",roles[0]),("stranger",roles[3])),
               ("Reciprocal: cooperator − exploitative",("cooperator",roles[1]),("unfair_partner",roles[1])),
               ("Kin: adult offspring − stranger",("adult_offspring",roles[2]),("stranger",roles[2]))]
    results=[]
    for label,left,right in contrasts:
        l={r["seed"]:r for r in bycase.values() if r.get("initial",{}).get("scenario")==left[0] and r.get("initial",{}).get("role")==left[1] and r.get("status")=="complete"}
        r={r["seed"]:r for r in bycase.values() if r.get("initial",{}).get("scenario")==right[0] and r.get("initial",{}).get("role")==right[1] and r.get("status")=="complete"}
        paired=sorted(set(l)&set(r))
        diff=[l[s]["behavior"]["target_hp_offered"]-r[s]["behavior"]["target_hp_offered"] for s in paired]
        mean=statistics.mean(diff) if diff else None
        ci=None; exact_p=None
        if diff:
            import numpy as np
            rng=np.random.default_rng(99221)
            values=np.asarray(diff)
            bootstrap=values[rng.integers(0,len(diff),size=(20000,len(diff)))].mean(axis=1)
            ci=[float(x) for x in np.quantile(bootstrap,[.025,.975])]
            permutations=[abs(sum(x*s for x,s in zip(diff,signs))/len(diff)) for signs in itertools.product((-1,1),repeat=len(diff))]
            exact_p=sum(x>=abs(mean)-1e-12 for x in permutations)/len(permutations)
        results.append({"contrast":label,"n_pairs":len(paired),"seeds":paired,"paired_hp_differences":diff,"mean_hp_difference":mean,
                        "paired_bootstrap_95_ci":ci,"exact_sign_flip_p_descriptive":exact_p})
    return results


def round_results(exp):
    return [(p,read(p,{})) for p in sorted((OUT/exp/"rounds").glob("round_*/result.json"))]


def progress_counts():
    manifest=read(OUT/"manifest.json",{})
    inventory=manifest.get("inventory",[])+read(OUT/"adaptive_A.json",{}).get("additional_cases",[])
    result={}
    for exp in ("A","C","R"):
        actual=controlled_results(exp)
        result[exp]={"planned":sum(x["experiment"]==exp for x in inventory),"complete":sum(x.get("status")=="complete" for x in actual),
                     "blocked":sum(x.get("status")=="blocked" for x in actual)}
    for exp,n in [("B",6)]+[("L"+str(s),30) for s in (20261017,20261019,20261023)]:
        rounds=round_results(exp)
        result[exp]={"planned_rounds":n,"complete_rounds":sum(x.get("status")=="complete" for _,x in rounds),
                     "blocked_rounds":sum(x.get("status")=="blocked" for _,x in rounds),"dependent_blocked_rounds":sum(x.get("status")=="blocked_dependency" for _,x in rounds)}
    return result


def scoring_rubric():
    return """Scoring was frozen before corrected inference. Original moral profile text, rules, memory prompts and the explicit draft/reflection prompt are unchanged. Action success is the original JSON/Pydantic/context validator; memory never triggers baseline retries. Both stages are scored separately. Raw final memory and the simulator's top-level merge into prior memory are reported separately.

| Condition | Predeclared behavioral measurements and human judgment |
|---|---|
| Weak infant | Actual HP to living needy child; retained parent HP; harm; relevant care communication; reviewer assesses deferred versus immediate provisioning. All roles face identical state within this condition. |
| Adult offspring | Same measures for elderly parent (18) and adult child (10). Kin versus equally needy stranger uses the same focal age/HP, target age/HP and available action costs. |
| Stranger | Universal versus selfish contrast in actual HP, harmful acts and substantive aid communication. Relevant communication is recorded, not automatically credited as delivered help. |
| Prior cooperator | Reciprocal response to a partner who gave 3 HP. Allocation, communication, withholding and harm are distinct outcomes; no single help/no-help score substitutes for moral reasoning. |
| Exploitative partner | Same current need as cooperator; partner promised 5 but gave 1. Reviewer distinguishes justified withholding/retaliation from confusion about events. Universal harm is a direct profile counterexample. |

Automated action categories are transfer, harm, relevant communication (declared keyword heuristic), and withhold/defer. Justified deferrals and moral-role consistency require independent ratings. Human reviewers should first infer the moral type blind to assigned role, then judge factual coherence and the assigned-role compatibility (0 contradiction, 1 ambiguous/partial, 2 consistent). At least two raters and disagreement resolution are required for a scored subjective claim; no such ratings are fabricated here. ≥80% role consistency remains **pending** without them. This pilot does not compute the paper's run-level evaluator confusion matrix or equate its earlier 29/36 proxy with that metric.

Memory structure checks five major sections, required subfields and the original **flat** event records. Empty no-content strings are allowed; nesting is never mandatory. Factual checks use original observations and prior memory to detect unsupported completed events, participant/outcome contradictions, and explicit predeclared entity-scoped fact anchors. Missing anchors are omissions under that automated test; paraphrases and relationship interpretations are exposed for review. Semantic hallucination detection is not exhaustive. Eight C anchors cover 3-HP cooperation, the 4-HP broken promise and zero transfer, a 5-HP promise versus actual 1 HP, 5-damage attack, 6-damage retaliation, and the distinct 2-HP donor. C1 receives those facts in input memory; C2 receives only the actual history, with target memory empty.

Provisional diagnostics: ≥99% eventual actions; ≥90% first complete two-stage attempt; ≥90% factual memory anchors; ≥80% independently reviewed moral consistency; no detected private-memory leakage or silent truncation. Wilson intervals are descriptive; repeated facts and rounds within an agent/run are correlated and are not independent Bernoulli trials. Passing thresholds alone cannot establish suitability.
"""


def generate_report(state):
    manifest=read(OUT/"manifest.json",{})
    counts=progress_counts()
    all_decisions=decisions()
    grouped=defaultdict(list)
    for d in all_decisions:
        grouped[(d["_experiment"],d.get("mode","off"))].append(d)
    quantitative={f"{e}/{m}":metrics_for(rows) for (e,m),rows in sorted(grouped.items())}
    write(OUT/"metrics_live.json",{"at":now(),"counts":counts,"groups":quantitative})
    historical=read(OUT/"historical_audit.json",{})
    preflight=read(OUT/"preflight/summary.json",{})
    telemetry=jsonlines(OUT/"telemetry/monitor.jsonl")
    gpus=[s["gpu"] for s in telemetry if s.get("gpu")]
    errors=jsonlines(OUT/"telemetry/errors.jsonl")
    recommendation=read(OUT/"final_audit.json",{}).get("recommendation","PENDING — runnable evaluations remain")
    completed_logs=[]
    for log in sorted((OUT/"tests").glob("*.log")):
        raw=log.read_bytes()
        text=raw.decode("utf-16" if raw.startswith(b'\xff\xfe') else "utf-8-sig",errors="replace")
        matches=re.findall(r"(?:\d+ failed, )?\d+ passed[^\r\n]*",text)
        if matches:
            completed_logs.append(f"`{log.name}`: {matches[-1]}")
    lines=["# Qwen3.8-27B remediation and viability evaluation, V2","",f"**Status: {state.get('status','preparing')}**. Updated {now()}. Recommendation: **{recommendation}**.","",
           "## Executive summary","",
           "The corrected suite evaluates moral decisions, original multi-agent dynamics, factual memory preservation/acquisition, native thinking with explicit reflection, and thirty-round readiness. All inference uses one local Qwen3.8-27B Q4_K_M llama.cpp instance on the RTX 3090. Historical A/B/C and stopped v3 baseline data are separate and never pooled. No full 240-round run is launched.","",
           "| Evaluation | Complete | Blocked | Planned |","|---|---:|---:|---:|"]
    for exp,c in counts.items():
        lines.append(f"| {exp} | {c.get('complete',c.get('complete_rounds',0))} | {c.get('blocked',c.get('blocked_rounds',0))+c.get('dependent_blocked_rounds',0)} | {c.get('planned',c.get('planned_rounds',0))} |")
    lines += ["","Human moral ratings and semantic adjudication are pending. Action validation, strict memory schema compliance, factual anchors and reviewer candidates are separate measurements. The results below are actual completed artifacts; unresolved units are never represented as successful.","",
              "## Previous audit findings and resolution","",
              f"Independent audit Markdown was not present in the workspace. The task's findings were checked directly. Historical recount: {historical.get('decision_units','pending')} units; {historical.get('allocation_actions','pending')} allocations offering {historical.get('planned_allocated_hp','pending')} HP; {historical.get('context_exceptions','pending')} explicit context exceptions. Strict new structural audit accepts {historical.get('memory_structure_valid','pending')}/{historical.get('memories_audited','pending')}; it checks more subfields than the earlier narrow audit, so this count is not a contradiction of its six cited defects. Full paths/errors: `data/qwen38_remediation_v2_20261016/historical_audit.json`.","",
              "| Finding | Remediation and status |","|---|---|",
              "| Untyped/incomplete memory | Read-only `deep_memory.py` validates original flat schema; factual fidelity scored independently; no silent correction or memory retries. Regression tests cover missing sections, fields, contradictions and wrong participants. |",
              "| Context exceptions and unchecked truncation | 64K Q8 tested; exact chat-template/tokenizer preflight plus output and reflection reserves. Original evidence retained; bounded retry feedback; nested `server_context.truncated` read correctly; generation limits explicitly flagged. |",
              "| Zero allocations in report | Immediate original action execution snapshots measure donor HP spent and each recipient's actual HP gain, separately from offered amounts and full-round metabolism. Known/capped allocation regression checks. Historical evidence remains unchanged. |",
              "| False watchdog restarts | Active slot, server health, log token/prompt progress, hard request timeout and process ownership are checked. Healthy slow work is never killed for reaching 300 seconds. |",
              "| v2/v3 mismatch | Canonical unchanged v2 prompt files for every corrected condition; frozen hashes and effective configuration. Historical v3 D was already stopped by the user at tick 4; no process was running at inspection. |",
              "| Seed bug | `set_random_seed` now seeds the existing shared instance so prior imports receive the seed. UUID identifiers remain nondeterministic; exact paired snapshots freeze IDs. |",
              "| Validator RNG side effect | Context validation executes copied actions. It now restores Python/shared/NumPy RNG state, so audit/retry validation cannot consume the real environment's random draws. Two initial engineering decisions and partial work are preserved under `preparation/initial_rng_audit_execution/` and excluded; the complete corrected inventory is rerun after this fix. |",
              "| Deaths and family integrity | Genealogy is checked across living agents and the existing cemetery; documented deceased parents/children are retained as valid relations. Inheritance is checked in B/longitudinal runs. |",
              "| Native thinking disabled | OFF and ON/low separately executed, both with original two-stage reflection; native reasoning parsed separately from final JSON. |","",
              "## Code changes and regression evidence","",
              "Application changes seed the existing RNG, make copied-action validation preserve RNG state, bound local retry artifacts while retaining original observations and last-chance feedback, and record local raw completion/finish metadata. Deep memory is an additional observational validator. New evaluation modules wrap the original agent, validator and `step()`; they do not replace orchestration. The old pilot truncation reader is corrected. All code and test logs are included in the package.","",
              "Before changes: **101 passed**, two existing Pydantic warnings (`tests/regression_before.log`). Post-change regression execution: see `tests/regression_after.log` and `tests/final_regression.log`; counts are taken only from completed log output.","",
              "Completed test log summaries: "+"; ".join(completed_logs)+". Initial engineering errors (mock finish metadata and overlong fixture messages) were fixed before corrected inference; their original traces remain in the preparation/test logs.","",
              "## Reproducibility","",
              f"Git: `{manifest.get('base_git_revision','pending')}`; canonical `{manifest.get('canonical_config','pending')}`. Model SHA256: `{manifest.get('model',{}).get('sha256','pending')}`. Quantization Q4_K_M; pinned HF revision `{manifest.get('model',{}).get('revision','pending')}`. llama.cpp: `{excerpt(manifest.get('llama_cpp'),240)}`.","",
              f"Effective generation: `{json.dumps(manifest.get('generation',{}),sort_keys=True)}`. Sampling seeds: `{manifest.get('sampling_seeds',[])}`; C seed 7271; longitudinal environment seeds `{manifest.get('environment_seeds',[])}`. Reflection uses seed+1; validation retries use seed+1000×retry index. Native low has a declared 2048-token thinking budget inside the 8192-token generation cap. No deterministic inference claim is made.","",
              "One loopback endpoint `127.0.0.1:8080/v1`; one model/slot; full offload, flash attention, q8_0 K/V; fit and context shifting disabled. Raw effective settings, server arguments/build, Python package versions and exact prompt hashes are in `manifest.json` and `environment-pip-freeze.txt`. No paid key or external LLM inference is used.","",
              f"64K engineering preflight all passed: **{preflight.get('all_passed','pending')}**. Native OFF/low channel probes and 48K/60K prompt retrieval probes, including three position markers, are saved under `preflight/`. Server allocation log reports 65/65 GPU layers, 17402.38 MiB CUDA weights, 2176 MiB Q8 KV, 149.62 MiB recurrent state, and 420.29 MiB compute. This verifies the tested server allocation, not perfect semantic retrieval over arbitrary histories.","",
              "## Protocol and fixed scoring rubric","",scoring_rubric(),
              "A uses one focal identity, agent_1, immutable per-condition checkpoints and five paired sampling repetitions across all four original profiles. Only its current moral type/profile is intervened on; its identity, HP, age, inventory, relationships, other agents, resources, legal options and event history are held fixed. Existing offspring types stay fixed to avoid a target-type confound. This controlled counterfactual reassignment is explicitly distinct from the original simulator's fixed inherited moral types; B and longitudinal runs retain original inheritance. Infant is age 1 with parent 10; adult offspring is 10 with parent 18; the stranger/cooperator/exploiter comparisons use focal 18, target 10. All targets have 2 HP, focal 20, produced through original fixture action/environment transitions. Snapshot pair hashes are checked. Adaptive extension is frozen in `manifest.json` and `adaptive_A.json`; it never stops sampling because a desired effect appears.","",
              "B starts eight agents (two/type), max_life_steps=2, two communication steps per life step, and runs six actual `step()` rounds. Longitudinal uses max_life_steps=10 and thirty actual rounds for each of three independent seeds, with births/deaths retained. `world.max_life_steps` is asserted on effective Pydantic configuration and controls the original runner; legacy `max_time_steps` is not used or silently accepted by this evaluator.","",
              "C has C1 preservation versus C2 acquisition × short/long/near-budget histories × four focal roles = 24 decisions, with matched seed 7271. Each history contains twelve actual fixture rounds executed with the original action/environment handlers, distinct partner identities, true transfers and attacks, a broken promise, retaliation, a separate helpful donor, and distinct legal route-planning communications as distractors. C2 target memory is empty; no target answers are inserted into it. Long histories must exceed 16K measured prompt tokens. Each communication stays within the contextual validator's 1000-character limit. Dense route-code proposals are a synthetic stress distribution, not evidence that ordinary messages are this long. Private bystander canary is excluded from every focal prompt and checked in outputs.","",
              "Native comparison is twenty predeclared exact OFF baseline cases (sixteen moral decisions, four long C2 histories) paired with ON/low. Prompt/seed/state are matched; the first-stage prompt equality is independently audited. Reflection's second input appropriately differs with the first generated draft. Mode selection is defined in `mode_selection.json`; it uses factual/structural/action regressions and possible improvements, not speed alone. Longitudinal inference uses the selected mode consistently across seeds.","",
              "## Progress, recovery and durable monitoring","",
              f"Started: {state.get('started_at','pending')}; current phase `{state.get('phase','preparing')}`; unit `{state.get('current_unit')}`; resume count {state.get('resume_count',0)}. Request errors recorded: {len(errors)}. Supervisor PID/runner PID, heartbeat and timestamped monitor samples are in the evidence directory. A process is only claimed persistent after its PID and changing heartbeat are verified.","",
              "Each decision cache is committed separately. Each simulation round has before/after checkpoints, RNG sidecars, per-action transitions and a durable commit marker. A successfully completed decision is not repeated on resume. At most initial+five validation/recovery attempts per unit. A persistent failed decision prevents committing its scientific trajectory; future dependent rounds are explicitly blocked. Partial checkpoints remain evidence. Private-memory object isolation and original top-level memory application are checked separately from factual correctness. Historical directories have frozen SHA256 manifests and are rechecked at finalization.","",
              "## Experiment A: paired moral decisions","",
              "| Scenario / role / seed | Status | Final action | HP offered to target | Communication / harm | Evidence |","|---|---|---|---:|---|---|"]
    for row in controlled_results("A"):
        b=row.get("behavior",{}); a=b.get("action",{})
        lines.append(f"| {row['case_id']} | {row.get('status')} | {a.get('action_type','none')} | {b.get('target_hp_offered',0)} | relevant communication {b.get('relevant_communication_heuristic')}; harmful {b.get('harmful_action')} | `{rel(row['_path'])}` |")
    lines += ["","Primary contrasts are paired HP differences, with bootstrap intervals and exact descriptive sign-flip probabilities. Small samples, shared state, multiple comparisons and unresolved subjective reasoning limit inference.","",
              "| Contrast | Pairs | Paired HP differences | Mean HP difference | Paired bootstrap 95% interval |","|---|---:|---|---:|---|"]
    for c in paired_contrasts():
        lines.append(f"| {c['contrast']} | {c['n_pairs']} | {c['paired_hp_differences']} | {c['mean_hp_difference']} | {c['paired_bootstrap_95_ci']} |")
    lines += ["","## Experiment B and longitudinal state transitions","",
              "The following are committed original simulator rounds; failed/dependent rows are explicit. Delivered HP is summed from immediate recipient deltas, with true recipient IDs in `action_transitions.jsonl`. HP before/after a whole round also includes actions, births, deaths and metabolism and is not itself a transfer metric.","",
              "| Run / round | Status | Living population before→after | Actions | HP offered / received | Births / removed | Private-memory leaks |","|---|---|---|---|---|---|---|"]
    for exp in ("B","L20261017","L20261019","L20261023"):
        for path,row in round_results(exp):
            lines.append(f"| {exp}/{row.get('round')} | {row.get('status')} | {row.get('population_before','?')}→{row.get('population_after','?')} | {row.get('actions',{})} | {row.get('offered_hp','?')} / {row.get('actual_hp_received','?')} | {len(row.get('births',[]))} / {len(row.get('removed_or_dead',[]))} | {row.get('private_memory_leak_events','not evaluated')} |")
            if row.get("error"):
                lines.append(f"| | **Blocker** | {excerpt(row['error'],500)} | | | | |")
    lines += ["","Per-agent actions, communication content, HP changes, action reasons, prey/plant resources, family links, memory changes and checkpoint hashes are preserved for every committed round. An absent retaliation or reproduction event is reported as not observed, not inferred from an action label or counted as a capability success.","",
              "### B: individual executed actions and recipients","",
              "| Round / agent | Action | Delivered HP and recipients | Observed interaction / reason | Independent private memories |","|---|---|---|---|---|"]
    for path,row in round_results("B"):
        if row.get("status")!="complete":
            continue
        for t in jsonlines(path.parent/"action_transitions.jsonl"):
            recipients=", ".join(f"{x['recipient']}: {x['received_hp']}" for x in t["allocation"].get("recipients",[]))
            lines.append(f"| {t['step']} / {t['agent_id']} | {t['action'].get('action_type')} (executed {t.get('execution_confirmed')}) | {recipients or 'none'} | {excerpt(t['action'].get('message') or '; '.join(t.get('new_observations',[])) or t['action'].get('reason'),260)} | {not bool(t['other_private_memories_changed'])} |")
    lines += ["",
              "## Experiment C: memory preservation versus acquisition","",
              "Eight predeclared facts per decision are scoped to the correct participant. The table shows automated anchors with stage changes, not human semantic accuracy. Each fact's expected truth, matched patterns, updated-memory text, alternative participant matches and structural errors are saved in the response audit.","",
              "| Case | Input tokens (first request) | Draft facts | Reflected facts | Final structure | Explicit factual findings |","|---|---:|---:|---:|---|---:|"]
    for row in controlled_results("C"):
        if not row.get("decision_path"):
            lines.append(f"| {row['case_id']} | blocked dependency | not evaluated | not evaluated | not evaluated | {excerpt(row.get('error'),220)} |")
            continue
        d=read(ROOT/row["decision_path"],{}); d["_path"]=ROOT/row["decision_path"]
        calls=calls_for(d); first=next((c for c in calls if c.get("stage")=="first_stage"),{}); final=final_call(d)
        f=first.get("audit",{}).get("memory_fidelity",{}); a=final.get("audit",{}); g=a.get("memory_fidelity",{})
        n=len(row.get("initial",{}).get("facts",[]))
        lines.append(f"| {row['case_id']} | {first.get('input_tokens','unreturned')} | {f.get('fact_hits',0)}/{n} | {g.get('fact_hits',0)}/{n} | {a.get('memory_structure',{}).get('valid',False)} | {len(g.get('findings',[]))} |")
    lines += ["","The 24 synthetic histories originate from original action-handler execution. Four additional C1/C2 cases use the last two verified B round snapshots, selected by a frozen rule. Original model memories are preserved in fixture metadata. C1 explicitly seeds ground-truth memory for a preservation intervention; C2 clears long-term memory and the prior short-term plan, retaining actual observations/messages for acquisition. These are declared fixture interventions, not repairs to B memories or outputs. Observed B/longitudinal event ledgers are also checked against subsequent memories; exhaustive semantic reliability is not claimed. Plans and allegations are not completed events. Memory-size growth and structural-event drift remain in per-call audits.","",
              "## Native thinking comparison","",
              "| Matched case | OFF action / facts / schema | ON-low action / facts / schema | OFF→ON seconds | OFF→ON completion tokens |","|---|---|---|---:|---:|"]
    for p in reasoning_comparison():
        a,b=p["off"],p["low"]
        lines.append(f"| {p['case']} | {a['action'].get('action_type','none')} / {a['fact_hits']}/{a['fact_total']} / {a['structure_valid']} | {b['action'].get('action_type','none')} / {b['fact_hits']}/{b['fact_total']} / {b['structure_valid']} | {a['seconds']:.1f}→{b['seconds']:.1f} | {a['tokens']}→{b['tokens']} |")
    selection=read(OUT/"mode_selection.json",{})
    lines += ["",f"Selected longitudinal mode: **{selection.get('selected_mode','pending')}**. {selection.get('rule','Selection has not run.')} Human behavioral reasoning assessment remains pending.","",
              "## Quantitative measurements with separate denominators","",
              "The unit of action validity is a decision, not a JSON call. Draft validity, first complete two-stage cycle, eventual action success and final raw memory schema are separate. Failed/no-return units count against decision reliability. Simulation denominators are actual living-agent slots encountered; future population-dependent slots after a blocker are unknown and not invented.","",
              "| Experiment / native mode | Attempted units | Eventual actions | First full cycle | First draft | Final raw memory schema | Fact anchors (returned final) | Retries |","|---|---:|---|---|---|---|---|---:|"]
    for label,m in quantitative.items():
        lines.append(f"| {label} | {m['units']} | {wilson(m['eventual_actions'],m['units'])} | {wilson(m['first_pipeline_actions'],m['units'])} | {wilson(m['first_draft_actions'],m['units'])} | {wilson(m['final_structures'],m['units'])} | {wilson(m['fact_hits'],m['fact_total'])} | {m['retries']} |")
    lines += ["","| Experiment / native mode | Input / output tokens | Max input | Request elapsed (s) | Possible truncations | Output cap hits | Factual findings |","|---|---:|---:|---:|---:|---:|---:|"]
    for label,m in quantitative.items():
        lines.append(f"| {label} | {m['input_tokens']:,} / {m['output_tokens']:,} | {m['max_input_tokens']:,} | {m['observed_request_seconds']:.1f} | {m['possible_silent_truncations']} | {m['output_limits']} | {m['explicit_factual_findings']} |")
    lines += ["","| Experiment / mode | Reflection improved / regressed factual anchors | Improved / regressed memory structure | Changed action JSON | Private canary output leaks |","|---|---:|---:|---:|---:|"]
    for label,m in quantitative.items():
        lines.append(f"| {label} | {m['reflection_fact_improved_units']} / {m['reflection_fact_regressed_units']} | {m['reflection_structure_improved_units']} / {m['reflection_structure_regressed_units']} | {m['reflection_action_changed_units']} | {m['private_canary_leaks']} |")
    if gpus:
        lines += ["",f"Board VRAM: {min(g['used_mib'] for g in gpus):.0f}–{max(g['used_mib'] for g in gpus):.0f} MiB, mean {statistics.mean(g['used_mib'] for g in gpus):.1f} MiB over {len(gpus)} samples. Mean sampled GPU utilization {statistics.mean(g['utilization_percent'] for g in gpus):.1f}%; mean board power {statistics.mean(g['power_watts'] for g in gpus):.1f} W. Windows/other GPU processes are included in board VRAM."]
    calls=[c for d in all_decisions for c in calls_for(d)]
    delta=Counter()
    for call in calls:
        delta.update(call.get("metrics_delta",{}))
    gen_t=delta.get("llamacpp:tokens_predicted_total",0); gen_s=delta.get("llamacpp:tokens_predicted_seconds_total",0)
    prompt_t=delta.get("llamacpp:prompt_tokens_total",0); prompt_s=delta.get("llamacpp:prompt_seconds_total",0)
    lines += ["",f"Measured server-counter generation throughput: {gen_t/gen_s if gen_s else 0:.2f} tokens/s ({gen_t:.0f} generated tokens / {gen_s:.2f} decoding seconds). Prompt-counter throughput: {prompt_t/prompt_s if prompt_s else 0:.2f} tokens/s ({prompt_t:.0f} / {prompt_s:.2f}s). Completion-token/request-latency ratios are not mislabeled as pure decoding throughput. Completed raw calls: {len(calls)}; request exceptions: {len(errors)}. Elapsed runner wall time: {max(0,time.time()-state.get('started_epoch',time.time()))/3600:.2f} hours; inspection/preflight timing and final completion timestamp are recorded separately.","",
              "## Significant failures and representative first/reflected evidence","",
              "Findings below distinguish objective errors from review candidates. A missing regex anchor may be a paraphrase; an unsupported relationship change is a candidate for review. Full response JSON is never silently repaired.",""]
    examples=[]
    for d in all_decisions:
        calls=calls_for(d); final=final_call(d); audit=final.get("audit",{})
        fidelity=audit.get("memory_fidelity",{}); defects=audit.get("memory_structure",{}).get("errors",[])
        if not d.get("success") or defects or fidelity.get("fact_hits",0)<fidelity.get("fact_total",0) or fidelity.get("findings"):
            examples.append((d,final,defects,fidelity))
    grouped_examples=defaultdict(list)
    for example in examples:
        grouped_examples[example[0]["_experiment"]].append(example)
    selected_examples=[]
    for exp,items in sorted(grouped_examples.items()):
        items.sort(key=lambda x:(not x[0].get("success"),len(x[3].get("findings",[])),x[3].get("fact_total",0)-x[3].get("fact_hits",0),len(x[2])),reverse=True)
        selected_examples+=items[:3 if exp.startswith("L") else 5]
    for i,(d,final,defects,fidelity) in enumerate(selected_examples):
        content=read(d["_path"],{}).get("response") or {}
        draft=next((c for c in calls_for(d) if c.get("stage")=="first_stage"),{})
        try:
            draft_data=json.loads(draft.get("content","{}"))
        except ValueError:
            draft_data={}
        lines += [f"### Evidence example {i+1}: `{rel(d['_path'])}`","",
                  f"Status {d.get('status')}; action `{action_dict(content)}`; error {excerpt(d.get('error'),700) or 'none'}. Missing/invalid schema paths: {excerpt(defects,900)}.","",
                  f"Draft thinking: {excerpt(draft_data.get('thinking'),1100)}","",
                  f"Reflected thinking: {excerpt(content.get('thinking'),1100)}","",
                  f"Factual results: {fidelity.get('fact_hits',0)}/{fidelity.get('fact_total',0)} anchors. Explicit findings: {excerpt(fidelity.get('findings'),1000)}. Review candidates: {excerpt(fidelity.get('review_candidates'),1000)}.",""]
        misses=[f for f in fidelity.get("facts",[]) if not f.get("automated_anchor_present")]
        for fact in misses[:3]:
            lines += [f"- Expected: {fact.get('expected')}. Updated participant memory: {excerpt(fact.get('evidence'),1100)}. Attribution alternatives: {fact.get('other_entities_matching')}."]
        lines.append("")
    if not examples:
        lines.append("No completed failure examples yet; this is not a claim of success for pending tests.")
    lines += ["","## Human review and limitations","",
              "Blinded packets and an unblinding key are generated under `blinded_review/`. Assigned labels and exact type names are redacted from the review text, but behavior and residual self-descriptions may still reveal role: blinding is imperfect. No independent human grades are available. Original paper evaluations use a separate evaluator/confusion matrix; this local test uses no model self-judge. Small samples, one model quantization/GPU, synthetic controlled histories, observed population dependence, imperfect semantic automation and unreviewed moral judgments prevent a definitive reproduction claim.","",
              "## Evidence-based readiness recommendation","",f"**{recommendation}**. Completing a simulation or passing JSON thresholds does not establish cognitive validity. Critical memory defects, causal confounds, context blockers, or unresolved independent ratings must be considered before the 240-round study. Final decision rationale and an exhaustive expected-artifact audit are saved in `final_audit.json`.","",
              "## Raw evidence and reproduction commands","",
              "Primary evidence: `data/qwen38_remediation_v2_20261016/`. It contains frozen fixtures/configuration, exact requests, separate draft/reflection/native reasoning, original and applied memories, per-action HP deltas, round checkpoints, RNG sidecars, metrics, errors, server logs, heartbeats, tests and checksums. Historical evidence hashes are in `historical_sha256.json`.","",
              "```powershell","powershell -NoProfile -ExecutionPolicy Bypass -File scripts/supervise-qwen38-v2.ps1","# Re-running resumes committed decisions/rounds and does not repeat completed trials.",".\\.venv\\Scripts\\python.exe -m pytest scr/tests -q","```","",
              "Sources: [paper](https://aclanthology.org/2026.acl-long.1339/), [repository](https://github.com/MoralAgentSim/social-evol-sim), [pinned llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/b11669/tools/server/README.md). Local PDF/text and server documentation copies are under `sources/`. PDF extraction was performed using analysis-only pypdf in `.local/review-tools`; no simulation dependency was added."]
    text="\n".join(lines)+"\n"
    with FILE_LOCK:
        temp=REPORT.with_suffix(".v2.tmp")
        temp.write_text(text,encoding="utf-8")
        os.replace(temp,REPORT)


def blinded_packets():
    folder=OUT/"blinded_review"; folder.mkdir(exist_ok=True)
    review=[]; keys=[]
    def redact(text):
        for role in ("universal_group_focused_moral","reciprocal_group_focused_moral","kin_focused_moral","reproductive_selfish"):
            text=text.replace(role,"[assigned moral profile]")
        return re.sub(r"\b(universal(?:ly)?|reciprocal|kin[- ]focused|reproductiv(?:e|ely)[- ]selfish)\b","[profile redacted]",text,flags=re.I)
    for d in decisions():
        identifier=object_hash(rel(d["_path"]))[:16]
        response=d.get("response") or {}
        first=next((c for c in calls_for(d) if c.get("stage")=="first_stage"),{})
        final=final_call(d)
        initial=read(d["_path"].parent/"initial.json",{})
        before=read(d["_path"].parent/"before.json",{})
        if not before and "decisions" in d["_path"].parts:
            parts=d["_path"].relative_to(OUT).parts
            before=read(OUT/parts[0]/"rounds"/parts[2]/"before.json",{})
        agents=before.get("social_environment",{}).get("agents",[])
        own=next((dict(a) for a in agents if a["id"]==d.get("agent_id")),{})
        own.pop("type",None);own.pop("logs",None);own.pop("response_history",None);own.pop("memory",None)
        others=[{k:a.get(k) for k in ("id","type","state","family")} for a in agents if a["id"]!=d.get("agent_id")]
        review.append({"blind_id":identifier,"condition":initial.get("scenario") or initial.get("mechanism") or "simulation",
                       "focal_state":json.dumps(own),"other_agent_states":json.dumps(others),
                       "ground_truth_facts":json.dumps(initial.get("facts",[])),"observed_history":json.dumps(before.get("observations",[])),
                       "draft":redact(first.get("content","")),"reflected_response":redact(final.get("content","")),
                       "guess_moral_type":"","factual_coherence_0_2":"","assigned_role_compatibility_0_2":"","justified_deferral":"","rater":"","notes":""})
        keys.append({"blind_id":identifier,"path":rel(d["_path"]),"mode":d.get("mode"),"role":next((a for a in ("universal_group_focused_moral","reciprocal_group_focused_moral","kin_focused_moral","reproductive_selfish") if a in str(d["_path"])),"simulation_agent_review_pending")})
    for name,rows in (("review.csv",review),("unblinding_key.csv",keys)):
        if rows:
            with (folder/name).open("w",encoding="utf-8-sig",newline="") as handle:
                writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write(folder/"README.json",{"status":"pending_independent_human_ratings","raters_required":2,"note":"See the frozen rubric in the report. Do not have the evaluated Qwen model judge itself. Assigned-role scoring requires a separate unblinding stage."})


def finalize(state):
    log=OUT/"tests/final_regression.log"
    with log.open("w",encoding="utf-8") as handle:
        completed_tests=subprocess.run([str(ROOT/".venv/Scripts/python.exe"),"-m","pytest","scr/tests","-q"],cwd=ROOT,
            stdout=handle,stderr=subprocess.STDOUT,env={**__import__('os').environ,"LITELLM_LOCAL_MODEL_COST_MAP":"True","HF_HUB_OFFLINE":"1","PYTHONUTF8":"1"})
    if completed_tests.returncode:
        raise RuntimeError("Final regression tests failed; see "+str(log))
    manifest=read(OUT/"manifest.json",{})
    inventory=manifest.get("inventory",[])+read(OUT/"adaptive_A.json",{}).get("additional_cases",[])
    missing=[]; blockers=[]; completed=0
    for item in inventory:
        path=OUT/item["experiment"]/item["case_id"]/"result.json"
        result=read(path)
        if not result:
            missing.append(rel(path))
        elif result.get("status")=="complete":
            completed+=1
        elif result.get("status")=="blocked":
            blockers.append({"path":rel(path),"error":result.get("error")})
        else:
            missing.append(rel(path))
    for exp,total in [("B",6)]+[("L"+str(s),30) for s in (20261017,20261019,20261023)]:
        for number in range(1,total+1):
            path=OUT/exp/"rounds"/f"round_{number:02d}"/"result.json"
            result=read(path)
            if not result:
                missing.append(rel(path))
            elif result.get("status") in ("blocked","blocked_dependency"):
                blockers.append({"path":rel(path),"error":result.get("error")})
    changed=[]
    for path,digest in read(OUT/"historical_sha256.json",{}).get("files",{}).items():
        if not (ROOT/path).exists() or sha(ROOT/path)!=digest:
            changed.append(path)
    all_decisions=decisions()
    continuity=[]
    transfer_checks=[]
    for exp in ("B","L20261017","L20261019","L20261023"):
        previous=None
        for path,result in round_results(exp):
            if result.get("status")!="complete":
                continue
            before=read(path.parent/"before.json")
            after=read(path.parent/"after.json")
            if previous is not None and object_hash(before)!=object_hash(previous):
                continuity.append(rel(path))
            previous=after
            transitions=jsonlines(path.parent/"action_transitions.jsonl")
            received=sum(t.get("allocation",{}).get("received_hp",0) for t in transitions)
            if received!=result.get("actual_hp_received"):
                transfer_checks.append(rel(path))
    pairs=reasoning_comparison()
    pair_mismatches=[p["case"] for p in pairs if p["off"]["success"] and p["low"]["success"] and (not p["first_prompt_matched"] or not p["sampling_seed_matched"])]
    final_memories=[final_call(d).get("audit",{}) for d in all_decisions if d.get("success")]
    structural_bad=sum(not a.get("memory_structure",{}).get("valid",False) for a in final_memories)
    memory_hits=sum(a.get("memory_fidelity",{}).get("fact_hits",0) for a in final_memories)
    memory_n=sum(a.get("memory_fidelity",{}).get("fact_total",0) for a in final_memories)
    c_breakdown=defaultdict(lambda:{"facts":0,"hits":0,"units":0,"structure":0})
    for row in controlled_results("C"):
        if not row.get("decision_path"):
            continue
        d=read(ROOT/row["decision_path"],{});d["_path"]=ROOT/row["decision_path"]
        audit=final_call(d).get("audit",{});fidelity=audit.get("memory_fidelity",{})
        key=row["initial"].get("mechanism","unknown")+"/"+row["initial"].get("length","unknown")
        c_breakdown[key]["facts"]+=len(row["initial"].get("facts",[]));c_breakdown[key]["hits"]+=fidelity.get("fact_hits",0)
        c_breakdown[key]["units"]+=1;c_breakdown[key]["structure"]+=int(audit.get("memory_structure",{}).get("valid",False))
    facts_low=any(v["facts"] and v["hits"]/v["facts"]<.9 for v in c_breakdown.values())
    recommendation="NO-GO" if blockers or facts_low or structural_bad else "CONDITIONAL"
    audit={"at":now(),"planned_controlled_units":len(inventory),"completed_controlled_units":completed,"missing_artifacts":missing,
           "blockers":blockers,"historical_files_changed":changed,"successful_raw_memories_schema_invalid":structural_bad,
           "critical_anchors_hits":memory_hits,"critical_anchors_total":memory_n,"recommendation":recommendation,
           "C_separate_mechanism_length_results":dict(c_breakdown),"native_pair_prompt_or_seed_mismatches":pair_mismatches,
           "round_checkpoint_continuity_mismatches":continuity,"actual_transfer_summary_mismatches":transfer_checks,
           "rationale":"Do not launch the 240-round study while documented technical/memory failures or independent human moral review remain unresolved. A clean systems run alone would yield at most CONDITIONAL readiness pending ratings and broader replication."}
    write(OUT/"final_audit.json",audit)
    state["blocker_count"]=len(blockers)
    blinded_packets()
    write(OUT/"paired_contrasts.json",paired_contrasts())
    write(OUT/"reasoning_comparison.json",reasoning_comparison())
    if missing or changed or continuity or transfer_checks or pair_mismatches:
        raise RuntimeError(f"Final consistency audit failed: missing={len(missing)}, historical changes={len(changed)}, continuity={len(continuity)}, transfers={len(transfer_checks)}, native pairing={len(pair_mismatches)}")


def finalize_package(state):
    """Stage one folder, hash every file, create ZIP, read/test every entry."""
    package=ROOT/"QWEN38_REMEDIATION_V2_BUNDLE"
    archive=ROOT/"QWEN38_REMEDIATION_V2_BUNDLE.zip"
    if archive.exists():
        existing=read(OUT/"archive_verification.json",{})
        if existing.get("sha256")==sha(archive) and existing.get("verified"):
            return
        raise RuntimeError("Archive exists without matching verification; preserve it and diagnose before replacing")
    package.mkdir(exist_ok=True)
    sources=[REPORT,ROOT/"README.local.md",ROOT/".gitignore",ROOT/"README.md",ROOT/"pyproject.toml",ROOT/"requirements.txt",ROOT/"environment.yml",ROOT/"main.py"]
    sources+=list((ROOT/"config/configZ_major_v2").rglob("*"))
    sources+=list((ROOT/"scr").rglob("*.py"))
    sources+=[p for p in (ROOT/"scripts").glob("*") if p.is_file() and ("v2" in p.name or "capability" in p.name or p.name in ("setup-local.ps1","start-local-server.ps1","smoke_local.py"))]
    sources+=list(OUT.rglob("*"))
    for source in sources:
        if not source.is_file() or "__pycache__" in source.parts:
            continue
        dest=package/source.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dest)
    historical=ROOT/"QWEN38_VIABILITY_EXPERIMENT.zip"
    if historical.exists():
        dest=package/"historical"/historical.name;dest.parent.mkdir(exist_ok=True);shutil.copy2(historical,dest)
    (package/"PACKAGE_README.md").write_text("# Qwen viability remediation V2\n\nStart with reports/QWEN38_REMEDIATION_AND_VIABILITY_V2.md. Paths preserve the source repository layout. The package includes executed evidence, original simulator Python source, canonical configuration, scripts and regression logs. FILE_SHA256.json hashes every packaged file except itself. Downloaded GGUF, runtime binaries and virtual environment are dependencies and are excluded. The older pilot bundle is preserved under historical/. No full 240-round experiment was launched.\n",encoding="utf-8")
    checksums={str(p.relative_to(package)).replace("\\","/"):sha(p) for p in package.rglob("*") if p.is_file() and p.name!="FILE_SHA256.json"}
    write(package/"FILE_SHA256.json",checksums)
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as zipped:
        for path in sorted(package.rglob("*")):
            if path.is_file():
                zipped.write(path,str(path.relative_to(ROOT)).replace("\\","/"))
    with zipfile.ZipFile(archive) as zipped:
        bad=zipped.testzip()
        if bad:
            raise RuntimeError("Archive integrity failure: "+bad)
        for name,digest in checksums.items():
            import hashlib
            if hashlib.sha256(zipped.read(package.name+"/"+name)).hexdigest()!=digest:
                raise RuntimeError("Archived checksum mismatch: "+name)
        count=len(zipped.infolist())
    digest=sha(archive)
    (ROOT/(archive.name+".sha256")).write_text(digest+"  "+archive.name+"\n",encoding="ascii")
    write(OUT/"archive_verification.json",{"at":now(),"archive":str(archive),"sha256":digest,"file_entries":count,
                                       "size_bytes":archive.stat().st_size,"verified":True,"all_packaged_file_hashes_verified":True})
