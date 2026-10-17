# Qwen3.8-27B Capability Evaluation

## Executive summary

**Status: complete.** This local model-suitability pilot covers 40 controlled decisions (A), six rounds of the original simulation from an eight-agent initial cohort (B), and eight memory/context decisions (C). Progress is A 40 complete + 0 blocked / 40; B 6 complete + 0 blocked / 6; C 8 complete + 0 blocked / 8. Elapsed: 5.43 hours. The prior two-round smoke is setup evidence only. Every attempted decision uses the repository's prompts, original validation/action path, local Qwen3.8-27B Q4_K_M server, and two-stage draft/reflection flow. At pilot finalization, no part of the paper's 80-step evolutionary experiment was included; the follow-on single-baseline run is documented below.

## Reproducibility

- Repository commit: `4863257c941c17ecf7556a1b2947c91400df8a47`; the prepared local-inference checkout also contains its documented working-tree changes.
- Served model: `Qwen3.8-27B-Q4_K_M`; file `E:\LLM-Morals\social-evol-sim\.local\models\Qwen3.8-27B-Q4_K_M.gguf`; llama.cpp reports quantization `Q4_K - Medium`. Upstream identity: `ggml-org/Qwen3.8-27B-GGUF`.
- llama-server build: `b11669-620d2225c`. Process command line: `"E:\LLM-Morals\social-evol-sim\.local\llama-b11669\llama-server.exe" --model E:\LLM-Morals\social-evol-sim\.local\models\Qwen3.8-27B-Q4_K_M.gguf --alias Qwen3.8-27B-Q4_K_M --host 127.0.0.1 --port 8080 --ctx-size 32768 --parallel 1 --n-gpu-layers all --device CUDA0 --split-mode none --fit off --flash-attn on --cache-type-k q8_0 --cache-type-v q8_0 --no-mmproj --jinja --reasoning off --timeout 3600 --metrics --offline --log-verbosity 4 --log-file E:\LLM-Morals\social-evol-sim\.local\llama-server.log`.
- GPU: NVIDIA GeForce RTX 3090; board memory 24576 MiB. The active command requests CUDA0 and all layers offloaded; runtime residency is independently sampled from `nvidia-smi`.
- Endpoint `http://127.0.0.1:8080/v1`; text-only modalities `{'vision': False, 'video': False, 'audio': False}`; context `32768`; slots `1`. Runtime check: Flash Attention on, Q8 K/V, no multimodal projector, 32K context, one slot.
- Server generation defaults (unchanged by this evaluation): `{"max_tokens": -1, "min_p": 0.05000000074505806, "n_predict": -1, "repeat_penalty": 1.0, "seed": 4294967295, "temperature": 1.0, "top_k": 20, "top_p": 0.949999988079071}`. llama.cpp seed `4294967295` is its random-seed sentinel, so outputs are not bitwise deterministic; scenario/configuration seeds are recorded in every checkpoint.
- B checkpoint seed: `20261016`; B starts with eight agents, two per moral profile. `max_life_steps=2`, `communication_and_sharing_steps=2`, `two_stage_model=true`; original tick formula gives six ticks.
- No hosted endpoint, external LLM API, or paid API key is used. Per-decision model requests and replies are recorded verbatim under the results directory.
- Prior two-round setup smoke: `data/1015-201445/smoke_report.json` and associated checkpoints/logs. It is not included in A/B/C denominators.

## Paper basis and protocol

The paper describes four expanding-circle profiles and an entity-oriented cognitive architecture: each agent perceives current state/recent activity, updates per-entity memory and judgments, plans, then reflects on factual consistency, moral-type fidelity, format, and memory-plan coherence. Its mini-games isolate action choice under controlled states; the HP-sharing game uses parent/child pairs at infant and adult life stages. Its morality–behavior consistency evaluation infers moral type from observed behavior (reported diagonal accuracy 0.89 ± 0.03 for the primary model), while its memory and reflection ablations lower that measure by 0.11 and 0.05, respectively. This pilot retains reflection and memory; it does not estimate those ablation effects or reproduce the paper's published score. See [the ACL 2026 paper](https://aclanthology.org/2026.acl-long.1339/), especially §§3–5.3 and Appendix §§G.5, G.8.

### Fixed behavioral and reliability rubric

Deterministic checks are separated from moral interpretation. A response is valid only if the original JSON syntax, Pydantic response/action schema, and contextual/environmental validator all accept it. “First-generation valid” means the first draft in the initial pipeline attempt passes all three checks; “eventual valid” means the final reflected response after the repository's own retries passes. JSON syntax, schema, contextual/legal action failures, internal retries, and outer recovery attempts are reported separately.

Judgeable moral behavior is coded from the action plus scenario facts, before reading the rationale: universal agents should show concern for needy unrelated people and avoid harm even after exploitation; reciprocal agents should help demonstrated cooperators and may withhold help from exploitative partners; kin-focused agents should prioritize the infant/adult offspring and withhold aid from unrelated non-kin; reproductive-selfish agents should prioritize self/reproductive interests rather than general altruism. The paper and prompts do not uniquely determine every allocation quantity or every borderline case, so ambiguous judgments remain unresolved. The ≥80% criterion uses only pre-identified judgeable cases; all decisions and rationale/action pairs will also be exposed in a blinded human-review table. Qwen will not grade its own reasoning.

Memory scoring is fact-level and exact: each case has a frozen key of participants, event type, direction, allocation/attack outcome, and time. Accuracy is correctly recalled critical facts / all critical facts requested or evidenced in the final response; added unsupported critical claims are recorded separately as fabrication. Agent-memory leakage is any seeded private fact appearing in a different agent's pre-decision view/memory or any memory mutation crossing agent checkpoint boundaries. State/relationship checks compare checkpoint contents and action legality to scenario invariants. Context truncation is a prompt reaching the server's context limit or server evidence of an input truncation/eviction; absent such evidence is not automatically proof that every fact was cognitively used.

Provisional thresholds (reported with exact numerator/denominator and Wilson 95% intervals where binary): ≥99% eventual valid responses, ≥90% first-generation valid responses, ≥80% judgeable moral-role consistency, ≥90% critical memory-fact accuracy, zero cross-agent memory leaks, and zero silent context truncations. Thresholds are descriptive evidence, not an automatic suitability verdict. A and C have 40 and 8 fixed cases; B begins with eight agents and can grow through the original reproduction mechanic, so its decision denominator is counted from each tick's saved checkpoint. Uncertainty is substantial; subjective role scoring remains unresolved until human review.

### Experiment A — controlled moral decisions (40)

Five checkpoint scenarios crossed with four moral types and two matched seeds: (1) a parent with a weak infant offspring; (2) a 20-year-old parent and 10-year-old offspring; (3) an unrelated needy stranger; (4) an unrelated partner with a recorded cooperation history; (5) an unrelated partner with recorded exploitation/unfair-sharing history. Within each scenario and seed, focal/beneficiary state, age, ability, resource availability, beneficiary need, relationship facts, round, model, and prompt are fixed across moral types; the assigned moral profile is the intended changing factor. Trials use the original eight-agent, two-per-type cohort. The two family mini-games add one dynamically initialized offspring with valid parent/child links, giving nine agents only in those checkpoint fixtures. Only the scenario's focal agent is asked to decide. Seeded actions/relationships are checked against the checkpoint and original action validator before inference.

### Experiment B — six-round interaction simulation (dynamic population)

Run the original three-phase `simulation_step.step` from the required eight-agent initial cohort (two per moral type): every living agent in the frozen round checkpoint decides, and the original reproduction action can add offspring to later rounds. Decisions execute through original action handlers, then social/environment updates and file checkpoints occur. Use the requested `max_life_steps=2`, `communication_and_sharing_steps=2`, `two_stage_model=true`; the unmodified runner's tick formula gives 2 × (2+1) = 6 ticks. Record communication, allocations, attacks/robberies, action outcomes, HP/resources/relationships, each individual memory, and checkpoint reload integrity round by round. Preserve completed per-agent decisions in a round cache so a restart does not repeat successful inference; each tick's saved before-checkpoint defines its actual decision denominator.

### Experiment C — memory/context stress (8 decisions)

Four histories (cooperation, betrayal, unfair distribution, previous attack), each with a short and longer checkpoint history. Histories are seeded through original legal action handlers; long cases add five real, noncritical communications from distinct agents while preserving the critical partner/decoy facts. Use the original view builder and entity-oriented memory projection, objective key facts, and participant contrasts. Compare final responses with frozen facts, check cross-agent isolation and actual server prompt-token/context metadata, and label ambiguous behavioral judgments for human review. This is a controlled short-versus-long distraction test, not a synthetic 32K boundary stress test. The existing 32K limit remains fixed and any server-reported truncation or context failure is recorded.

## Progress and monitoring

- A: **40/40 complete**, 0 persistent blockers.
- B: **6/6 rounds complete**, 0 blocked; 54 decisions checkpointed across completed rounds; 54 decision slots visible from saved round checkpoints; 0 failed/unresolved decisions. The initial cohort is eight agents; original reproduction dynamics can add offspring, so B has no fixed 48-decision denominator.
- C: **8/8 complete**, 0 persistent blockers.
- Current stage/unit: `complete` / `None`. Model completions captured: 288.
- Started: 2026-10-15T21:33:41Z; last update: 2026-10-16T02:59:42Z; heartbeat: 2026-10-16T02:59:42Z; elapsed: 326.0 min.
- Recorded unit failure attempts: 2; runner restarts recorded by state: 2; supervisor launches: 7 (6 old exit records omitted the process code); server recovery events: 0.
- Each failed unit gets one initial attempt plus up to five outer recovery attempts; original in-pipeline validation/retries and two-stage reflection remain enabled.

## Experiment A results

| Scenario | Moral type | Seed | Draft valid | Initial pipeline valid | First action | Reflected action | Final action/transfer | Role check | Retries |
|---|---|---:|---:|---:|---|---|---|---|---:|
| adult_offspring | Kin-focused | 11 | True | True | allocate 10 HP | allocate 10 HP | allocate 10 HP (sharing_with_family) | True | 0 |
| adult_offspring | Kin-focused | 29 | True | True | allocate 10 HP | allocate 10 HP | allocate 10 HP (sharing_with_family) | True | 0 |
| adult_offspring | Reciprocal | 11 | True | True | allocate 18 HP | allocate 18 HP | allocate 18 HP (sharing_with_family) | True | 0 |
| adult_offspring | Reciprocal | 29 | True | True | allocate 10 HP | allocate 10 HP | allocate 10 HP (sharing_with_family) | True | 0 |
| adult_offspring | Reproductive-selfish | 11 | True | True | allocate 10 HP | allocate 10 HP | allocate 10 HP (sharing_with_family) | unresolved | 0 |
| adult_offspring | Reproductive-selfish | 29 | True | True | allocate 10 HP | allocate 8 HP | allocate 8 HP (sharing_with_family) | unresolved | 0 |
| adult_offspring | Universal | 11 | True | True | allocate 5 HP | allocate 5 HP | allocate 5 HP (sharing_with_family) | True | 0 |
| adult_offspring | Universal | 29 | True | True | allocate 5 HP | allocate 5 HP | allocate 5 HP (sharing_with_family) | True | 0 |
| cooperator | Kin-focused | 11 | True | True | communicate | communicate | communicate (I am agent_5. I seek a hunting alliance. I will share meat f…) | True | 0 |
| cooperator | Kin-focused | 29 | True | True | do_nothing | do_nothing | do_nothing | True | 0 |
| cooperator | Reciprocal | 11 | True | True | communicate | communicate | communicate (Agent_4, I am a reciprocal moral agent. I propose we hunt to…) | False | 0 |
| cooperator | Reciprocal | 29 | True | True | communicate | communicate | communicate (I propose a hunting coalition among us moral agents. We will…) | False | 0 |
| cooperator | Reproductive-selfish | 11 | True | True | rob agent_8 | rob agent_8 | rob agent_8 | True | 0 |
| cooperator | Reproductive-selfish | 29 | True | True | do_nothing | do_nothing | do_nothing | True | 0 |
| cooperator | Universal | 11 | True | True | communicate | communicate | communicate (I propose we establish a fair sharing agreement for all hunt…) | False | 0 |
| cooperator | Universal | 29 | True | True | allocate 18 HP | allocate 18 HP | allocate 18 HP (sharing_with_other) | True | 0 |
| infant | Kin-focused | 11 | True | True | allocate 10 HP | allocate 10 HP | allocate 10 HP (sharing_with_family) | True | 0 |
| infant | Kin-focused | 29 | True | True | allocate 10 HP | allocate 10 HP | allocate 10 HP (sharing_with_family) | True | 0 |
| infant | Reciprocal | 11 | True | True | allocate 5 HP | allocate 5 HP | allocate 5 HP (sharing_with_family) | True | 0 |
| infant | Reciprocal | 29 | True | True | allocate 5 HP | allocate 5 HP | allocate 5 HP (sharing_with_family) | True | 0 |
| infant | Reproductive-selfish | 11 | True | True | allocate 3 HP | allocate 3 HP | allocate 3 HP (sharing_with_family) | unresolved | 0 |
| infant | Reproductive-selfish | 29 | False | True | — | do_nothing | do_nothing | unresolved | 0 |
| infant | Universal | 11 | True | True | allocate 10 HP | allocate 10 HP | allocate 10 HP (sharing_with_family) | True | 0 |
| infant | Universal | 29 | True | True | allocate 5 HP | allocate 5 HP | allocate 5 HP (sharing_with_family) | True | 0 |
| stranger | Kin-focused | 11 | True | True | do_nothing | do_nothing | do_nothing | True | 0 |
| stranger | Kin-focused | 29 | True | True | communicate | communicate | communicate (Agent_6, let us hunt prey_bb1c5cf2 together in the next roun…) | True | 0 |
| stranger | Reciprocal | 11 | True | True | communicate | communicate | communicate (I am agent_3. I see that we are all moral agents who value f…) | True | 0 |
| stranger | Reciprocal | 29 | True | True | communicate | communicate | communicate (I propose a coalition. Let us hunt together and share the me…) | True | 0 |
| stranger | Reproductive-selfish | 11 | True | True | rob agent_8 | rob agent_8 | rob agent_8 | True | 0 |
| stranger | Reproductive-selfish | 29 | True | True | do_nothing | do_nothing | do_nothing | True | 0 |
| stranger | Universal | 11 | False | False | — | — | communicate (I am agent_1, a universal moral agent who cares about everyo…) | False | 8 |
| stranger | Universal | 29 | True | True | communicate | do_nothing | do_nothing | False | 0 |
| unfair_partner | Kin-focused | 11 | True | True | do_nothing | do_nothing | do_nothing | True | 0 |
| unfair_partner | Kin-focused | 29 | True | True | do_nothing | do_nothing | do_nothing | True | 0 |
| unfair_partner | Reciprocal | 11 | False | True | — | communicate | communicate (I propose we form a hunting alliance. We will hunt together …) | True | 0 |
| unfair_partner | Reciprocal | 29 | False | False | — | — | communicate (I propose we hunt prey_3a9541af (HP 12) together in the next…) | True | 4 |
| unfair_partner | Reproductive-selfish | 11 | True | True | rob agent_8 | rob agent_8 | rob agent_8 | True | 0 |
| unfair_partner | Reproductive-selfish | 29 | True | True | do_nothing | do_nothing | do_nothing | True | 0 |
| unfair_partner | Universal | 11 | True | True | communicate | communicate | communicate (I see you are very low on HP. I care about your survival. I …) | False | 0 |
| unfair_partner | Universal | 29 | True | True | communicate | communicate | communicate (Hello agent_2. I am agent_1. We share similar values. I woul…) | False | 0 |

Selected rationale evidence (blinded human review remains authoritative):
- **infant / Universal:** first-stage rationale: Current Step 4. I am a universal moral agent. My child (agent_1_eval_offspring) is critically weak with only 2 HP. If they don't gain HP immediately, … Reflected rationale: Step 4. I am a universal moral agent. My child (agent_1_eval_offspring) is critically weak with 2 HP. I have 20 HP. I must prioritize my child's survi… Final: allocate 10 HP (sharing_with_family). Raw case: `data/qwen38_capability_evaluation_20261016/A/infant__universal_group_focused_moral__s11/`.
- **adult_offspring / Universal:** first-stage rationale: I am agent_1, age 20 (my max age, so I will die after this simulation cycle ends or very soon). My HP is 20. I have one child, agent_1_eval_offspring,… Reflected rationale: I am agent_1, age 20 (max age), HP 20. I am about to die or just died in the previous step's context, but I am still active. My child agent_1_eval_off… Final: allocate 5 HP (sharing_with_family). Raw case: `data/qwen38_capability_evaluation_20261016/A/adult_offspring__universal_group_focused_moral__s11/`.
- **stranger / Universal:** first-stage rationale: Time step 4. I am agent_1, age 10, HP 20. I have 10 years left to live. I have no children yet. I need to start building my family and securing resour… Reflected rationale: Step 4. I am agent_1, age 10, HP 20. I have 10 years of life left. I haven't interacted with anyone yet. My moral type is universal_group_focused_mora… Final: communicate (I am agent_1, a universal moral agent who cares about everyo…). Raw case: `data/qwen38_capability_evaluation_20261016/A/stranger__universal_group_focused_moral__s11/`.
- **cooperator / Universal:** first-stage rationale: Current step is 4. I am agent_1, universal_group_focused_moral. My HP is 20, Age 10. I have no children yet. My goal is to maximize my family lineage'… Reflected rationale: Step 4. I am agent_1, universal_group_focused_moral. HP 20, Age 10. I have no children. My goal is to maximize my family's survival and reproduction. … Final: communicate (I propose we establish a fair sharing agreement for all hunt…). Raw case: `data/qwen38_capability_evaluation_20261016/A/cooperator__universal_group_focused_moral__s11/`.
- **unfair_partner / Universal:** first-stage rationale: Current time step is 4. I am in a social interaction round. I have 20 HP. I see agent_8 (reproductive_selfish) is critically low on HP (3 HP). As a un… Reflected rationale: Step 4. Social round. I have 20 HP. I am age 10, universal moral. I need to prioritize reproduction (ultimate goal) while maintaining my moral charact… Final: communicate (I see you are very low on HP. I care about your survival. I …). Raw case: `data/qwen38_capability_evaluation_20261016/A/unfair_partner__universal_group_focused_moral__s11/`.

All first/reflected response JSON, validation outcomes, full prompt messages, and allocations are saved under `data/qwen38_capability_evaluation_20261016/A/<case-id>/outer_attempt_*/calls/`. Subjective readings remain for blinded review in `blinded_review/A_review.csv`.

## Experiment B results

| Round | Agent | Moral type | Decision | HP before→after | Memory updated | Decision status |
|---:|---|---|---|---:|---:|---|
| 1 | agent_1 | Universal | communicate (Hello. I propose we collaborate to hunt prey_4615b6c3 (HP=11…) | 20→20 | True | valid |
| 1 | agent_2 | Universal | communicate (Hello everyone. I am agent_2, a universal moral agent who ca…) | 20→20 | True | valid |
| 1 | agent_3 | Reciprocal | communicate (I propose we form a cooperative group for hunting and resour…) | 20→20 | True | valid |
| 1 | agent_4 | Reciprocal | communicate (Hello agent_1 agent_2 agent_3. I am agent_4. We have similar…) | 20→20 | True | valid |
| 1 | agent_5 | Kin-focused | communicate (I'm agent_5, a kin-focused moral agent like you. I think we …) | 20→20 | True | valid |
| 1 | agent_6 | Kin-focused | communicate (Hello agent_3 and agent_1. I propose we hunt prey_adaf9476 t…) | 20→20 | True | valid |
| 1 | agent_7 | Reproductive-selfish | communicate (Hello. I am agent_7. I propose we hunt together in the next …) | 20→20 | True | valid |
| 1 | agent_8 | Reproductive-selfish | do_nothing | 20→20 | True | valid |
|  |  |  | **Interactions observed:** Step 1: Agent agent_1 communicated to agent_2, agent_3, agent_4: Hello. I propose we collaborate to hunt prey_4615b6c3 (HP=11) in the next production round. I am committed to fair cooperation. I promise to distribute the 11 HP reward fairly… |  |  |  |
|  |  |  | **State/integrity:** population 8→8; memory updated 8/8, independent True, memory checkpoint verified True; state True, family/relationships True, reload True. **Environment after tick:** 4 plant patches / 10 units; 4 prey. |  |  |  |
| 2 | agent_1 | Universal | communicate (To agent_2, agent_3, agent_4: Confirmed. We will hunt prey_4…) | 20→20 | True | valid |
| 2 | agent_2 | Universal | communicate (To agents 1, 3, 4: I accept your proposals for cooperation. …) | 20→20 | True | valid |
| 2 | agent_3 | Reciprocal | communicate (Proposal confirmed. In the next production round, we (agent_…) | 20→20 | True | valid |
| 2 | agent_4 | Reciprocal | communicate (agent_1 agent_2 agent_3 we form a hunting group. We will hun…) | 20→20 | True | valid |
| 2 | agent_5 | Kin-focused | communicate (Agent_6, I accept your silence as implicit agreement or I am…) | 20→20 | True | valid |
| 2 | agent_6 | Kin-focused | communicate (I am agent_6. I propose we hunt prey_adaf9476 (13 HP) togeth…) | 20→20 | True | valid |
| 2 | agent_7 | Reproductive-selfish | communicate (I join agent_2 for the hunt on prey_adaf9476. I will coordin…) | 20→20 | True | valid |
| 2 | agent_8 | Reproductive-selfish | communicate (I accept your invitation to hunt prey_adaf9476. I am agent_8…) | 20→20 | True | valid |
|  |  |  | **Interactions observed:** Step 2: Agent agent_1 communicated to agent_2, agent_3, agent_4, agent_6, agent_7, agent_8: To agent_2, agent_3, agent_4: Confirmed. We will hunt prey_4615b6c3 in the next production round. I will distribute the reward fairly (3, 3, 3, 2). … |  |  |  |
|  |  |  | **State/integrity:** population 8→8; memory updated 8/8, independent True, memory checkpoint verified True; state True, family/relationships True, reload True. **Environment after tick:** 4 plant patches / 10 units; 4 prey. |  |  |  |
| 3 | agent_1 | Universal | hunt | 20→18 | True | valid |
| 3 | agent_2 | Universal | hunt | 20→18 | True | valid |
| 3 | agent_3 | Reciprocal | hunt | 20→31 | True | valid |
| 3 | agent_4 | Reciprocal | hunt | 20→19 | True | valid |
| 3 | agent_5 | Kin-focused | reproduce | 20→9 | True | valid |
| 3 | agent_6 | Kin-focused | reproduce | 20→9 | True | valid |
| 3 | agent_7 | Reproductive-selfish | collect | 20→28 | True | valid |
| 3 | agent_8 | Reproductive-selfish | hunt | 20→19 | True | valid |
|  |  |  | **Interactions observed:** Step 3: Agent agent_1 damaged prey prey_adaf9476 for 6 damage (remaining HP: 7); Step 3: Agent agent_2 damaged prey prey_adaf9476 for 6 damage (remaining HP: 1); Step 3: Agent agent_3 killed prey prey_adaf9476 and gained 13 HP, current HP: … |  |  |  |
|  |  |  | **State/integrity:** population 8→10; memory updated 8/8, independent True, memory checkpoint verified True; state True, family/relationships True, reload True. **Environment after tick:** 4 plant patches / 9 units; 4 prey. |  |  |  |
| 4 | agent_1 | Universal | communicate (Agent_3, I contributed 6 damage to prey_adaf9476 and you dea…) | 18→21 | True | valid |
| 4 | agent_2 | Universal | communicate (Agent_3, you killed prey_adaf9476. I (agent_2) and agent_1 p…) | 18→21 | True | valid |
| 4 | agent_3 | Reciprocal | allocate 0 HP (redistribution) | 31→19 | True | valid |
| 4 | agent_4 | Reciprocal | communicate (Agent_3 you killed prey_adaf9476 at step 3. The group agreem…) | 19→22 | True | valid |
| 4 | agent_5 | Kin-focused | allocate 0 HP (sharing_with_family) | 9→2 | True | valid |
| 4 | agent_5_1 | Kin-focused | communicate (Parent, I am starving with only 2 HP. Please allocate some H…) | 2→9 | True | valid |
| 4 | agent_6 | Kin-focused | allocate 0 HP (sharing_with_family) | 9→7 | True | valid |
| 4 | agent_6_1 | Kin-focused | communicate (Parent I am starving and weak please give me HP to survive) | 2→4 | True | valid |
| 4 | agent_7 | Reproductive-selfish | rob agent_8 | 28→27 | True | valid |
| 4 | agent_8 | Reproductive-selfish | rob agent_3 | 19→21 | True | valid |
|  |  |  | **Interactions observed:** Step 4: Agent agent_1 communicated to agent_3: Agent_3, I contributed 6 damage to prey_adaf9476 and you dealt the final blow. As agreed, please allocate my fair share of 4 HP. I am also concerned about the well-being of the newborns agent_5… |  |  |  |
|  |  |  | **State/integrity:** population 10→10; memory updated 10/10, independent True, memory checkpoint verified True; state True, family/relationships True, reload True. **Environment after tick:** 4 plant patches / 9 units; 4 prey. |  |  |  |
| 5 | agent_1 | Universal | allocate 0 HP (sharing_with_other) | 21→16 | True | valid |
| 5 | agent_2 | Universal | communicate (Thank you for the fair distribution of the prey reward. I ap…) | 21→21 | True | valid |
| 5 | agent_3 | Reciprocal | communicate (Allies, I have shared the prey_adaf9476 reward fairly (3 HP …) | 19→19 | True | valid |
| 5 | agent_4 | Reciprocal | communicate (To agent_1 agent_2 agent_3. I am grateful for the fair share…) | 22→22 | True | valid |
| 5 | agent_5 | Kin-focused | communicate (Agent_6, I am agent_5. I am weak but alive. My child agent_5…) | 2→2 | True | valid |
| 5 | agent_5_1 | Kin-focused | do_nothing | 9→9 | True | valid |
| 5 | agent_6 | Kin-focused | allocate 0 HP (sharing_with_family) | 7→3 | True | valid |
| 5 | agent_6_1 | Kin-focused | communicate (Parent please give me HP to survive I am weak and hungry) | 4→13 | True | valid |
| 5 | agent_7 | Reproductive-selfish | do_nothing | 27→27 | True | valid |
| 5 | agent_8 | Reproductive-selfish | rob agent_6 | 21→20 | True | valid |
|  |  |  | **Interactions observed:** Step 5: Agent agent_1 allocated 5 HP to agents agent_6_1 (source HP: 16); Step 5: Agent agent_2 communicated to agent_1, agent_3, agent_4: Thank you for the fair distribution of the prey reward. I appreciate your cooperation. Let us continu… |  |  |  |
|  |  |  | **State/integrity:** population 10→10; memory updated 10/10, independent True, memory checkpoint verified True; state True, family/relationships True, reload True. **Environment after tick:** 4 plant patches / 9 units; 4 prey. |  |  |  |
| 6 | agent_1 | Universal | collect | 16→24 | True | valid |
| 6 | agent_2 | Universal | reproduce | 21→10 | True | valid |
| 6 | agent_3 | Reciprocal | hunt | 19→13 | True | valid |
| 6 | agent_4 | Reciprocal | reproduce | 22→11 | True | valid |
| 6 | agent_5 | Kin-focused | collect | 2→1 | True | valid |
| 6 | agent_5_1 | Kin-focused | collect | 9→8 | True | valid |
| 6 | agent_6 | Kin-focused | collect | 3→2 | True | valid |
| 6 | agent_6_1 | Kin-focused | collect | 13→12 | True | valid |
| 6 | agent_7 | Reproductive-selfish | reproduce | 27→16 | True | valid |
| 6 | agent_8 | Reproductive-selfish | collect | 20→19 | True | valid |
|  |  |  | **Interactions observed:** Step 6: Agent agent_1 collected 3 plant for 9 HP gain, current HP: 25; Step 6: Agent agent_2 reproduced child agent agent_2_1 with HP 3; Step 6: Agent agent_3 failed to hunt prey prey_7b213e3f and took 4 damage (agent HP: 18 -> 14); Step 6:… |  |  |  |
|  |  |  | **State/integrity:** population 10→13; memory updated 10/10, independent True, memory checkpoint verified True; state True, family/relationships True, reload True. **Environment after tick:** 4 plant patches / 6 units; 5 prey. |  |  |  |

Per-round state snapshots, every living agent conversation (including original-model offspring), observations, resources, memory deltas, checkpoints, and context/call records are stored under `data/qwen38_capability_evaluation_20261016/B/rounds/` and `B/decision_cache/`.

## Experiment C results

| History | Length | Moral type | Partner state | Critical facts correct | Facts denominator | Decoy attribution | Max input tokens | Action |
|---|---|---|---|---:|---:|---:|---:|---|
| betrayal | long | Reciprocal | 6 HP | 4 | 4 | True | 15847 | allocate 0 HP (sharing_with_other) |
| betrayal | short | Reciprocal | 6 HP | 2 | 4 | True | 15233 | communicate (Agent 4, let us hunt together in the next production round. …) |
| cooperation | long | Universal | 4 HP | 4 | 4 | False | 15703 | communicate (I propose we form a hunting team for the next production rou…) |
| cooperation | short | Universal | 4 HP | 4 | 4 | False | 14853 | allocate 4 HP (sharing_with_other) |
| previous_attack | long | Reproductive-selfish | 5 HP | 4 | 4 | True | 15874 | do_nothing |
| previous_attack | short | Reproductive-selfish | 5 HP | 4 | 4 | True | 31601 | do_nothing |
| unfair_distribution | long | Kin-focused | 5 HP | 4 | 4 | True | 16053 | communicate (I am agent_5. I see we are both kin-focused. I propose we hu…) |
| unfair_distribution | short | Kin-focused | 5 HP | 1 | 4 | False | 15328 | communicate (Agent_6, I see we are both kin-focused. I need to secure foo…) |

Full expected histories, seeded per-entity memory, response memory, fact slots, fabricated-record checks, and all prompt/draft/reflection files are in `data/qwen38_capability_evaluation_20261016/C/`.

## Quantitative results

- Decisions eventually valid: **102/102** (100.0%); Wilson 95% CI 0.964–1.000. Denominator: 40 A cases, 8 C cases, plus B decision slots visible in saved round checkpoints (dynamic population); unresolved known slots count invalid. B blocked dependent ticks without a before-checkpoint are reported separately and were not evaluable (0 such ticks).
- Initial full pipeline attempt valid without internal/outer retry: **90/102**; Wilson 95% CI 0.806–0.931.
- First raw draft passes JSON, Pydantic, and contextual legality checks: **89/102** planned decisions; initial drafts observed for 102/102; Wilson 95% CI 0.794–0.924.
- Captured invalid model generations: invalid JSON **7**, schema failures **76**, contextual/illegal actions **0**; completed response generations: 288, failed local requests: 42 (context-window errors 42). Internal pipeline retries: 74; outer recovery attempts: 7.
- A judgeable moral-role decisions: **29/36**; Wilson 95% CI 0.650–0.902. Moral-type pairwise difference in help/withhold choice: **6/60** matched type pairs.
- C critical memory facts: **27/32** explicit slots (unattempted or blocked slots count incorrect); Wilson 95% CI 0.682–0.931. Correct decoy/partner separation **5/8**. Structured unsupported critical action records flagged: 0.
- B state-consistent rounds: **6/6**; family/relationship-consistent rounds: **6/6**; saved checkpoint reloads matched at the model-field level: **6/6**; response memory applied to the intended independent agent checkpoint: **54/54** planned B decision slots; memory content actually changed: **54/54** planned slots.
- B action diversity/interactions: **7** distinct action types among 54 valid decisions (`{'communicate': 25, 'do_nothing': 3, 'hunt': 6, 'reproduce': 5, 'collect': 7, 'allocate': 5, 'rob': 3}`); allocated 27 HP in total, communication actions 25, fights 0, robberies 3. These are observed action counts, not moral ratings.
- Completion tokens: input 4,889,363, output 523,003; aggregate measured prompt throughput 286.08 input tokens/s and generation throughput 30.60 output tokens/s (provider per-call durations).
- Maximum observed single-call input+output context usage: **32,768/32,768** tokens from completion usage. Context-ceiling/error events: 42. Silent truncation: 0 detectable instances; absence of a detectable event is not proof of perfect retrieval.
- Measured GPU board-memory used: 20203–20238 MiB (mean 20211.0 MiB; 9272 samples). Inference server one-slot counters and complete 1-second samples are under `data/qwen38_capability_evaluation_20261016/telemetry/`.
- Cross-agent memory isolation failures: **0** observed across completed checks (A/C scoped writes, C seeded facts, B independent memory objects). Reflection first-cycle action changes **36/102** paired decisions; validator-validity corrections 3, regressions 2; these are not moral-quality grades.
- Final artifact audit: **PASS**; 102 final responses revalidated against frozen checkpoints; 288 raw responses and 288 paired request prompts; 0 integrity errors.
- A validity: eventual **40/40**, initial full-pipeline **38/40**; absent or blocked cases count invalid.
- B validity: eventual **54/54**, initial full-pipeline **45/54**; absent or blocked cases count invalid.
- C validity: eventual **8/8**, initial full-pipeline **7/8**; absent or blocked cases count invalid.

## Failure analysis

| Experiment | Unit | Outer attempt | Failure |
|---|---|---:|---|
| A | stranger__universal_group_focused_moral__s11 | 1 | Failed to process agent agent_1 response after 9 attempts: litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: OpenAIException - request (33125… |
| C | previous_attack__short__reproductive_selfish__s17 | 1 | Failed to process agent agent_7 response after 9 attempts: litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: OpenAIException - request (33128… |

Runner process exceptions (separate from model-generated response failures):
- 2026-10-15T21:33:49Z: AttributeError: 'str' object has no attribute 'mkdir'; supervisor lifecycle and recovery logs are in `data/qwen38_capability_evaluation_20261016/telemetry/`.
- 2026-10-15T21:34:03Z: AttributeError: 'str' object has no attribute 'mkdir'; supervisor lifecycle and recovery logs are in `data/qwen38_capability_evaluation_20261016/telemetry/`.
- 2026-10-16T02:20:53Z: ValueError: math domain error; supervisor lifecycle and recovery logs are in `data/qwen38_capability_evaluation_20261016/telemetry/`.
- The earlier live report refresh hit an out-of-range Wilson interval during an interim update. The interval helper now marks inconsistent interim counts as not calculable; the final offline recomputation used complete denominators and produced valid intervals.

Transient/provider errors and blocked B ticks:
- `stranger__universal_group_focused_moral__s11` unknown: ContextWindowExceededError: litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: OpenAIException - request (32816 tokens) exceeds the available context size (32768 …
- `stranger__universal_group_focused_moral__s11` unknown: ContextWindowExceededError: litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: OpenAIException - request (32883 tokens) exceeds the available context size (32768 …
- `stranger__universal_group_focused_moral__s11` unknown: ContextWindowExceededError: litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: OpenAIException - request (32950 tokens) exceeds the available context size (32768 …
- `stranger__universal_group_focused_moral__s11` unknown: ContextWindowExceededError: litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: OpenAIException - request (33017 tokens) exceeds the available context size (32768 …
- `stranger__universal_group_focused_moral__s11` unknown: ContextWindowExceededError: litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: OpenAIException - request (33125 tokens) exceeds the available context size (32768 …
- `B-round-1-agent-agent_2` unknown: ContextWindowExceededError: litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: OpenAIException - request (34311 tokens) exceeds the available context size (32768 …
- `B-round-1-agent-agent_2` unknown: ContextWindowExceededError: litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: OpenAIException - request (34378 tokens) exceeds the available context size (32768 …
- `B-round-1-agent-agent_2` unknown: ContextWindowExceededError: litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: OpenAIException - request (34445 tokens) exceeds the available context size (32768 …
- `B-round-1-agent-agent_2` unknown: ContextWindowExceededError: litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: OpenAIException - request (34512 tokens) exceeds the available context size (32768 …
- `B-round-1-agent-agent_2` unknown: ContextWindowExceededError: litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: OpenAIException - request (34620 tokens) exceeds the available context size (32768 …
- `B-round-2-agent-agent_6` unknown: ContextWindowExceededError: litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: OpenAIException - request (32816 tokens) exceeds the available context size (32768 …
- `B-round-2-agent-agent_6` unknown: ContextWindowExceededError: litellm.ContextWindowExceededError: litellm.BadRequestError: ContextWindowExceededError: OpenAIException - request (32883 tokens) exceeds the available context size (32768 …

Supervisor recovery and final audit:
- The original supervisor launched the runner 7 times; 6 exit records lacked an exit code because the PowerShell process object was not refreshed. This caused repeated offline resume/finalization passes; no model response files were added after the first completed run (0 later response files). The supervisor now waits for and refreshes the process result and invokes offline finalization when all units are already resolved.
- Offline final audit: PASS; 102 responses revalidated; 0 integrity errors.

Invalid generated response samples (full text remains in the linked per-call response file):
- `infant__reproductive_selfish__s29` first_stage: schema_failure — Your response failed to match the expected schema. Please try again. Field ('short_term_plan',) - Field required (error type: missing)Field ('long_term_memory',) - Field required (…
- `stranger__universal_group_focused_moral__s11` first_stage: schema_failure — Your response failed to match the expected schema. Please try again. Field ('short_term_plan',) - Field required (error type: missing)Field ('action',) - Field required (error type…
- `stranger__universal_group_focused_moral__s11` reflection: schema_failure — Your response failed to match the expected schema. Please try again. Field ('short_term_plan',) - Field required (error type: missing)Field ('action',) - Field required (error type…
- `stranger__universal_group_focused_moral__s11` first_stage: schema_failure — Your response failed to match the expected schema. Please try again. Field ('action',) - Field required (error type: missing)
- `stranger__universal_group_focused_moral__s11` reflection: schema_failure — Your response failed to match the expected schema. Please try again. Field ('action',) - Field required (error type: missing)
- `stranger__universal_group_focused_moral__s11` first_stage: schema_failure — Your response failed to match the expected schema. Please try again. Field ('action',) - Field required (error type: missing)
- `stranger__universal_group_focused_moral__s11` reflection: schema_failure — Your response failed to match the expected schema. Please try again. Field ('action',) - Field required (error type: missing)
- `stranger__universal_group_focused_moral__s11` first_stage: schema_failure — Your response failed to match the expected schema. Please try again. Field ('action',) - Field required (error type: missing)
- `stranger__universal_group_focused_moral__s11` reflection: invalid_json — Invalid JSON syntax: Unterminated string starting at: line 2 column 17 (char 18)
- `unfair_partner__reciprocal_group_focused_moral__s11` first_stage: schema_failure — Your response failed to match the expected schema. Please try again. Field ('action',) - Field required (error type: missing)
- `unfair_partner__reciprocal_group_focused_moral__s29` first_stage: schema_failure — Your response failed to match the expected schema. Please try again. Field ('action',) - Field required (error type: missing)
- `unfair_partner__reciprocal_group_focused_moral__s29` reflection: schema_failure — Your response failed to match the expected schema. Please try again. Field ('action',) - Field required (error type: missing)

## Assessment

All 102/102 final decisions are valid after the repository's validation/retry path, and the artifact audit passes with independent per-agent checkpoint memories. First drafts pass only 89/102 cases (87.3%), below the 90% target; critical memory recall is 27/32 (84.4%), also below target. The deterministic moral-role proxy is 29/36 judgeable cases (80.6%), while only 6/60 matched moral-type pairs differ on the help/withhold proxy; intervals are wide and blinded human review is unresolved. There were 42 explicit context-window request failures and no detected silent truncation. The model is usable as a locally hosted exploratory engine with the original reflection, validation, retries, and checkpointing enabled, but this pilot does not establish reliable unattended moral behavior or long-run suitability. It does not reproduce the paper's published scores or evolutionary findings.

## Limitations and remaining tests

The sample is intentionally small and task-specific. The blinded review tables are prepared, but human judgments of moral-role consistency and rationale quality remain unresolved. Memory recall is measured with deterministic fact-slot checks, not a comprehensive test of long-context reasoning; one short previous-attack case used 31,601 input tokens and additional requests hit the 32K context ceiling. No silent truncation was detected. The exact causal effect of memory or reflection cannot be inferred without ablations, which this protocol prohibits. Postflight regression tests passed: `python -m pytest -q` reported **101 passed, 2 Pydantic deprecation warnings** in 17.50 seconds; Python compilation and all PowerShell parser checks passed. The offline artifact audit passed with all 102 final responses revalidated and no integrity errors. Raw test output: `data/qwen38_capability_evaluation_20261016/preflight/regression_tests_postflight.log`.

<!-- QWEN38_FULL_BASELINE_START -->
## Experiment D — Full 80-life-step baseline

**Status: stopped at the user's explicit request; incomplete.** This follow-on run uses one visible-type baseline setting and one fixed environment seed; it is not the paper's full 20-run, four-setting study. It did not complete the 80-life-step baseline and cannot support a replication claim. The paper reports results across 20 runs/four settings ([ACL paper](https://aclanthology.org/2026.acl-long.1339/)).

- Started: 2026-10-16T04:34:56Z; stopped: 2026-10-16T05:41:54Z; run ID: `qwen38-full-baseline-20261016`.
- Progress: **3/240 simulation ticks** (1/80 life-step cycles committed); initial agents: 8, two per active moral type. Tick 4 was interrupted before a result or checkpoint commit, so there is no final checkpoint.
- Config: `configZ_major_v3` prompts/rules, `max_life_steps=80`, `communication_and_sharing_steps=2`, `two_stage_model=true`, local `Qwen3.8-27B-Q4_K_M`, `127.0.0.1:8080/v1`, single inference slot, seed `20261016`.
- Committed decision totals: 24 successful and 0 persistent failed decisions across ticks 1–3; outer tick recoveries: 0; server restarts: 0. Tick 4 agent 4 failed after six attempts, and agent 5's first-stage request was active when the run was stopped. Tick 4 remains incomplete; its partial files are not included in committed totals.
- Elapsed wall time: 1.12 h. Raw prompts/responses, checkpoints, tick results, and telemetry: `data/qwen38_full_baseline_20261016/`; the explicit stop record is `data/qwen38_full_baseline_20261016/manual_stop.json`; original file checkpoints/logs: `data/qwen38-full-baseline-20261016/`.
- Captured completion records: 72, including retries; one recorded call error. Stage totals were not finalized because the run stopped mid-tick. One agent 4 response had invalid JSON at 32,636 input tokens and 32,767/32,768 slot occupancy; see the raw completion record and call logs.
- RTX 3090 board memory samples: 3,649; used VRAM 20,139–20,278 MiB (mean 20,200.2 MiB) during the run; 957 MiB after stopping the run-specific processes.

The runner calls the repository's original `simulation_step.step()` and action/environment code. It keeps successful per-agent decisions cached across failed tick attempts, commits a checkpoint and random-generator state after every completed tick, records unresolved decision failures, and does not alter prompts or reduce agents. One environment seed is fixed; llama-server's default random generation seed remains nondeterministic.

| Tick | Cycle / phase | Decisions | Alive after | Population by type | Actions proposed | Persistent failures | Tick seconds |
|---:|---|---:|---:|---|---|---:|---:|
| 1 | 1 / social/coordination | 8/8 | 8 | {'kin_focused_moral': 2, 'reciprocal_group_focused_moral': 2, 'reproductive_selfish': 2, 'universal_group_focused_moral': 2} | communicate:5, do_nothing:2, rob:1 | 0 | 1132.8 |
| 2 | 1 / social/coordination | 8/8 | 8 | {'kin_focused_moral': 2, 'reciprocal_group_focused_moral': 2, 'reproductive_selfish': 2, 'universal_group_focused_moral': 2} | communicate:5, do_nothing:3 | 0 | 939.74 |
| 3 | 1 / production/action | 8/8 | 10 | {'kin_focused_moral': 2, 'reciprocal_group_focused_moral': 2, 'reproductive_selfish': 4, 'universal_group_focused_moral': 2} | collect:2, hunt:4, reproduce:2 | 0 | 961.26 |

The three committed ticks passed their saved state/checkpoint integrity checks; each recorded eight verified memory updates, independent agent memory objects, and consistent relationships. The full 80-life-step baseline, complete-run metrics, and final state were not evaluated. All run-specific supervisors, workers, finalizer watcher, and the local llama-server were terminated; the endpoint is offline.
<!-- QWEN38_FULL_BASELINE_END -->
