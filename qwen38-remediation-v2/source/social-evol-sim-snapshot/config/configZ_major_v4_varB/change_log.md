# configZ_major_v4_varB — Structural & Framing Rewrite

Prompt sensitivity experiment Variant B, derived from `configZ_major_v4`.

## Rewriting Strategy
- **Approach**: Information reordering + narrative frame shift
- Presentation sequence altered (e.g., constraints before identity, or goals last)
- Narrative shifted from direct imperatives to self-referential framing ("As an agent who...")
- Positive/negative frame inversion (e.g., "You won't rob" → "You always choose peaceful means")
- strategies.txt reorganized into thematic sections (Core Principles, Hunting, Allocation, Threats, Reproduction, Communication)
- All action names (`rob`, `fight`, `allocate`, `hunt`, `collect`, `reproduce`, `communicate`, `do_nothing`) kept verbatim

## Files Changed (vs. configZ_major_v4)
- `prompts/morality/*.txt` — all 7 morality profiles rewritten
- `prompts/strategies.txt` — rewritten with reorganized structure

## Files Unchanged
- `settings.json` — identical to v4
- `rules_template.txt` — identical to v4
- `prompts/rules.txt` — identical to v4 (mechanical rules + output schema)
