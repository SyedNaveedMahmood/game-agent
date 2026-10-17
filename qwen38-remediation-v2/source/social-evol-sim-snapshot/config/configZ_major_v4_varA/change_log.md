# configZ_major_v4_varA — Lexical & Syntactic Paraphrase

Prompt sensitivity experiment Variant A, derived from `configZ_major_v4`.

## Rewriting Strategy
- **Approach**: Lexical substitution + syntactic restructuring
- Same information order and paragraph structure preserved
- Synonym replacement for adjectives/verbs (e.g., "genuinely moral" → "deeply ethical")
- Active/passive voice interchanged where natural
- All action names (`rob`, `fight`, `allocate`, `hunt`, `collect`, `reproduce`, `communicate`, `do_nothing`) kept verbatim

## Files Changed (vs. configZ_major_v4)
- `prompts/morality/*.txt` — all 7 morality profiles rewritten
- `prompts/strategies.txt` — rewritten with same structure

## Files Unchanged
- `settings.json` — identical to v4
- `rules_template.txt` — identical to v4
- `prompts/rules.txt` — identical to v4 (mechanical rules + output schema)
