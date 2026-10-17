"""
Token Usage Estimator.

Config-parameterized formula for estimating LLM token consumption and cost.
Calibrated from two independent runs:
  - K=1, N=8, W=15 (measured token counts via litellm)
  - K=0, N=2, W=15 (estimated from checkpoint file sizes)

Derivation (see docs/Token_Usage_Analysis.md §9.2):

  Cost = Pᵢ·Tᵢ + Pₒ·Tₒ

  Tᵢ = M · Σ N(t)·tᵢ(t),  Tₒ = M · Σ N(t)·tₒ(t)

  tᵢ(t) = (1+K)·[Sₚ + O(t)] + K·(R₁ + Pᵣ)
  tₒ(t) = R₁ + K·R₂

  where T₀(t) = Sₚ + O(t) + R₁ is computed via calibrated coefficients,
  and Sₚ, O(t), R₁ are recovered by decomposing T₀.
"""

# Calibrated constants (tokens)
# R1, R2 are effective values consistent with the calibrated T0 intercept (9400)
# and reflection premium (3750): 9400 = S_P + alpha_0 + R1, 3750 = P_R + R2.
_S_P = 7_900   # system prompt
_P_R = 1_750   # reflect prompt
_R1 = 1_500    # effective Call 1 output (calibrated)
_R2 = 2_000    # effective Call 2 output (calibrated)


def _compute_step(t: int, N: int, W: int, K: int) -> tuple[int, int]:
    """Compute (input_tokens, output_tokens) for one agent at step t."""
    # T₀(t) = Sₚ + O(t) + R₁  (calibrated form, constants absorbed)
    t0 = (
        (9_400 + 400 * N)
        + 500 * min(t, 2)
        + (110 + 45 * N) * min(t, W)
        + 55 * max(t - W, 0)
    )

    # Recover Sₚ + O(t) = T₀ - R₁
    sp_plus_o = t0 - _R1

    # tᵢ(t) = (1+K)·[Sₚ + O(t)] + K·(R₁ + Pᵣ)
    t_in = (1 + K) * sp_plus_o + K * (_R1 + _P_R)

    # tₒ(t) = R₁ + K·R₂
    t_out = _R1 + K * _R2

    return t_in, t_out


def estimate_tokens(
    steps: int,
    agents: int,
    two_stage: bool,
    visible_window: int = 15,
    failure_rate: float = 0.10,
    retry_cost: float = 1.2,
) -> dict:
    """
    Estimate total token consumption for a simulation run.

    Args:
        steps: Number of simulation life steps (world.max_life_steps).
        agents: Number of agents (agent.initial_count).
        two_stage: Whether reflection is enabled (llm.two_stage_model).
        visible_window: Observation history window (agent.view.visible_steps).
        failure_rate: Empirical per-attempt retry failure rate (0.0-1.0).
        retry_cost: Retry overhead factor (retry is ~120% of base due to
            accumulated error messages).

    Returns:
        Dict with total_tokens, input_tokens, output_tokens.
    """
    K = 1 if two_stage else 0
    N = agents
    W = visible_window

    total_in = 0
    total_out = 0
    for t in range(1, steps + 1):
        t_in, t_out = _compute_step(t, N, W, K)
        total_in += N * t_in
        total_out += N * t_out

    M = 1 + failure_rate * retry_cost
    total_in = int(total_in * M)
    total_out = int(total_out * M)

    return {
        "total_tokens": total_in + total_out,
        "input_tokens": total_in,
        "output_tokens": total_out,
    }


def estimate_cost(tokens: dict, price_in: float, price_out: float) -> float:
    """
    Calculate dollar cost given per-million-token prices.

    Args:
        tokens: Dict from estimate_tokens() with input_tokens and output_tokens.
        price_in: Price per million input tokens (e.g. 0.25 for gpt-5-mini).
        price_out: Price per million output tokens (e.g. 2.00 for gpt-5-mini).

    Returns:
        Cost in dollars.
    """
    return (
        tokens["input_tokens"] * price_in + tokens["output_tokens"] * price_out
    ) / 1_000_000


def format_estimate(
    steps: int,
    agents: int,
    two_stage: bool,
    visible_window: int = 15,
    failure_rate: float = 0.10,
    retry_cost: float = 1.2,
) -> str:
    """Return a formatted string summarizing the token/cost estimate."""
    K = 1 if two_stage else 0
    N = agents
    W = visible_window

    tokens = estimate_tokens(
        steps, agents, two_stage, visible_window, failure_rate, retry_cost
    )

    lines = []
    lines.append("=" * 60)
    lines.append("  Token Usage Estimate")
    lines.append("=" * 60)
    lines.append(f"  Agents (N):          {N}")
    lines.append(f"  Steps (S):           {steps}")
    lines.append(f"  Two-stage (K):       {'Yes' if two_stage else 'No'}")
    lines.append(f"  Visible window (W):  {W}")
    lines.append(f"  Failure rate (f):    {failure_rate:.0%}")
    lines.append("-" * 60)

    milestones = sorted(set(
        [1, 2, min(5, steps), min(W, steps), steps]
    ))
    milestones = [s for s in milestones if 1 <= s <= steps]

    lines.append(f"  {'Step':>6}  {'Input':>10}  {'Output':>10}  {'Total':>10}  {'Cumul.':>12}")
    cumul = 0
    for t in range(1, steps + 1):
        t_in, t_out = _compute_step(t, N, W, K)
        t_agent = t_in + t_out
        per_step = N * t_agent
        cumul += per_step
        if t in milestones:
            lines.append(
                f"  {t:>6}  {t_in:>10,}  {t_out:>10,}  {t_agent:>10,}  {cumul:>12,}"
            )
    lines.append("-" * 60)

    M = 1 + failure_rate * retry_cost
    lines.append(f"  Retry multiplier:    x{M:.2f}")
    lines.append(f"  Total tokens:        {tokens['total_tokens']:>12,}")
    lines.append(f"    Input tokens:      {tokens['input_tokens']:>12,}")
    lines.append(f"    Output tokens:     {tokens['output_tokens']:>12,}")

    cost = estimate_cost(tokens, price_in=0.25, price_out=2.00)
    lines.append("-" * 60)
    lines.append(f"  Est. cost (gpt-5-mini):  ${cost:.2f}")
    lines.append(f"    ($0.25/M in, $2.00/M out)")
    lines.append("=" * 60)

    return "\n".join(lines)
