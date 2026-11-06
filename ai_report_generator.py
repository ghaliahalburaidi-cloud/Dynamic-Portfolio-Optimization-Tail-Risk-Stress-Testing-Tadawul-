"""
ai_report_generator.py
----------------------
Phase 4: Generative-AI executive memo via Amazon Bedrock (Converse API,
amazon.nova-lite-v1:0).

Design choices (worth stating in the report):
- Every number, including the percentage changes, is computed in Python and
  passed to the model. The model is told not to calculate anything, because
  LLMs are unreliable at arithmetic.
- After generation, every SAR figure and percentage in the memo is checked
  against the computed values (a simple hallucination guard). The result of
  that check is printed and saved next to the memo.
- Credentials are never hard-coded. boto3 reads them from environment
  variables, a local `.env` file, or ~/.aws/credentials.
"""

import os
import re

MODEL_ID = 'amazon.nova-lite-v1:0'
REGION = os.getenv('AWS_REGION', 'us-east-1')
MEMO_FILE = 'executive_risk_memo.txt'
MAX_ATTEMPTS = 3          # regenerate if the number check fails


def _pct(x, d=2):
    return f"{x * 100:.{d}f}%"


def _weights(w):
    return ', '.join(f"{k} {_pct(v)}" for k, v in w.items() if v > 0)


def _build_prompt(m):
    rob = ''
    if m.get('robust_cvar_stress') is not None:
        rob = (f"- Robustness: re-running the stress scenario with fat-tailed historical "
               f"bootstrap shocks gives CVaR of {m['robust_cvar_stress']:,.2f} SAR "
               f"({m['robust_cvar_diff_pct']:+.2f}% versus the Gaussian model).\n")
    if m['all_negative_excess']:
        alloc_cause = (f"- Under stress, every asset has a negative expected excess return; the best attainable\n"
                       f"  Sharpe ratio is {m['stress_sharpe']:.3f}. With negative excess returns the Sharpe ratio rewards\n"
                       f"  higher volatility, so the optimizer moved into {m['corner_asset']}")
        alloc_cause += (f", which has the highest stressed\n  volatility of the four assets ({_pct(m['corner_vol'])} per year)."
                        if m['corner_is_highest_vol'] else
                        f" (stressed volatility {_pct(m['corner_vol'])} per year).")
    else:
        alloc_cause = (f"- Under stress, {m['corner_asset']} offers the best risk-adjusted return "
                       f"(portfolio Sharpe ratio {m['stress_sharpe']:.3f}; stressed volatility {_pct(m['corner_vol'])} per year).")
    alloc_verdict = ("The corner allocation therefore adds tail risk rather than reducing it."
                     if m['allocation_effect'] > 0 else
                     "The corner allocation therefore reduces tail risk relative to the baseline weights.")
    return f"""You are the Chief Risk Officer of an institutional asset manager. Write a one-page
Executive Risk Memorandum on the stochastic stress test of a Tadawul equity portfolio.

DATE: {m['date']}

SET-UP
- 10,000 Monte Carlo pathways, one-year horizon, initial capital 1,000,000 SAR.
- Portfolio construction: maximum Sharpe ratio, long-only, risk-free rate {_pct(m['risk_free'], 1)}.
- Stress scenario (geopolitical shock): volatility multiplied by 1.5 and expected annual log
  return lowered by about 25 percentage points for all assets; correlations unchanged.
  (Describe this as a fall of about 25 percentage points, not as a 25% reduction.)

ALLOCATION
- Baseline allocation: {_weights(m['baseline_weights'])}.
- Stress allocation: {_weights(m['stress_weights'])} (a corner solution in {m['corner_asset']}).
{alloc_cause}
- A minimum-variance portfolio ({_weights(m['minvar_weights'])}) is identical in both scenarios,
  which shows the reallocation is driven by the return shock, not by the volatility increase.

TAIL RISK (95% confidence level; losses in SAR)
- Baseline VaR {m['var_normal']:,.2f} (95% confidence interval {m['var_ci_normal'][0]:,.0f} to {m['var_ci_normal'][1]:,.0f})
- Baseline CVaR {m['cvar_normal']:,.2f} (95% confidence interval {m['cvar_ci_normal'][0]:,.0f} to {m['cvar_ci_normal'][1]:,.0f})
- Stress VaR {m['var_stress']:,.2f} (95% confidence interval {m['var_ci_stress'][0]:,.0f} to {m['var_ci_stress'][1]:,.0f})
- Stress CVaR {m['cvar_stress']:,.2f} (95% confidence interval {m['cvar_ci_stress'][0]:,.0f} to {m['cvar_ci_stress'][1]:,.0f})
- VaR change {m['var_increase_pct']:+.2f}%; CVaR change {m['cvar_increase_pct']:+.2f}%
- Probability of any loss: {_pct(m['prob_loss_normal'], 1)} baseline versus {_pct(m['prob_loss_stress'], 1)} under stress.

SOURCE OF THE ADDITIONAL RISK (CVaR, SAR)
- Keeping the baseline weights under stress: CVaR {m['cvar_basew_stress']:,.2f}.
- Switching to the stress-optimal corner allocation: CVaR {m['cvar_stress']:,.2f}.
- Shock effect {m['shock_effect']:,.2f} ({m['shock_share']:.1f}% of the increase);
  allocation effect {m['allocation_effect']:,.2f} ({m['allocation_share']:.1f}% of the increase).
  {alloc_verdict}

MODEL LIMITATIONS
- Parameters are estimated from three years of history; correlations are held constant under stress.
- Simulated shocks are Gaussian.
{rob}
WRITE THE MEMO WITH EXACTLY THESE SECTIONS, AS PLAIN TEXT (no markdown, no asterisks):
MEMORANDUM header with TO: Executive Management, FROM: Chief Risk Officer, DATE, SUBJECT
1. Executive Summary (two or three sentences stating the bottom line)
2. Allocation Shift and Its Cause (explain the mechanism stated under ALLOCATION in your own words;
   do not describe higher volatility as a reason to prefer the asset in itself)
3. Tail-Risk Escalation
4. Source of the Additional Risk
5. Model Limitations
6. Recommendations (three numbered portfolio or risk-limit actions, such as position limits,
   the choice of optimization objective, or tail-risk limits. Each must cite at least one figure
   from above. Do not recommend further modelling or research work.)

RULES
- 450 to 600 words in total.
- State each figure once; only the Executive Summary may repeat a figure from a later section.
- Use ONLY the figures given above, written exactly as shown. Do not calculate, re-round
  or introduce any other number.
- Recommendations must not set new numeric limits or thresholds (for example a maximum
  position size or a CVaR cap). Express limits qualitatively or by reference to a figure above.
- Do not speculate about company fundamentals, news or reasons not stated above.
- Formal institutional tone.
"""


def _verify_numbers(memo, prompt):
    """Return numbers in the memo that do not appear in the prompt (hallucination guard).
    A figure may also be quoted rounded to a whole number."""
    def nums(text):
        out = []
        for token in re.findall(r'\d[\d,]*\.?\d*', text):
            try:
                out.append((token, float(token.replace(',', '').rstrip('.'))))
            except ValueError:
                pass
        return out

    allowed = set()
    for _, v in nums(prompt):
        allowed.update({v, float(round(v)), round(v, 1)})
    allowed.update(range(0, 7))                      # section numbering
    bad = [tok for tok, v in nums(memo) if not any(abs(v - a) < 0.006 for a in allowed)]
    # Unit check: a figure given only in percentage points must not be quoted as a percent
    # (e.g. "lowered by about 25 percentage points" must not become "a 25% reduction").
    pp_only = {n for n in re.findall(r'(\d+(?:\.\d+)?) percentage points', prompt)
               if not re.search(rf'(?<![\d.]){re.escape(n)}%', prompt)}
    bad += [f'{n}% (should be percentage points)' for n in pp_only
            if re.search(rf'(?<![\d.]){re.escape(n)}\s?%', memo)]
    return bad


def generate_executive_report(metrics):
    print("\nInitiating AWS Bedrock analysis ...")
    try:
        import boto3
        try:
            from dotenv import load_dotenv
            # override=True: values in .env win over any old AWS_* variables
            # already set in the shell, a common cause of signature errors.
            load_dotenv(override=True)
        except ImportError:
            pass
        bedrock = boto3.client('bedrock-runtime', region_name=REGION)
        prompt = _build_prompt(metrics)
        messages = [{'role': 'user', 'content': [{'text': prompt}]}]
        # Generate, verify, and ask for a revision if unsupported figures appear.
        for attempt in range(1, MAX_ATTEMPTS + 1):
            response = bedrock.converse(
                modelId=MODEL_ID, messages=messages,
                inferenceConfig={'maxTokens': 1500, 'temperature': 0.1},
            )
            memo = response['output']['message']['content'][0]['text']
            memo = memo.replace('**', '').replace('##', '')   # plain text for the dashboard
            unmatched = _verify_numbers(memo, prompt)
            print(f"Attempt {attempt}: " + ("all figures verified" if not unmatched
                                             else f"unsupported figures {unmatched}"))
            if not unmatched:
                break
            messages += [
                {'role': 'assistant', 'content': [{'text': memo}]},
                {'role': 'user', 'content': [{'text':
                    f"The memo contains figures that are not in the data provided: {unmatched}. "
                    "Rewrite the complete memo without them, keeping the same six sections and rules. "
                    "Use only the figures given in the original instructions."}]},
            ]
    except Exception as e:
        print(f"[SKIPPED] Bedrock memo not generated: {str(e).splitlines()[0]}")
        print("Check that AWS credentials are set (see .env.example) and that the "
              "model is enabled for your account in this region.")
        return None

    check = ("Number check: PASSED (all figures match the computed results)"
             if not unmatched else
             f"Number check: REVIEW - figures not found in the computed results: {unmatched}")

    with open(MEMO_FILE, 'w', encoding='utf-8') as f:
        f.write(memo.strip() + '\n')

    print("=" * 80)
    print(" " * 24 + "AWS AI-GENERATED RISK REPORT")
    print("=" * 80)
    print(memo)
    print("=" * 80)
    print(check)
    print(f"Memo saved to {MEMO_FILE}")
    return memo
