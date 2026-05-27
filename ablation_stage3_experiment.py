"""
Ablation Study — Stage 3: Experiment Runner
============================================
Runs 2 conditions × 3 repetitions using only the local cache.
Zero SEC requests during this stage.

Conditions:
    A — Llama 3.1 (Ollama) + Structural prompt
        "What type of relationship does this sentence currently describe?"
    B — Llama 3.1 (Ollama) + Predictive prompt
        "If Company A expands, will demand for Company B inevitably increase?"

Design rationale:
    Groq free tier daily limit prevents reliable repeated runs with Llama 3.3.
    Model fixed at Llama 3.1 (Ollama, local) to isolate pure prompt effect.
    A vs B = controlled ablation: same model, same data, prompt only changes.

Usage:
    python ablation_stage3_experiment.py

Output:
    ablation_cache.db        — experiment_results + reasoning_samples tables
    ablation_results/        — per-condition CSV + summary + reasoning report
"""

import os
import json
import re
import time
import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

DB_PATH     = "ablation_cache.db"
RESULTS_DIR = Path("ablation_results")
RESULTS_DIR.mkdir(exist_ok=True)

N_RUNS         = 3
N_REASONING    = 20   # Number of samples to collect for reasoning analysis

BACKTEST_START = "2025-05-18"
BACKTEST_END   = "2026-05-18"
BIG_TECH       = ['MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA']

TICKER_TO_NAME = {
    'MSFT': 'Microsoft',      'AAPL': 'Apple',              'NVDA': 'NVIDIA',
    'AVGO': 'Broadcom',       'ORCL': 'Oracle',             'CRM':  'Salesforce',
    'ADBE': 'Adobe',          'CSCO': 'Cisco Systems',      'AMD':  'Advanced Micro Devices',
    'QCOM': 'QUALCOMM',       'TXN':  'Texas Instruments',  'INTU': 'Intuit',
    'IBM':  'International Business Machines',               'AMAT': 'Applied Materials',
    'MU':   'Micron Technology','NOW': 'ServiceNow',         'LRCX': 'Lam Research',
    'ADI':  'Analog Devices', 'PANW': 'Palo Alto Networks', 'KLAC': 'KLA',
    'SNPS': 'Synopsys',       'CDNS': 'Cadence Design Systems','MSI':'Motorola Solutions',
    'APH':  'Amphenol',       'CDW':  'CDW',                'TEL':  'TE Connectivity',
    'FTNT': 'Fortinet',       'ANET': 'Arista Networks',    'KEYS': 'Keysight Technologies',
    'GLW':  'Corning',        'TER':  'Teradyne',           'STX':  'Seagate Technology',
    'NTAP': 'NetApp',         'FSLR': 'First Solar',        'TYL':  'Tyler Technologies',
    'AKAM': 'Akamai Technologies','GEN':'Gen Digital',      'JNPR': 'Juniper Networks',
    'QRVO': 'Qorvo',          'SWKS': 'Skyworks Solutions', 'WDC':  'Western Digital',
    'ENPH': 'Enphase Energy', 'TRMB': 'Trimble',            'ZBRA': 'Zebra Technologies',
    'PTC':  'PTC',            'VRT':  'Vertiv Holdings',    'ETN':  'Eaton',
    'HPE':  'Hewlett Packard Enterprise'
}

RELATIONSHIP_WEIGHTS = {
    "partner": 1.0, "customer": 0.8, "supplier": 0.6,
    "competitor": 0.2, "none": 0.0
}

SUB_MULTIPLIER = {"low": 1.0, "medium": 0.5, "high": 0.1}

# Llama 3.1 via Ollama only — no Groq
# Groq free tier daily limit prevents reliable 3-run repetition with Llama 3.3
CONDITIONS = {
    "A": {"model": "llama3.1", "backend": "ollama", "prompt": "structural"},
    "B": {"model": "llama3.1", "backend": "ollama", "prompt": "predictive"},
}


# ── Prompts ───────────────────────────────────────────────────────────────────

def build_structural_prompt(sentence: str, company_a: str, company_b: str) -> str:
    """
    Prompt A: classify the *current* relationship type.
    Asks the model to describe what kind of business relationship
    already exists based on the 10-K text.
    """
    return f"""You are analyzing SEC 10-K filings to classify business relationships.
Company A (the filing company): {company_a}
Company B (mentioned company): {company_b}
Sentence from the 10-K: \"{sentence}\"

Classify the relationship from Company A's perspective.
Reply with a SINGLE valid JSON object and nothing else. No explanation, no markdown.

{{"relationship_type": "partner|customer|supplier|competitor|none",
  "confidence": 0.0-1.0,
  "direction": "A_depends_on_B|B_depends_on_A|mutual|none"}}

Rules:
- partner: strategic alliance, joint development
- customer: B buys from A (A earns revenue from B)
- supplier: A buys from B (A depends on B for components/services)
- competitor: both compete in same market
- none: incidental mention"""


def build_structural_reasoning_prompt(sentence: str, company_a: str, company_b: str) -> str:
    """
    Prompt A with reasoning: same classification task but asks the model
    to explain its logic before giving the JSON answer.
    Used only for reasoning analysis, not for the main experiment.
    """
    return f"""You are analyzing SEC 10-K filings to classify business relationships.
Company A (the filing company): {company_a}
Company B (mentioned company): {company_b}
Sentence from the 10-K: \"{sentence}\"

First, briefly explain your reasoning (2-3 sentences).
Then provide the classification as a JSON object.

Your reasoning:
[explain why you chose this relationship type]

JSON:
{{"relationship_type": "partner|customer|supplier|competitor|none",
  "confidence": 0.0-1.0,
  "direction": "A_depends_on_B|B_depends_on_A|mutual|none"}}

Rules:
- partner: strategic alliance, joint development
- customer: B buys from A (A earns revenue from B)
- supplier: A buys from B (A depends on B for components/services)
- competitor: both compete in same market
- none: incidental mention"""


def build_predictive_prompt(sentence: str, company_a: str, company_b: str) -> str:
    """
    Prompt B: reason about *future demand transfer* and substitutability.
    Asks the model to make a forward-looking judgment about whether
    Company A's growth would structurally increase demand for Company B.
    """
    return f"""You are a financial analyst evaluating supply chain bottlenecks.
Context from Company A's 10-K filing: \"{sentence}\"

If Company A ({company_a}) expands or the overall industry grows,
will demand for Company B ({company_b})'s products/services inevitably increase?

Reply with a SINGLE valid JSON object and nothing else. No explanation, no markdown.

{{"demand_leverage_score": 0.0-1.0,
  "substitutability": "low|medium|high",
  "direction": "A_depends_on_B|B_depends_on_A|mutual|none"}}

- demand_leverage_score: 1.0 = B is critical; A MUST buy more from B as A grows
- substitutability low  = B has monopoly / high switching costs
- substitutability high = B is a commodity; A can easily switch vendors"""


def build_predictive_reasoning_prompt(sentence: str, company_a: str, company_b: str) -> str:
    """
    Prompt B with reasoning: same predictive task but asks the model
    to explain its economic logic before giving the JSON answer.
    Used only for reasoning analysis, not for the main experiment.
    """
    return f"""You are a financial analyst evaluating supply chain bottlenecks.
Context from Company A's 10-K filing: \"{sentence}\"

If Company A ({company_a}) expands or the overall industry grows,
will demand for Company B ({company_b})'s products/services inevitably increase?

First, briefly explain your economic reasoning (2-3 sentences).
Then provide your assessment as a JSON object.

Your reasoning:
[explain the demand transfer logic and substitutability judgment]

JSON:
{{"demand_leverage_score": 0.0-1.0,
  "substitutability": "low|medium|high",
  "direction": "A_depends_on_B|B_depends_on_A|mutual|none"}}

- demand_leverage_score: 1.0 = B is critical; A MUST buy more from B as A grows
- substitutability low  = B has monopoly / high switching costs
- substitutability high = B is a commodity; A can easily switch vendors"""


# ── LLM Caller ────────────────────────────────────────────────────────────────

def call_ollama(prompt: str, model: str) -> str:
    """Call local Ollama instance. No rate limits."""
    import ollama
    response = ollama.chat(
        model=model,
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response['message']['content']


def parse_json_response(raw: str) -> dict:
    """
    Robustly parse LLM JSON output.
    Handles: extra text before/after JSON, markdown fences, multiple blocks.
    """
    raw = raw.strip()
    # Strip markdown fences
    raw = re.sub(r'^```json\s*|^```\s*|```$', '', raw, flags=re.MULTILINE).strip()
    # Find first complete JSON object
    start = raw.find('{')
    end   = raw.find('}')
    if start == -1 or end == -1:
        raise ValueError("No JSON object found")
    return json.loads(raw[start:end + 1])


# ── Classifier ────────────────────────────────────────────────────────────────

def classify(snippet: str, company_a: str, company_b: str,
             model: str, prompt_type: str) -> dict:
    """
    Unified classifier for main experiment.
    Returns: weight, direction, rel_type, confidence.
    Falls back to zero-weight on any error with visible error message.
    """
    prompt = (build_structural_prompt(snippet, company_a, company_b)
              if prompt_type == "structural"
              else build_predictive_prompt(snippet, company_a, company_b))
    try:
        raw    = call_ollama(prompt, model)
        result = parse_json_response(raw)

        if prompt_type == "structural":
            rel_type   = result.get("relationship_type", "none")
            confidence = float(result.get("confidence", 0.0))
            direction  = result.get("direction", "none")
            weight     = RELATIONSHIP_WEIGHTS.get(rel_type, 0.0) * confidence
        else:
            demand     = float(result.get("demand_leverage_score", 0.0))
            sub        = result.get("substitutability", "high")
            direction  = result.get("direction", "none")
            rel_type   = "predictive"
            confidence = demand
            weight     = demand * SUB_MULTIPLIER.get(sub, 0.1)

        return {"weight": weight, "direction": direction,
                "rel_type": rel_type, "confidence": confidence}

    except Exception as e:
        print(f"  ❌ classify error ({company_a}→{company_b}): {type(e).__name__}: {e}")
        return {"weight": 0.0, "direction": "none",
                "rel_type": "none", "confidence": 0.0}


# ── Reasoning Extractor ───────────────────────────────────────────────────────

def extract_reasoning(snippet: str, company_a: str, company_b: str,
                      model: str, prompt_type: str) -> dict:
    """
    Ask the model to explain its logic before giving the JSON answer.
    Extracts the reasoning text and the JSON separately.
    Used for qualitative analysis only — not for main experiment scoring.

    Returns:
        dict with keys: reasoning, result_json, raw_response
    """
    prompt = (build_structural_reasoning_prompt(snippet, company_a, company_b)
              if prompt_type == "structural"
              else build_predictive_reasoning_prompt(snippet, company_a, company_b))
    try:
        raw = call_ollama(prompt, model)

        # Extract reasoning: text before the JSON block
        json_start = raw.find('{')
        reasoning  = raw[:json_start].strip() if json_start > 0 else ""

        # Clean up reasoning — remove prompt echoes
        for marker in ["Your reasoning:", "JSON:", "[explain"]:
            reasoning = reasoning.replace(marker, "").strip()

        # Extract JSON
        result_json = parse_json_response(raw) if json_start != -1 else {}

        return {
            "reasoning":     reasoning,
            "result_json":   result_json,
            "raw_response":  raw
        }
    except Exception as e:
        return {
            "reasoning":    f"Error: {e}",
            "result_json":  {},
            "raw_response": ""
        }


def run_reasoning_analysis(
    n_samples: int = N_REASONING,
    model: str = "llama3.1",
    db_path: str = DB_PATH
) -> pd.DataFrame:
    """
    Collect reasoning samples from both prompt types on the same snippets.
    Compares the model's logic side-by-side for Structural vs Predictive.

    Selects n_samples snippets that produced different outcomes in A vs B
    (i.e., one got weight > 0.3, the other didn't) — these are the most
    informative cases for understanding the fundamental difference.

    Saves results to:
        ablation_results/reasoning_samples.csv
        ablation_cache.db → reasoning_samples table
    """
    print("\n" + "=" * 55)
    print("  Reasoning Analysis")
    print("  Comparing Structural vs Predictive logic on same snippets")
    print("=" * 55)

    # Load snippets
    conn     = sqlite3.connect(db_path)
    snippets = pd.read_sql("SELECT * FROM snippets", conn)

    # Try to find divergent cases from experiment_results
    try:
        exp = pd.read_sql("""
            SELECT ticker_a, ticker_b, prompt_type, AVG(weight) as avg_weight
            FROM experiment_results
            WHERE model = 'llama3.1'
            GROUP BY ticker_a, ticker_b, prompt_type
        """, conn)

        if not exp.empty:
            pivot = exp.pivot_table(
                index=['ticker_a', 'ticker_b'],
                columns='prompt_type',
                values='avg_weight'
            ).reset_index()

            if 'structural' in pivot.columns and 'predictive' in pivot.columns:
                pivot['diff'] = abs(
                    pivot['predictive'].fillna(0) - pivot['structural'].fillna(0)
                )
                top_divergent = pivot.nlargest(n_samples, 'diff')
                sample_pairs  = list(zip(
                    top_divergent['ticker_a'], top_divergent['ticker_b']
                ))
                print(f"  → Using {len(sample_pairs)} most divergent pairs from experiment results")
            else:
                sample_pairs = None
        else:
            sample_pairs = None
    except Exception:
        sample_pairs = None

    # Fallback: random sample
    if not sample_pairs:
        sample_df    = snippets.sample(min(n_samples, len(snippets)), random_state=42)
        sample_pairs = list(zip(sample_df['ticker_a'], sample_df['ticker_b']))
        print(f"  → Using {len(sample_pairs)} random pairs (no experiment results found)")

    conn.close()

    records = []
    for i, (ta, tb) in enumerate(sample_pairs, 1):
        name_a = TICKER_TO_NAME.get(ta, ta)
        name_b = TICKER_TO_NAME.get(tb, tb)

        # Get snippet
        row = snippets[(snippets['ticker_a'] == ta) & (snippets['ticker_b'] == tb)]
        if row.empty:
            continue
        snippet = row.iloc[0]['snippet']

        print(f"\n  [{i:02d}/{len(sample_pairs)}] {name_a} → {name_b}")
        print(f"  Snippet: ...{snippet[:80]}...")

        # Structural reasoning
        print("  Running Structural prompt...")
        struct_out = extract_reasoning(snippet, name_a, name_b, model, "structural")
        print(f"  → Structural reasoning: {struct_out['reasoning'][:120]}...")

        # Predictive reasoning
        print("  Running Predictive prompt...")
        pred_out   = extract_reasoning(snippet, name_a, name_b, model, "predictive")
        print(f"  → Predictive reasoning: {pred_out['reasoning'][:120]}...")

        records.append({
            "ticker_a":              ta,
            "ticker_b":              tb,
            "company_a":             name_a,
            "company_b":             name_b,
            "snippet":               snippet,
            # Structural outputs
            "struct_reasoning":      struct_out["reasoning"],
            "struct_rel_type":       struct_out["result_json"].get("relationship_type", ""),
            "struct_confidence":     struct_out["result_json"].get("confidence", 0.0),
            "struct_direction":      struct_out["result_json"].get("direction", ""),
            # Predictive outputs
            "pred_reasoning":        pred_out["reasoning"],
            "pred_leverage_score":   pred_out["result_json"].get("demand_leverage_score", 0.0),
            "pred_substitutability": pred_out["result_json"].get("substitutability", ""),
            "pred_direction":        pred_out["result_json"].get("direction", ""),
        })

    df = pd.DataFrame(records)

    # Save to CSV
    out_path = RESULTS_DIR / "reasoning_samples.csv"
    df.to_csv(out_path, index=False)

    # Save to DB
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE IF EXISTS reasoning_samples")
    df.to_sql("reasoning_samples", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

    # Print summary report
    print("\n\n" + "=" * 65)
    print("  REASONING ANALYSIS — SUMMARY")
    print("=" * 65)
    for _, row in df.iterrows():
        print(f"\n  {row['company_a']} → {row['company_b']}")
        print(f"  Snippet: ...{row['snippet'][:100]}...")
        print(f"\n  [Structural] rel_type={row['struct_rel_type']} | "
              f"confidence={row['struct_confidence']}")
        print(f"  Reasoning: {row['struct_reasoning'][:200]}")
        print(f"\n  [Predictive] leverage={row['pred_leverage_score']} | "
              f"substitutability={row['pred_substitutability']}")
        print(f"  Reasoning: {row['pred_reasoning'][:200]}")
        print("  " + "-" * 60)

    print(f"\n  Full reasoning saved to: {out_path}")
    print("=" * 65)

    return df


# ── Network & Score ───────────────────────────────────────────────────────────

def build_network(edges: list):
    """Build directed graph and return (G, in_degree_dict)."""
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges)
    return G, dict(G.in_degree(weight='weight'))


def compute_scores(in_degrees: dict, profiles: dict,
                   total_capex: pd.Series) -> pd.DataFrame:
    """
    Bottleneck Score = ((norm_in + 0.1) * (capex_rev_corr + 1)) / (PE / 10)
    Returns empty DataFrame if no scores can be computed.
    """
    max_in  = max(in_degrees.values(), default=1)
    results = []

    for ticker, deg in in_degrees.items():
        if ticker not in profiles:
            continue
        try:
            rev_df = profiles[ticker]['stock'].financials.T
            if 'Total Revenue' not in rev_df.columns:
                continue
            rev = rev_df['Total Revenue']
            rev.index = pd.to_datetime(rev.index).year
            rev = rev[rev.index <= 2024].dropna()
            rev = rev.groupby(rev.index).last()

            combined = pd.DataFrame({'Capex': total_capex, 'Rev': rev}).dropna()
            if len(combined) < 2:
                continue

            corr    = combined.corr().iloc[0, 1]
            norm_in = deg / max_in
            pe      = profiles[ticker]['pe']
            score   = ((norm_in + 0.1) * (corr + 1)) / (max(pe, 5) / 10)

            results.append({
                'Ticker':           ticker,
                'In_Degree':        round(deg, 3),
                'PE':               round(pe, 2),
                'Bottleneck_Score': round(score, 4)
            })
        except Exception as e:
            print(f"  [{ticker}] score error: {e}")
            continue

    if not results:
        return pd.DataFrame()

    return (pd.DataFrame(results)
              .set_index('Ticker')
              .sort_values('Bottleneck_Score', ascending=False))


# ── Backtest ──────────────────────────────────────────────────────────────────

def run_backtest(top6: list, label: str) -> dict:
    """Equal-weighted 1-year backtest. Returns cumulative return, MDD, Sharpe."""
    tickers = list(set(top6 + ['SPY']))
    try:
        prices = yf.download(
            tickers, start=BACKTEST_START, end=BACKTEST_END,
            auto_adjust=True, progress=False
        )['Close'].dropna()
    except Exception as e:
        print(f"  Backtest download error: {e}")
        return {}

    available = [t for t in top6 if t in prices.columns]
    if not available:
        return {}

    port     = prices[available].pct_change().dropna().mean(axis=1)
    spy      = prices['SPY'].pct_change().dropna()
    cum_port = (1 + port).cumprod()
    cum_spy  = (1 + spy).cumprod()

    roll_max = cum_port.cummax()
    mdd      = ((cum_port - roll_max) / roll_max).min()
    sharpe   = (port.mean() / port.std()) * (252 ** 0.5) if port.std() > 0 else 0.0

    return {
        'label':             label,
        'top6':              available,
        'cumulative_return': round((cum_port.iloc[-1] - 1) * 100, 2),
        'spy_return':        round((cum_spy.iloc[-1]  - 1) * 100, 2),
        'alpha':             round((cum_port.iloc[-1] - cum_spy.iloc[-1]) * 100, 2),
        'mdd':               round(mdd * 100, 2),
        'sharpe':            round(sharpe, 3),
    }


# ── Single Run ────────────────────────────────────────────────────────────────

def run_condition(condition_id: str, run_id: int,
                  profiles: dict, total_capex: pd.Series,
                  db_path: str = DB_PATH):
    """
    One condition × one repetition.
    Reads snippets from cache — no SEC calls.
    """
    cfg       = CONDITIONS[condition_id]
    model     = cfg['model']
    ptype     = cfg['prompt']
    label     = f"{condition_id}_run{run_id}"

    print(f"\n{'='*55}")
    print(f"  Condition {condition_id} | Run {run_id}/{N_RUNS}")
    print(f"  Model: {model} (Ollama) | Prompt: {ptype}")
    print(f"{'='*55}")

    conn     = sqlite3.connect(db_path)
    snippets = pd.read_sql("SELECT * FROM snippets", conn)
    conn.close()

    if snippets.empty:
        print("❌ No snippets. Run Stage 1 & 2 first.")
        return None

    edges, raw_results = [], []
    total = len(snippets)

    for i, row in snippets.iterrows():
        ta, tb    = row['ticker_a'], row['ticker_b']
        name_a    = TICKER_TO_NAME.get(ta, ta)
        name_b    = TICKER_TO_NAME.get(tb, tb)
        result    = classify(row['snippet'], name_a, name_b, model, ptype)
        weight    = result['weight']
        direction = result['direction']

        raw_results.append({
            'condition':   condition_id,
            'model':       model,
            'prompt_type': ptype,
            'run_id':      run_id,
            'ticker_a':    ta,
            'ticker_b':    tb,
            'rel_type':    result['rel_type'],
            'confidence':  result['confidence'],
            'direction':   direction,
            'weight':      weight,
            'created_at':  datetime.now().isoformat()
        })

        if weight > 0.3:
            if direction == 'A_depends_on_B':
                edges.append((ta, tb, weight))
            elif direction == 'B_depends_on_A':
                edges.append((tb, ta, weight))
            elif direction == 'mutual':
                edges.append((ta, tb, weight))
                edges.append((tb, ta, weight))

        if (i + 1) % 100 == 0:
            print(f"  [{i+1:,}/{total:,}] edges so far: {len(edges)}")

    # Persist raw LLM outputs to DB
    conn = sqlite3.connect(db_path)
    pd.DataFrame(raw_results).to_sql(
        'experiment_results', conn, if_exists='append', index=False
    )
    conn.commit()
    conn.close()

    print(f"\n  Total edges extracted: {len(edges)}")

    if not edges:
        print("  ⚠️ No edges above threshold — skipping")
        return None

    G, in_degrees = build_network(edges)
    score_df      = compute_scores(in_degrees, profiles, total_capex)

    if score_df.empty:
        print("  ⚠️ Score computation failed — skipping")
        return None

    top6 = score_df.head(6).index.tolist()
    bt   = run_backtest(top6, label)

    none_rate = sum(1 for r in raw_results if r['rel_type'] == 'none') / len(raw_results)
    avg_conf  = float(np.mean([r['confidence'] for r in raw_results]))

    summary = {
        'condition':       condition_id,
        'run_id':          run_id,
        'model':           model,
        'prompt_type':     ptype,
        'edges_extracted': len(edges),
        'none_rate':       round(none_rate, 3),
        'avg_confidence':  round(avg_conf, 3),
        'top6':            top6,
        **bt
    }

    score_df.to_csv(RESULTS_DIR / f"scores_{condition_id}_run{run_id}.csv")
    return summary


# ── All Conditions ────────────────────────────────────────────────────────────

def run_all_conditions(profiles: dict, total_capex: pd.Series):
    """
    Run A and B, each N_RUNS times.
    No cooldown needed — Ollama has no rate limits.
    """
    all_summaries = []

    for cond_id in ['A', 'B']:
        for run_id in range(1, N_RUNS + 1):
            summary = run_condition(cond_id, run_id, profiles, total_capex)
            if summary:
                all_summaries.append(summary)
                print(f"\n  ✅ {cond_id} run {run_id}: "
                      f"return={summary.get('cumulative_return')}% | "
                      f"alpha={summary.get('alpha')}%p | "
                      f"edges={summary['edges_extracted']}")

    if not all_summaries:
        print("❌ No successful runs. Check logs above.")
        return pd.DataFrame()

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(RESULTS_DIR / "all_runs.csv", index=False)

    agg = summary_df.groupby(['condition', 'model', 'prompt_type']).agg(
        edges_mean      =('edges_extracted',    'mean'),
        none_rate_mean  =('none_rate',          'mean'),
        confidence_mean =('avg_confidence',     'mean'),
        return_mean     =('cumulative_return',  'mean'),
        return_std      =('cumulative_return',  'std'),
        alpha_mean      =('alpha',              'mean'),
        mdd_mean        =('mdd',                'mean'),
        sharpe_mean     =('sharpe',             'mean'),
    ).round(3)

    agg.to_csv(RESULTS_DIR / "summary_aggregated.csv")

    print("\n\n" + "=" * 65)
    print("  ABLATION STUDY — FINAL SUMMARY")
    print("  Condition A: Llama 3.1 + Structural prompt")
    print("  Condition B: Llama 3.1 + Predictive prompt")
    print("=" * 65)
    print(agg.to_string())
    print("=" * 65)
    print(f"\nResults saved to: {RESULTS_DIR}/")

    plot_summary(agg)
    return agg


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_summary(agg: pd.DataFrame):
    """Bar chart comparing A vs B across edges, return, and alpha."""
    conditions = agg.index.get_level_values('condition').tolist()
    colors     = ['#4C72B0', '#DD8452']  # Blue for A, Orange for B

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Ablation Study — Structural vs Predictive Prompt (Llama 3.1, same model)",
        fontsize=13, fontweight='bold'
    )

    metrics = [
        ('edges_mean',  'Edges Extracted (avg)',     axes[0]),
        ('return_mean', 'Cumulative Return % (avg)', axes[1]),
        ('alpha_mean',  'Alpha vs SPY %p (avg)',     axes[2]),
    ]

    for col, title, ax in metrics:
        vals = agg[col].values
        bars = ax.bar(conditions, vals,
                      color=colors[:len(conditions)], edgecolor='black')
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Condition  (A=Structural  |  B=Predictive)")
        ax.axhline(0, color='black', linewidth=0.8)
        for bar, val in zip(bars, vals):
            ypos = (bar.get_height() + abs(val) * 0.02
                    if val >= 0
                    else bar.get_height() - abs(val) * 0.08)
            ax.text(bar.get_x() + bar.get_width() / 2, ypos,
                    f"{val:.1f}", ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ablation_summary_chart.png",
                dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Chart saved: {RESULTS_DIR}/ablation_summary_chart.png")


# ── Shared Resources ──────────────────────────────────────────────────────────

def build_profiles(tickers: list) -> dict:
    """Build P/E profiles once — shared across both conditions."""
    profiles = {}
    print("Building company profiles...")
    for t in tickers:
        try:
            stock = yf.Ticker(t)
            hist  = stock.history(start="2025-05-12", end="2025-05-17")
            price = hist['Close'].iloc[-1] if not hist.empty else None

            pe  = 30
            fin = stock.financials.T
            fin.index = pd.to_datetime(fin.index).year
            if 2024 in fin.index and price:
                row = fin.loc[2024]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                eps = next(
                    (row[c] for c in fin.columns
                     if 'EPS' in c or 'Earnings Per Share' in c), None
                )
                if (eps is None or pd.isna(eps) or eps <= 0) and 'Net Income' in row:
                    shares = row.get('Diluted Average Shares',
                                     row.get('Basic Average Shares', 1))
                    eps = row['Net Income'] / shares if shares > 0 else None
                if eps and eps > 0:
                    pe = price / eps

            profiles[t] = {'pe': pe, 'stock': stock}
        except Exception:
            continue

    print(f"  → {len(profiles)} companies profiled.")
    return profiles


def fetch_big_tech_capex(tickers: list, max_year: int = 2024) -> pd.Series:
    """Aggregate Big Tech CapEx as macro demand anchor."""
    capex_list = []
    for t in tickers:
        try:
            cf = yf.Ticker(t).cashflow.T
            if 'Capital Expenditure' not in cf.columns:
                continue
            capex = cf['Capital Expenditure'].abs()
            capex.index = pd.to_datetime(capex.index).year
            capex = capex[capex.index <= max_year].dropna()
            capex = capex.groupby(capex.index).last()
            if capex.empty:
                continue
            capex_list.append(capex)
        except Exception as e:
            print(f"  [{t}] CapEx error: {e}")
            continue
    if not capex_list:
        return pd.Series(dtype=float)
    return pd.concat(capex_list, axis=1).sum(axis=1)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from ablation_stage1_cache import IT_TICKERS, inspect_cache

    inspect_cache()

    tickers_in_cache = pd.read_sql(
        "SELECT ticker FROM filings", sqlite3.connect(DB_PATH)
    )['ticker'].tolist()

    profiles    = build_profiles(tickers_in_cache)
    total_capex = fetch_big_tech_capex(BIG_TECH)

    if total_capex.empty:
        print("❌ Could not retrieve Big Tech CapEx. Check network.")
        exit(1)

    # ── Step 1: Run main experiment (A vs B, 3 runs each) ─────────────────
    run_all_conditions(profiles, total_capex)

    # ── Step 2: Reasoning analysis (why does each prompt decide differently?)
    print("\n\nStarting reasoning analysis...")
    print("This collects qualitative evidence for the blog — not used in scoring.")
    run_reasoning_analysis(n_samples=N_REASONING)