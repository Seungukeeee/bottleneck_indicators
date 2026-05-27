# Bottleneck Score — Supply Chain Graph Analysis

A series of experiments using SEC 10-K filings to build supply chain networks
and identify structurally dominant companies for portfolio construction.

---

## Series Overview

| Part | Method | Key Finding |
|------|--------|-------------|
| [Part 1](https://medium.com/@hugesisulee/why-cheap-stocks-fail-and-how-supply-chain-graph-analysis-fixes-it-627c86d38108) | Keyword proximity matching | Network centrality + low P/E beats simple value |
| [Part 2](https://medium.com/@hugesisulee/i-gave-my-stock-picking-ai-a-brain-upgrade-and-the-prompt-mattered-more-than-the-model-3faeb0b80abb) | LLM-based classification | Prompt framing changes which companies emerge as bottlenecks |
| Part 3 *(this repo)* | Ablation study | Same model, prompt only changed → +33.9%p alpha gap confirmed |

---

## Repository Structure

```
bottleneck_indicators/
│
├── v1_keyword_baseline.py          # Part 1: keyword proximity pipeline
│
├── bottleneck_indicators_LLM_v2.ipynb  # Part 2: LLM classifier notebook
│                                       # (Structural vs Predictive prompt)
│
├── ablation_stage1_cache.py        # Part 3 Stage 1: download & cache 10-K filings
├── ablation_stage3_experiment.py   # Part 3 Stage 3: run ablation + reasoning analysis
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

> **Note:** There is no Stage 2 file — snippet extraction is included inside
> `ablation_stage1_cache.py` as `build_snippet_cache()`, called automatically
> after filing download.

---

## Part 3: Ablation Study

### What This Experiment Does

Part 2 compared two prompts but used different models (Llama 3.3 via Groq vs
Llama 3.1 via Ollama), leaving the question open: was the performance gap
caused by the prompt or the model?

Part 3 fixes that. Model is held constant at **Llama 3.1 (Ollama, local)**.
Only the prompt changes.

```
Condition A — Structural prompt
"What type of relationship does this sentence currently describe?"

Condition B — Predictive prompt
"If Company A expands, will demand for Company B inevitably increase?"
```

Each condition runs 3 times to average out LLM stochasticity.

### Why Not Llama 3.3?

Llama 3.3 70B requires ~48GB RAM to run locally. The test machine has 24GB.
Groq free tier daily limits prevent reliable repeated runs.
Model fixed at Llama 3.1 — the prompt effect is the variable of interest,
not model capability.

### Results (3-run average)

| | A: Structural | B: Predictive |
|---|---|---|
| Edges Extracted | 78.7 | 130.3 |
| None Rate | 0.364 | 0.000 |
| Cumulative Return | +8.1% | +42.0% |
| Alpha vs SPY | -17.6%p | +16.3%p |
| Max Drawdown | -19.8% | -14.4% |
| Sharpe Ratio | 0.689 | 1.525 |

Alpha gap: **+33.9%p** from prompt framing alone.

---

## Setup

### 1. Clone & install

```bash
git clone https://github.com/Seungukeeee/bottleneck_indicators.git
cd bottleneck_indicators
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Fill in SEC_EMAIL (required by SEC fair-access policy)
# GROQ_API_KEY is optional — not used in Part 3
```

### 3. Install Ollama and pull Llama 3.1

```bash
# Install Ollama: https://ollama.com
ollama pull llama3.1
ollama serve  # keep running in a separate terminal
```

### 4. Run Part 3

```bash
# Stage 1: download 10-K filings and extract snippets (~15 min, run once)
python ablation_stage1_cache.py

# Stage 3: run ablation experiment + reasoning analysis
python ablation_stage3_experiment.py
```

Stage 1 is resumable — already-cached tickers are skipped automatically.

### Output

```
ablation_cache.db               — SQLite: filings, snippets, results
ablation_results/
├── scores_A_run{1,2,3}.csv     — Bottleneck Score rankings per run
├── scores_B_run{1,2,3}.csv
├── all_runs.csv                — Raw results across all 6 runs
├── summary_aggregated.csv      — 3-run averages by condition
├── ablation_summary_chart.png  — Bar chart: edges / return / alpha
└── reasoning_samples.csv       — Qualitative reasoning comparison (20 pairs)
```

---

## Part 1 & 2: Quick Start

**Part 1 — Keyword baseline**
```bash
python v1_keyword_baseline.py
```
Requires: `SEC_EMAIL` in `.env`, internet access for SEC EDGAR + yfinance.

**Part 2 — LLM notebook**
```bash
jupyter notebook bottleneck_indicators_LLM_v2.ipynb
```
Requires: Ollama running with `llama3.1` pulled.
Run cells in order. After Sections 2 and 3, copy top-6 tickers into Section 4.

---

## SEC EDGAR Policy Note

Per SEC fair-access rules, all HTTP requests must include a valid contact
email in the `User-Agent` header. Set `SEC_EMAIL` in your `.env` file.
Scraping without a valid email may result in IP blocking.

---

## Reference

> *"The contribution of LLMs to relation extraction in the economic field"* (2025)
> Used to select Llama 3.3 70B as the structural classifier model in Part 2.
