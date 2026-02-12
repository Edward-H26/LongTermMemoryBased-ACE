# CL-bench Evaluation: Baseline vs ACE Comparison

## Table 1: Solving Rate by Category

| Model | Overall (%) | DKR (%) | EDS (%) | PTE (%) | RSA (%) |
|---|---|---|---|---|---|
| GPT-5.1 (High) baseline | 13.5 | 15.0 | 0.0 | 16.7 | 13.8 |
| GPT-5.1 (High) + ACE | 6.0 | 5.0 | 0.0 | 16.7 | 5.7 |
| Delta | -7.5 | -10.0 | +0.0 | +0.0 | -8.2 |

## Table 2: Error Analysis Distribution

| Model | Context Ignored (%) | Context Misused (%) | Format Error (%) | Refusal (%) |
|---|---|---|---|---|
| Baseline | 66.0 | 6.5 | 54.0 | 9.0 |
| ACE | 73.4 | 4.5 | 68.3 | 28.6 |

## Table 3: Token Usage and Latency

| Model | Avg Tokens/Task | Avg Latency (ms) | Total Tokens |
|---|---|---|---|
| Baseline | 17751 | 25274 | 3,550,273 |
| ACE | 114524 | 34062 | 22,790,358 |