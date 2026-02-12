# CL-bench Evaluation: Baseline vs ACE Comparison (V2)

## Table 1: Solving Rate by Category

| Model | Overall (%) | DKR n=20 (%) | EDS n=9 (%) | PTE n=12 (%) | RSA n=159 (%) |
|---|---|---|---|---|---|
| GPT-5.1 (High) baseline | 13.5 | 15.0 | 0.0 | 16.7 | 13.8 |
| GPT-5.1 (High) + ACE | 6.0 | 5.0 | 0.0 | 16.7 | 5.7 |
| Delta | -7.5 | -10.0 | +0.0 | +0.0 | -8.2 |

## Table 2: Error Analysis Distribution

| Model | Context Ignored (%) | Context Misused (%) | Format Error (%) | Refusal (%) |
|---|---|---|---|---|
| Baseline | 66.0 | 6.5 | 54.0 | 9.0 |
| ACE | 73.4 | 4.5 | 68.3 | 28.6 |

## Table 3: Token Usage, Latency, and Cost

| Metric | Baseline | ACE | Delta |
|---|---|---|---|
| Avg Tokens/Task | 17,751 | 114,524 | +96,773 |
| Avg Prompt Tokens | 16,260 | 112,639 | +96,379 |
| Avg Completion Tokens | 1,492 | 1,886 | +394 |
| Total Tokens | 3,550,273 | 22,790,358 | +19,240,085 |
| Avg Latency (ms) | 25,274 | 34,062 | +8,788 |
| p50 Latency (ms) | 21,496 | 30,746 | +9,250 |
| p95 Latency (ms) | 62,922 | 71,853 | +8,931 |
| Estimated Cost ($) | $7.05 | $31.77 | $+24.72 |

## Table 4: Per-Category Token Usage and Latency

| Category | Baseline Avg Tokens | ACE Avg Tokens | Baseline Avg Latency (ms) | ACE Avg Latency (ms) |
|---|---|---|---|---|
| DKR | 11,656 | 71,218 | 22,684 | 31,411 |
| EDS | 14,153 | 111,320 | 35,004 | 49,907 |
| PTE | 26,364 | 214,166 | 40,026 | 40,457 |
| RSA | 18,072 | 112,613 | 23,935 | 33,115 |