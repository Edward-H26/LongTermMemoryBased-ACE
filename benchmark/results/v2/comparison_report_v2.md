# CL-bench Evaluation: Baseline vs ACE Comparison (V2)

## Table 1: Solving Rate by Category

| Model | Overall (%) | DKR n=20 (%) | EDS n=9 (%) | PTE n=12 (%) | RSA n=159 (%) |
|---|---|---|---|---|---|
| GPT-5.1 (High) baseline | 18.0 | 25.0 | 11.1 | 33.3 | 16.4 |
| GPT-5.1 (High) + ACE | 16.5 | 10.0 | 11.1 | 16.7 | 17.6 |
| Delta | -1.5 | -15.0 | +0.0 | -16.7 | +1.3 |

## Table 2: Error Analysis Distribution

| Model | Context Ignored (%) | Context Misused (%) | Format Error (%) | Refusal (%) |
|---|---|---|---|---|
| Baseline | 61.5 | 5.0 | 49.5 | 11.0 |
| ACE | 58.5 | 4.5 | 47.0 | 13.0 |

## Table 3: Token Usage, Latency, and Cost

| Metric | Baseline | ACE | Delta |
|---|---|---|---|
| Avg Tokens/Task | 18,119 | 18,211 | +92 |
| Avg Prompt Tokens | 16,665 | 16,855 | +190 |
| Avg Completion Tokens | 1,454 | 1,356 | -98 |
| Total Tokens | 3,623,872 | 3,642,275 | +18,403 |
| Avg Latency (ms) | 21,254 | 29,911 | +8,657 |
| p50 Latency (ms) | 15,463 | 21,452 | +5,989 |
| p95 Latency (ms) | 61,664 | 86,677 | +25,013 |
| Estimated Cost ($) | $7.08 | $6.93 | $-0.15 |

## Table 4: Per-Category Token Usage and Latency

| Category | Baseline Avg Tokens | ACE Avg Tokens | Baseline Avg Latency (ms) | ACE Avg Latency (ms) |
|---|---|---|---|---|
| DKR | 11,902 | 12,064 | 16,428 | 32,896 |
| EDS | 14,436 | 14,716 | 24,043 | 35,770 |
| PTE | 27,327 | 27,383 | 40,992 | 34,372 |
| RSA | 18,415 | 18,490 | 20,213 | 28,867 |