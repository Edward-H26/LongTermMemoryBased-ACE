# CL-bench Evaluation: Baseline vs ACE Comparison (V4)

## Table 1: Solving Rate by Category

| Model | Overall (%) | DKR n=3 (%) | RSA n=2 (%) |
|---|---|---|---|
| GPT-5.1 (High) baseline | 40.0 | 33.3 | 50.0 |
| GPT-5.1 (High) + ACE | 20.0 | 0.0 | 50.0 |
| Delta | -20.0 | -33.3 | +0.0 |

## Table 2: Error Analysis Distribution

| Model | Context Ignored (%) | Context Misused (%) | Format Error (%) | Refusal (%) |
|---|---|---|---|---|
| Baseline | 40.0 | 0.0 | 60.0 | 0.0 |
| ACE | 60.0 | 0.0 | 40.0 | 0.0 |

## Table 3: Token Usage, Latency, and Cost

| Metric | Baseline | ACE | Delta |
|---|---|---|---|
| Avg Tokens/Task | 11,028 | 11,177 | +150 |
| Avg Prompt Tokens | 9,357 | 9,489 | +132 |
| Avg Completion Tokens | 1,671 | 1,688 | +18 |
| Total Tokens | 55,138 | 55,886 | +748 |
| Avg Latency (ms) | 27,493 | 29,375 | +1,882 |
| p50 Latency (ms) | 16,177 | 14,830 | -1,347 |
| p95 Latency (ms) | 81,669 | 80,239 | -1,431 |
| Estimated Cost ($) | $0.14 | $0.14 | $+0.00 |

## Table 4: Per-Category Token Usage and Latency

| Category | Baseline Avg Tokens | ACE Avg Tokens | Baseline Avg Latency (ms) | ACE Avg Latency (ms) |
|---|---|---|---|---|
| DKR | 12,161 | 12,350 | 38,315 | 36,609 |
| RSA | 9,328 | 9,418 | 11,261 | 18,523 |

## Table 5: V4 Diagnostics

| Metric | Baseline | ACE | Delta |
|---|---|---|---|
| Carryover Coverage (%) | 0.0 | 0.0 | +0.0 |
| Learned Retrieval Rate | 0.000 | 0.000 | +0.000 |
| Capped Output Rate (%) | 0.0 | 0.0 | +0.0 |
| Mean Step Score | 0.000 | 0.676 | +0.676 |
| Quality Gate Apply Rate (%) | 0.0 | 0.0 | +0.0 |