# CL-bench Evaluation: Baseline vs ACE Comparison (V3)

## Table 1: Solving Rate by Category

| Model Names | Overall (%) | Domain Knowledge Reasoning n=71 (%) | Rule System Application n=60 (%) | Procedural Task Execution n=46 (%) | Empirical Discovery & Simulation n=23 (%) |
|---|---|---|---|---|---|
| GPT-5.1 (High) baseline | 12.0 | 11.3 | 11.7 | 15.2 | 8.7 |
| GPT-5.1 (High) + ACE | 16.0 | 14.1 | 20.0 | 15.2 | 13.0 |
| Delta | +33.3% | +25.0% | +71.4% | +0.0% | +50.0% |

## Table 2: Error Analysis Distribution

| Model Names | Context Ignored (%) | Context Misused (%) | Format Error (%) | Refusal (%) |
|---|---|---|---|---|
| Baseline | 57.5 | 3.0 | 40.5 | 21.5 |
| ACE | 53.5 | 4.0 | 42.0 | 22.0 |

## Table 3: Token Usage, Latency, and Cost

| Metric | Baseline | ACE | Delta |
|---|---|---|---|
| Avg Tokens/Task | 12,101 | 12,145 | +0.4% |
| Avg Prompt Tokens | 10,008 | 10,141 | +1.3% |
| Avg Completion Tokens | 2,093 | 2,004 | -4.2% |
| Total Tokens | 2,420,199 | 2,429,056 | +0.4% |
| Avg Latency (ms) | 24,481 | 24,025 | -1.9% |
| p50 Latency (ms) | 24,120 | 20,243 | -16.1% |
| p95 Latency (ms) | 51,742 | 56,241 | +8.7% |
| Estimated Cost ($) | $6.69 | $6.54 | -2.1% |

## Table 4: Per-Category Token Usage and Latency

| Category | Baseline Avg Tokens | ACE Avg Tokens | Baseline Avg Latency (ms) | ACE Avg Latency (ms) |
|---|---|---|---|---|
| DKR | 9,180 | 9,281 | 24,681 | 26,742 |
| RSA | 15,326 | 15,403 | 23,553 | 21,343 |
| PTE | 9,283 | 9,176 | 27,759 | 26,185 |
| EDS | 18,338 | 18,426 | 19,731 | 18,318 |