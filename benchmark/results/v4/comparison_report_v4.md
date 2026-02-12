# CL-bench Evaluation: Baseline vs ACE Comparison (V4)

## Table 1: Solving Rate by Category

| Model Names | Overall (%) | Domain Knowledge Reasoning n=85 (%) | Rule System Application n=62 (%) | Procedural Task Execution n=47 (%) | Empirical Discovery & Simulation n=6 (%) |
|---|---|---|---|---|---|
| GPT-5.1 (High) baseline | 20.5 | 18.8 | 27.4 | 14.9 | 16.7 |
| GPT-5.1 (High) + ACE | 23.5 | 25.9 | 22.6 | 19.1 | 33.3 |
| Delta | +14.6% | +37.5% | -17.6% | +28.6% | +100.0% |

## Table 2: Error Analysis Distribution

| Model Names | Context Ignored (%) | Context Misused (%) | Format Error (%) | Refusal (%) |
|---|---|---|---|---|
| Baseline | 65.5 | 6.0 | 41.5 | 1.5 |
| ACE | 65.5 | 6.5 | 40.0 | 2.5 |

## Table 3: Token Usage, Latency, and Cost

| Metric | Baseline | ACE | Delta |
|---|---|---|---|
| Avg Tokens/Task | 11,064 | 11,266 | +1.8% |
| Avg Prompt Tokens | 8,759 | 8,916 | +1.8% |
| Avg Completion Tokens | 2,305 | 2,350 | +2.0% |
| Total Tokens | 2,212,731 | 2,253,290 | +1.8% |
| Avg Latency (ms) | 39,817 | 33,736 | -15.3% |
| p50 Latency (ms) | 23,830 | 21,408 | -10.2% |
| p95 Latency (ms) | 101,447 | 95,417 | -5.9% |
| Estimated Cost ($) | $6.80 | $6.93 | +1.9% |

## Table 4: Per-Category Token Usage and Latency

| Category | Baseline Avg Tokens | ACE Avg Tokens | Baseline Avg Latency (ms) | ACE Avg Latency (ms) |
|---|---|---|---|---|
| DKR | 11,199 | 11,288 | 34,719 | 30,771 |
| RSA | 8,123 | 8,259 | 20,573 | 25,841 |
| PTE | 12,546 | 13,093 | 48,484 | 49,030 |
| EDS | 27,922 | 27,738 | 242,995 | 37,502 |

## Table 5: V4 Diagnostics

| Metric | Baseline | ACE | Delta |
|---|---|---|---|
| Carryover Coverage (%) | 0.0 | 0.0 | N/A |
| Learned Retrieval Rate | 0.000 | 0.000 | N/A |
| Capped Output Rate (%) | 1.0 | 1.5 | +50.0% |
| Mean Step Score | 0.000 | 0.622 | N/A |
| Quality Gate Apply Rate (%) | 0.0 | 18.0 | N/A |