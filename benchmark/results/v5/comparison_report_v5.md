# CL-bench Evaluation: Baseline vs ACE Comparison (V5)

## Table 1: Solving Rate by Category

| Model Names | Overall (%) | Domain Knowledge Reasoning n=85 (%) | Rule System Application n=62 (%) | Procedural Task Execution n=47 (%) | Empirical Discovery & Simulation n=6 (%) |
|---|---|---|---|---|---|
| GPT-5.1 (High) baseline | 19.5 | 17.6 | 25.8 | 14.9 | 16.7 |
| GPT-5.1 (High) + ACE | 23.0 | 14.1 | 33.9 | 25.5 | 16.7 |
| Delta | +17.9% | -20.0% | +31.2% | +71.4% | +0.0% |

## Table 2: Error Analysis Distribution

| Model Names | Context Ignored (%) | Context Misused (%) | Format Error (%) | Refusal (%) |
|---|---|---|---|---|
| Baseline | 67.0 | 7.0 | 40.5 | 1.5 |
| ACE | 63.0 | 4.5 | 40.5 | 1.5 |

## Table 3: Token Usage, Latency, and Cost

| Metric | Baseline | ACE | Delta |
|---|---|---|---|
| Avg Tokens/Task | 11,045 | 44,516 | +303.1% |
| Avg Prompt Tokens | 8,711 | 35,530 | +307.9% |
| Avg Completion Tokens | 2,333 | 8,985 | +285.1% |
| Total Tokens | 2,208,906.0 | 8,903,147.0 | +303.1% |
| Avg Latency (ms) | 36,735 | 130,008 | +253.9% |
| p50 Latency (ms) | 28,328 | 74,550 | +163.2% |
| p95 Latency (ms) | 96,838 | 480,594 | +396.3% |
| Estimated Cost ($) | $6.84 | $26.85 | +292.4% |

## Table 4: Per-Category Token Usage and Latency

| Category | Baseline Avg Tokens | ACE Avg Tokens | Baseline Avg Latency (ms) | ACE Avg Latency (ms) |
|---|---|---|---|---|
| DKR | 11,236 | 45,729 | 39,873 | 136,826 |
| RSA | 8,268 | 34,945 | 23,027 | 85,145 |
| PTE | 12,216 | 52,085 | 47,907 | 183,297 |
| EDS | 27,847 | 66,941 | 46,428 | 79,574 |

## Table 5: Runtime Diagnostics

| Metric | Baseline | ACE | Delta |
|---|---|---|---|
| Carryover Coverage (%) | 0.0 | 0.0 | N/A |
| Learned Retrieval Rate | 0.000 | 0.000 | N/A |
| Capped Output Rate (%) | 0.0 | 0.0 | N/A |
| Mean Step Score | 0.000 | 0.808 | N/A |
| Quality Gate Apply Rate (%) | 0.0 | 0.0 | N/A |
| Resume Recovery Rate (%) | 0.0 | 100.0 | N/A |
| Memory Write Failure Rate (%) | 0.0 | 0.0 | N/A |
| Progress Checkpoint Count | 0 | 200 | N/A |

## Table 5B: Planner Policy Diagnostics

| Metric | Baseline | ACE | Delta |
|---|---|---|---|
| Dominant Action | N/A (0.0%) | deep_refine (30.0%) | N/A |
| Action Distribution (%) | N/A | deep_refine:30.0%, direct:19.0%, explore:22.0%, refine:29.0% | N/A |
| Explore Rate (%) | 0.0 | 9.5 | N/A |
| Recursion Success Rate (%) | 0.0 | 29.0 | N/A |
| Mean Reward Proxy | 0.000 | 0.673 | N/A |
| Policy Update Count | 0 | 200 | N/A |
| Policy Update Rate (%) | 0.0 | 100.0 | N/A |

## Table 6: Full Pipeline Actual Metered Cost

| Phase | Prompt Tokens | Completion Tokens | Total Tokens | Actual Metered Cost ($) |
|---|---|---|---|---|
| Inference (Baseline) | 1,742,275 | 466,631 | 2,208,906 | $6.84 |
| Inference (ACE Primary) | 7,106,086 | 1,797,061 | 8,903,147 | $26.85 |
| ACE Auxiliary (Reflector/Step) | 86,460,145 | 1,471,621 | 87,931,766 | $122.79 |
| Evaluation (Baseline) | 664,536 | 445,322 | 1,109,858 | $5.28 |
| Evaluation (ACE) | 685,494 | 428,932 | 1,114,426 | $5.15 |
| Error Analysis (Baseline) | 284,187 | 86,472 | 370,659 | $1.22 |
| Error Analysis (ACE) | 272,213 | 84,295 | 356,508 | $1.18 |
| Baseline Total | 2,690,998 | 998,425 | 3,689,423 | $13.35 |
| ACE Total | 94,523,938 | 3,781,909 | 98,305,847 | $155.97 |
| Combined Total | 97,214,936 | 4,780,334 | 101,995,270 | $169.32 |

## Table 7: OpenAI Billed Reconciliation

| Item | Value |
|---|---|
| Cost Mode | dual_source |
| Billing Policy | off |
| Reconciliation Status | disabled_policy_off |
| Project Scope | N/A |
| Run Window Start (UTC) | N/A |
| Run Window End (UTC) | N/A |
| Metered Cost ($) | $169.32 |
| Billed Cost ($) | N/A |
| Reconciliation Delta ($) | N/A |
| Reconciliation Delta (%) | N/A |
| Notes | Billing reconciliation disabled by policy. |