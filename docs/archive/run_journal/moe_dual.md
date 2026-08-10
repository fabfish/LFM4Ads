# Run journal -- moe_dual

Append-only log written by `.codebuddy/skills/long-run-watch`.
One bullet per checkpoint: observation + interpretation, newest at the bottom.

- **2026-08-09 21:09:10** batch1 done: v2_full / v1_rxonly / k1_full. KEY: k1_full (no-MoE control, K=1) also drops pooled AUC 0.7775->0.7709 and per-scenario mean 0.7173->0.7143, so 'MoE hurts' is a continued-training artifact; baseline must be k1_full not vanilla.
