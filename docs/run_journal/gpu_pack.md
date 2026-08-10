# Run journal -- gpu_pack

Append-only log written by `.codebuddy/skills/long-run-watch`.
One bullet per checkpoint: observation + interpretation, newest at the bottom.

- **2026-08-10 17:26:33** ticks 1-6 (16:02→17:23) 全部 job 健康无停滞。D6/backbone 矩阵 4/7 FINISH(bb_v1_s42/s123、bb_v2_s42/s123 均 rc=0); bb 预训练基线 V1=0.7739/0.7742、V2(s42)=0.7709(test_auc_all vs vanilla 0.7775)。D6 V2×3 + V2 s123 在三级下游评估, ETA~20:25。D20 rx-only: s1/s7 DONE, MoE 0.7734/0.7737 vs vanilla 0.7775, Δ≈-0.004(负迁移在 shuffle-downstream 下稳健复现, D9-A 否定结论幸存); s2024 下游评估中 ETA~20:45。D20b freeze-dnn-head 臂误冻 sparse 已修(21bb17a), rx-only 完成后自动触发。nan ALERT 为良性次要指标(test_auc_rx 在 shuffled 下游退化类分布)。gpu0 98%/8.7GB。
