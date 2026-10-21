#!/usr/bin/env bash
# Fresh stage5 experiment. Activate the isaaclab environment before running.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
checkpoint="logs/rsl_rl/deeprobotics_m20_stair_teacher_v1_stage2/2026-10-20_17-01-23_ascent_entry_v1_stage3/model_1100.pt"
shopt -s nullglob
demonstrations=(docs/test_docs/07_上楼步态校正/v1_20261021/stage5_20261021/verified_contact_demonstrations/*/verified_demonstration.npz)
if ((${#demonstrations[@]} == 0)); then
    printf '%s\n' 'Missing verified stage5 demonstrations.' >&2
    exit 1
fi
python_bin="${M20_PYTHON:-python}"
"$python_bin" scripts/tools/preflight_ascent_stage5.py \
    --checkpoint "$checkpoint" --demonstration "${demonstrations[@]}"
if [[ "${1:-}" == '--check-only' ]]; then
    exit 0
fi
exec env OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" "$python_bin" -u \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task Rough-Deeprobotics-M20-StairTeacher-v1 \
    --m20_profile ascent_entry_v1_stage5 \
    --init_actor_from "$checkpoint" \
    --init_critic_from "$checkpoint" \
    --ascent_demo "${demonstrations[@]}" \
    --run_name ascent_entry_v1_stage5_contact_curriculum \
    --num_envs "${M20_STAGE5_ENVS:-4096}" \
    --max_iterations "${M20_STAGE5_ITERS:-20000}" \
    --seed 42 --device "${M20_STAGE5_DEVICE:-cuda:0}" --headless "$@"
