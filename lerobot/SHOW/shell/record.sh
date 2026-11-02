#!/usr/bin/env bash
# Record one named B601 action with the Star Arm 102 leader.
set -euo pipefail

SHOW_ROOT="${SHOW_ROOT:-/home/ubuntu/rebot_lerobot/SHOW}"
LEROBOT_BIN="${LEROBOT_BIN:-/home/ubuntu/miniforge3/envs/lerobot/bin}"
DATA_ROOT="$SHOW_ROOT/data"

usage() {
    cat <<'EOF'
Usage:
  bash record.sh ACTION_NAME [TASK_DESCRIPTION] [EPISODES] [--resume]

Examples:
  bash record.sh pick_block "Pick up the left block" 1
  bash record.sh pick_block "Pick up the left block" 1 --resume

During recording: Right arrow = next episode, Left arrow = rerecord, ESC = stop.
EOF
}

if [[ $# -lt 1 || $# -gt 4 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

ACTION="$1"
TASK="${2:-$1}"
EPISODES="${3:-1}"
RESUME=false
if [[ "${4:-}" == "--resume" ]]; then
    RESUME=true
elif [[ $# -eq 4 ]]; then
    usage >&2
    exit 2
fi

if [[ ! "$ACTION" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "ACTION_NAME may contain only letters, digits, underscore and hyphen." >&2
    exit 2
fi
if [[ ! "$EPISODES" =~ ^[1-9][0-9]*$ ]]; then
    echo "EPISODES must be a positive integer." >&2
    exit 2
fi
if pgrep -x reBotArmControl >/dev/null || pgrep -f '^/usr/bin/python3 /opt/ros/humble/bin/ros2 launch rebotarm_bringup' >/dev/null; then
    echo "ROS still controls CAN. Stop it before recording." >&2
    exit 1
fi
if pgrep -f 'board_server.py.*reflex-arm|^/home/ubuntu/reflex-arm/.venv-leader/bin/python board_server.py' >/dev/null; then
    echo "The old leader_web service still owns the leader serial port." >&2
    exit 1
fi
if [[ ! -e /dev/ttyUSB0 ]]; then
    echo "Leader not found at /dev/ttyUSB0." >&2
    exit 1
fi
if ! ip -brief link show can0 2>/dev/null | grep -q 'UP'; then
    echo "can0 is not UP at 1 Mbps." >&2
    exit 1
fi

DATASET_DIR="$DATA_ROOT/$ACTION"
mkdir -p "$DATA_ROOT"
if [[ -e "$DATASET_DIR" && "$RESUME" == false ]]; then
    echo "Dataset already exists: $DATASET_DIR" >&2
    echo "Use --resume to append episodes or choose another ACTION_NAME." >&2
    exit 1
fi
if [[ ! -e "$DATASET_DIR" && "$RESUME" == true ]]; then
    echo "Cannot resume a missing dataset: $DATASET_DIR" >&2
    exit 1
fi

printf 'Action: %s\nEpisodes: %s\nDataset: %s\nResume: %s\n' \
    "$ACTION" "$EPISODES" "$DATASET_DIR" "$RESUME"
printf '%s\n' 'Keep the workspace clear. Move the leader gently; press ESC to stop.'

exec "$LEROBOT_BIN/lerobot-record" \
    --robot.type=seeed_b601_rs_follower \
    --robot.port=can0 \
    --robot.id=follower1 \
    --robot.can_adapter=socketcan \
    --teleop.type=rebot_arm_102_leader \
    --teleop.port=/dev/ttyUSB0 \
    --teleop.id=rebot_arm_102_leader \
    --display_data=false \
    --play_sounds=false \
    --dataset.repo_id="local/$ACTION" \
    --dataset.root="$DATASET_DIR" \
    --dataset.single_task="$TASK" \
    --dataset.num_episodes="$EPISODES" \
    --dataset.episode_time_s="${EPISODE_TIME_S:-30}" \
    --dataset.reset_time_s="${RESET_TIME_S:-10}" \
    --dataset.fps=30 \
    --dataset.video=false \
    --dataset.push_to_hub=false \
    --resume="$RESUME"
