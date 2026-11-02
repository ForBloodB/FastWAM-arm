#!/usr/bin/env bash
# Replay a named B601 action, selected by an action name or a sentence keyword.
set -euo pipefail

SHOW_ROOT="${SHOW_ROOT:-/home/ubuntu/rebot_lerobot/SHOW}"
LEROBOT_BIN="${LEROBOT_BIN:-/home/ubuntu/miniforge3/envs/lerobot/bin}"
DATA_ROOT="$SHOW_ROOT/data"

usage() {
    cat <<'EOF'
Usage:
  bash replay.sh ACTION_NAME_OR_SENTENCE [EPISODE]

Examples:
  bash replay.sh pick_block 0
  bash replay.sh "抓起左边方块" 0
  bash replay.sh "返回初始位置" 0

Built-in sentence mapping:
  pick/grab/抓取/抓起 -> pick_block
  place/放置/放下     -> place_block
  home/归位/初始位置  -> home

Add new case patterns below when adding new action datasets.
EOF
}

if [[ $# -lt 1 || $# -gt 2 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

QUERY="$1"
EPISODE="${2:-0}"
if [[ ! "$EPISODE" =~ ^[0-9]+$ ]]; then
    echo "EPISODE must be a non-negative integer." >&2
    exit 2
fi

if [[ "$QUERY" =~ ^[A-Za-z0-9_-]+$ && -d "$DATA_ROOT/$QUERY" ]]; then
    ACTION="$QUERY"
else
    QUERY_LOWER="${QUERY,,}"
    case "$QUERY_LOWER" in
        *pick*|*grab*|*抓取*|*抓起*|*抓*)
            ACTION=pick_block
            ;;
        *place*|*放置*|*放下*|*放*)
            ACTION=place_block
            ;;
        *home*|*归位*|*初始位置*|*返回初始*)
            ACTION=home
            ;;
        *)
            echo "No action matches: $QUERY" >&2
            echo "Available actions:" >&2
            for dir in "$DATA_ROOT"/*; do
                [[ -d "$dir" ]] && printf '  %s\n' "${dir##*/}" >&2
            done
            exit 2
            ;;
    esac
fi

DATASET_DIR="$DATA_ROOT/$ACTION"
if [[ ! -f "$DATASET_DIR/meta/info.json" ]]; then
    echo "No valid dataset for action '$ACTION': $DATASET_DIR" >&2
    exit 1
fi
if pgrep -x reBotArmControl >/dev/null || pgrep -f '^/usr/bin/python3 /opt/ros/humble/bin/ros2 launch rebotarm_bringup' >/dev/null; then
    echo "ROS still controls CAN. Stop it before replay." >&2
    exit 1
fi
if ! ip -brief link show can0 2>/dev/null | grep -q 'UP'; then
    echo "can0 is not UP at 1 Mbps." >&2
    exit 1
fi

printf 'Matched action: %s\nEpisode: %s\nDataset: %s\n' "$ACTION" "$EPISODE" "$DATASET_DIR"
printf '%s\n' 'WARNING: the follower will move without needing leader motion.'
if [[ "${REPLAY_CONFIRM:-}" != "YES" ]]; then
    read -r -p 'Type YES to confirm that the workspace is clear: ' CONFIRM
    if [[ "$CONFIRM" != "YES" ]]; then
        echo "Replay cancelled."
        exit 1
    fi
fi

exec "$LEROBOT_BIN/lerobot-replay" \
    --robot.type=seeed_b601_rs_follower \
    --robot.port=can0 \
    --robot.id=follower1 \
    --robot.can_adapter=socketcan \
    --play_sounds=false \
    --dataset.repo_id="local/$ACTION" \
    --dataset.root="$DATASET_DIR" \
    --dataset.episode="$EPISODE" \
    --dataset.fps=30
