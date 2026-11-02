#!/usr/bin/env bash
# Run from the project root in Git Bash. Only J1 is enabled; all other joints
# must remain disabled. The test script restores its temporary parameters.
set -u
ROOT=$(pwd)
ASKPASS="$ROOT/ssh_askpass.sh"
if [[ ! -f "$ASKPASS" || ! -f "$ROOT/机器密码.txt" ]]; then
  echo "Missing SSH credential helper in project root" >&2
  exit 2
fi
RUN_ID=${1:-j1_visual_$(date +%Y%m%d_%H%M%S)}
if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "Run ID may contain only letters, digits, underscore and hyphen" >&2
  exit 2
fi
LOCAL="$ROOT/evidence_2026-10-28/$RUN_ID"
REMOTE_395="/home/aup/$RUN_ID"
REMOTE_KR="/home/ubuntu/reflex-arm/runs/$RUN_ID"
mkdir -p "$LOCAL"
ssh395() {
  env SSH_ASKPASS="$ASKPASS" SSH_ASKPASS_REQUIRE=force DISPLAY=kimi MACHINE_KEY=395 \
    ssh -o NumberOfPasswordPrompts=1 -o ConnectTimeout=10 aup@172.21.16.96 "$@"
}
scp395() {
  env SSH_ASKPASS="$ASKPASS" SSH_ASKPASS_REQUIRE=force DISPLAY=kimi MACHINE_KEY=395 \
    scp -o NumberOfPasswordPrompts=1 "$@"
}
sshkr() {
  env SSH_ASKPASS="$ASKPASS" SSH_ASKPASS_REQUIRE=force DISPLAY=kimi MACHINE_KEY=krs \
    ssh -o NumberOfPasswordPrompts=1 -o ConnectTimeout=10 ubuntu@172.21.16.34 "$@"
}
scpkr() {
  env SSH_ASKPASS="$ASKPASS" SSH_ASKPASS_REQUIRE=force DISPLAY=kimi MACHINE_KEY=krs \
    scp -o NumberOfPasswordPrompts=1 "$@"
}
# Refuse to reuse a previous result directory.
if sshkr "test -e '$REMOTE_KR'" || ssh395 "test -e '$REMOTE_395'"; then
  echo "Run ID already exists; choose a new one" >&2
  exit 2
fi
ssh395 "mkdir -p '$REMOTE_395/global' '$REMOTE_395/wrist'; \
  python3 /home/aup/record_demo_camera.py --device /dev/v4l/by-id/usb-Generic_USB_Camera_202001010001-video-index0 --out '$REMOTE_395/global' --seconds 18 & p1=\$!; \
  python3 /home/aup/record_demo_camera.py --device /dev/v4l/by-id/usb-Orbbec_R__Orbbec_Gemini_305_CV278640000A-video-index4 --out '$REMOTE_395/wrist' --seconds 18 & p2=\$!; \
  wait \$p1; s1=\$?; wait \$p2; s2=\$?; test \$s1 -eq 0 -a \$s2 -eq 0" \
  >"$LOCAL/cameras.log" 2>&1 &
CAMERA_PID=$!
sleep 3
if ! ssh395 "test -f '$REMOTE_395/global/ready' -a -f '$REMOTE_395/wrist/ready'"; then
  echo "Cameras did not become ready; motion was not started" >&2
  wait "$CAMERA_PID"
  exit 1
fi
MACHINE_KEY=krs "$ASKPASS" | sshkr "sudo -S -p '' env PYTHONPATH=/home/ubuntu/reflex-arm \
  python3 /home/ubuntu/reflex-arm/scripts/repair_hardware_live_j1.py move-j1 \
  --out '$REMOTE_KR' --execute-authorized-test" | tee "$LOCAL/j1_test.log"
TEST_STATUS=${PIPESTATUS[0]}
wait "$CAMERA_PID"
CAMERA_STATUS=$?
scpkr -r "ubuntu@172.21.16.34:$REMOTE_KR" "$LOCAL/kr"
scp395 -r "aup@172.21.16.96:$REMOTE_395/global" "aup@172.21.16.96:$REMOTE_395/wrist" "$LOCAL/"
echo "Evidence: $LOCAL"
echo "Motion status=$TEST_STATUS camera status=$CAMERA_STATUS"
if [[ $TEST_STATUS -ne 0 || $CAMERA_STATUS -ne 0 ]]; then
  echo "Do not present this run as passed; inspect summary.json and logs" >&2
  exit 1
fi
