#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m grpc_tools.protoc \
  -I proto \
  --python_out=kr260 --grpc_python_out=kr260 \
  proto/robot_service.proto
python -m grpc_tools.protoc \
  -I proto \
  --python_out=host --grpc_python_out=host \
  proto/robot_service.proto
