# LeRobot + FastWAM split runtime

This tree contains an un-deployed prototype for a 395 policy host and a KR260 robot executor. No service has been installed or started on either machine. The KR260 service defaults to disarmed, and wrist-roll is inhibited because its RobStride velocity telemetry was anomalous during previous checks.

## Current deployment gate

395 HTTPS egress to `github.com` times out. FastWAM source, checkpoint, and gRPC wheels have not been fetched. The policy class is only an adapter interface; it does not load FastWAM. No local-network download fallback is allowed.

The installed LeRobot source on KR260 is pinned to `8c920c4270460851cedd2737657584586d3dc66f`; motorbridge is 0.5.6 and motorbridge-smart-servo is 0.0.4. KR260 lacks `grpcio`; the default Python and the existing 395 `b601` virtual environment lack PyTorch. Existing 395 GPU containers are still running and must be preserved. Both systems report `NTPSynchronized=no`; observed clocks differ by about 7-8 minutes, so RPC expiry and state freshness checks will fail until time is synchronized. No time configuration has been changed.

## Build generated gRPC bindings on 395

After downloading the pinned `grpcio-tools` and dependencies through 395:

```bash
python -m grpc_tools.protoc -I proto --python_out=kr260 --grpc_python_out=kr260 proto/robot_service.proto
python -m grpc_tools.protoc -I proto --python_out=host --grpc_python_out=host proto/robot_service.proto
```

After verifying server egress and clocks, install `kr260/requirements.txt` into `/home/ubuntu/b601-official-venv` and install this tree on KR260. Install `host/requirements.txt` in a separate ROCm-compatible environment, not in an existing model container. Neither installation has been performed.

## Network and service

Bind the KR260 service to `172.21.16.91:50051`; restrict TCP/50051 to source `172.21.16.96` with the host firewall. Require synchronized host clocks (chrony) because command expiry uses Unix nanoseconds. Configure the robot service under systemd with a dedicated device-access group, `CAP_NET_RAW`, `NoNewPrivileges=true`, and writable access only to its log/calibration directories.

The default `KRIA_ALLOWED_JOINTS` is empty. Never set it to all joints until the velocity anomaly is explained and the staged tests pass. The server also rejects callers other than `172.21.16.96`. Do not change torque clamps to make a failed motion test pass.

## Tests

From this directory, run contract tests on any Python 3.10+ machine:

```bash
python -m unittest discover -s tests -v
```

Then test the gRPC server/client with an injected mock hardware backend; perform hardware tests in order: read-only state, one joint hold, one joint small move, gripper, paired joints, then full arm. FastWAM is offline-only until its exact input/output and ROCm support are verified from upstream. The current executor is not yet qualified for hardware motion: its motor command interface and coordinate conversion still require comparison against the installed driver source and a mock or bench test.

The Gemini 305 publishes ten V4L2 nodes on this host. Identify the color stream with `v4l2-ctl --list-formats-ext -d /dev/videoN` and configure its explicit node; do not guess among the depth/IR interfaces. The Python camera wrapper is a capture primitive, not yet a LeRobot camera plugin.
