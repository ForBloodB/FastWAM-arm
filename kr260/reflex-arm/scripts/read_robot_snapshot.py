"""Read-only B601 position, voltage and status snapshot; no enable or writes."""
import json
import math
from reflex.canbus import ReadOnlyCAN

c = ReadOnlyCAN(host_id=0xFC)
result = {"evidence": "real_read_only_CAN", "writes_sent": 0, "motor_enables_sent": 0}
try:
    c.poll(0.1)
    rows = []
    for joint in range(1, 8):
        row = {"joint": joint}
        for key, param in (("position_rad", 0x7019), ("voltage_V", 0x701C), ("zero_state", 0x7029), ("mode_parameter", 0x7005)):
            try:
                row[key] = c.read(joint, param, 0.15)
            except Exception as exc:
                row[key + "_error"] = repr(exc)
                break
        rows.append(row)
    c.poll(0.1)
    result.update(joints=rows, status=c.latest_status)
finally:
    c.close()
print(json.dumps(result, indent=2))
