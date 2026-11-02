"""Offline evidence audit of a wrist guard stop; never opens a device."""
import argparse
import bisect
import json
import math
from pathlib import Path
import struct

p = argparse.ArgumentParser()
p.add_argument('--case', type=Path, required=True)
a = p.parse_args()
summary = json.loads((a.case / 'summary.json').read_text())
rows = json.loads((a.case / 'samples.json').read_text())
event = summary.get('guard_stop_evidence')
if not event:
    p.error('This audit requires a saved guard event')
times = [row['ns'] for row in rows]
anchor = summary['anchor_motor_rad']
selected = [row for row in rows if event['from_ns'] - 120_000_000 <= row['ns'] <= event['to_ns']]
details = []
for row in selected:
    details.append({key: value for key, value in {
        'relative_event_ms': (row['ns'] - event['to_ns']) / 1e6,
        'elapsed_s': row['elapsed_s'],
        'position_deg_from_anchor': math.degrees(row['actual_rad'] - anchor),
        'command_deg_from_anchor': math.degrees(row['command_rad'] - anchor),
        'raw_velocity_deg_s': math.degrees(row['raw_velocity_rad_s']),
        'position_reply_ns': row.get('position_reply_ns'),
        'status_position_u16': row['status']['position_u16'],
        'status_rx_ns': row['status']['rx_ns'],
        'iq_A': row['iq_A'], 'estimated_MIT_torque_Nm': row.get('estimated_MIT_torque_Nm'),
    }.items()})
first = next(row for row in rows if row['ns'] == event['from_ns'])
last = next(row for row in rows if row['ns'] == event['to_ns'])
reply_dt = (last['position_reply_ns'] - first['position_reply_ns']) / 1e9
slopes = []
for ms in (20, 50, 100, 200, 500):
    index = bisect.bisect_right(times, last['ns'] - ms * 1_000_000) - 1
    if index >= 0:
        old = rows[index]
        dt = (last['position_reply_ns'] - old['position_reply_ns']) / 1e9
        slopes.append({'minimum_window_ms': ms, 'actual_reply_window_ms': dt * 1000,
                       'position_speed_deg_s': math.degrees(last['actual_rad'] - old['actual_rad']) / dt})
raw_status = []
foreign_control = []
for item in json.loads((a.case / 'rx_frames.json').read_text()):
    ns = item['rx_ns']
    canid, dlc, data = struct.unpack('=IB3x8s', bytes.fromhex(item['frame_hex']))
    eid = canid & 0x1fffffff
    kind = (eid >> 24) & 31
    if event['from_ns'] - 30_000_000 <= ns <= event['to_ns'] + 10_000_000 and kind in (2, 24) and (eid >> 8) & 255 == summary['joint']:
        raw_status.append({'rx_ns': ns, 'type': kind, 'position_u16': struct.unpack('>4H', data)[0], 'frame_hex': item['frame_hex']})
    if kind in (1, 3, 4, 18) and ns < event['to_ns']:
        # SocketCAN does not loop this socket's own TX back to itself.
        foreign_control.append({'rx_ns': ns, 'kind': kind, 'frame_hex': item['frame_hex']})
result = {'guard_event': event, 'same_KR_clock_reply_dt_ms': reply_dt * 1000,
          'speed_recomputed_with_position_reply_times_deg_s': math.degrees(last['actual_rad'] - first['actual_rad']) / reply_dt,
          'position_change_deg': math.degrees(last['actual_rad'] - first['actual_rad']),
          'event_window_rows': details, 'ending_window_slopes': slopes,
          'packed_status_frames_near_event': raw_status,
          'other_control_frames_received_before_stop': foreign_control,
          'maximum_loop_gap_ms': max(row['gap_s'] for row in rows) * 1000,
          'peak_current_A': max(abs(row['iq_A']) for row in rows),
          'peak_estimated_MIT_effort_Nm': max(abs(row.get('estimated_MIT_torque_Nm', 0)) for row in rows),
          'physical_cause_proven': False, 'device_fault_seen': any(row['status']['fault_mask_raw'] for row in rows),
          'hardware_commands_sent_by_this_audit': 0,
          'interpretation_limit': '7019 and packed status originate in the same motor encoder; video is not a 20 ms metrology source.'}
print(json.dumps(result, indent=2))
