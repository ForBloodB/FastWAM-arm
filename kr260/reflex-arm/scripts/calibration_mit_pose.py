"""Bounded single-joint MIT pose acquisition for J1/J5/J6.

One fixed anchor, one excursion, a settled hold, one return, then disable and
restore volatile parameters. No zero/save/firmware operations and no writes to
other joints. Disabled motors need not provide periodic status before enable;
fresh target status is mandatory immediately after enable and throughout.
"""
import argparse
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import signal
import struct
import time
from reflex.canbus import ReadOnlyCAN
from reflex.motion import VelocityGuard, can_timeout_raw

p = argparse.ArgumentParser()
p.add_argument('--joint', type=int, choices=(1, 5, 6), required=True)
p.add_argument('--amplitude', type=float, choices=(2., 5., 10.), required=True)
p.add_argument('--direction', type=int, choices=(-1, 1), required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--execute-authorized-test', action='store_true', required=True)
a = p.parse_args()
mid = a.joint
amplitude = a.amplitude
duration = max(40., (8. if mid == 1 else 16.) * amplitude)
peak_hold = 20.
a.out.mkdir(parents=True, exist_ok=False)
lock_fd = os.open('/tmp/reflex-can-actuation.lock', os.O_RDWR | os.O_NOFOLLOW)
lock = os.fdopen(lock_fd, 'r+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
c = ReadOnlyCAN()
c.capture = []
tx = []
samples = []
peak_samples = []
original = {}
changed = []
result = {'joint': mid, 'amplitude_deg': amplitude, 'direction': a.direction,
          'duration_s': duration, 'peak_hold_s': peak_hold,
          'success': False, 'completed_profile': False,
          'no_other_joint_writes': True, 'parameter_persistence_writes': 0,
          'physical_grasp': False, 'camera_extrinsic_calibrated': False}
old_gc = gc.isenabled()

def interrupted(*_):
    raise KeyboardInterrupt('Operator/host interruption')
signal.signal(signal.SIGINT, interrupted)
signal.signal(signal.SIGTERM, interrupted)

def command(kind, data=b'\0' * 8):
    if kind not in (3, 4, 18):
        raise ValueError('Unapproved command')
    frame = struct.pack('=IB3x8s', 0x80000000 | (kind << 24) | (0xFD << 8) | mid, 8, data)
    c.sock.send(frame)
    tx.append({'ns': time.monotonic_ns(), 'kind': kind, 'joint': mid, 'frame': frame.hex()})

def write(param, value):
    kinds = {0x700B: 'f', 0x7028: 'I'}
    if param not in kinds:
        raise ValueError('Parameter not whitelisted')
    command(18, struct.pack('<H', param) + b'\0\0' + struct.pack('<' + kinds[param], value).ljust(4, b'\0'))
    actual = c.read(mid, param, .03)
    if not math.isclose(actual, value, rel_tol=1e-5, abs_tol=1e-6):
        raise RuntimeError('Volatile parameter readback mismatch')

def send_mit(position):
    if mid == 1:
        kp, kd, kpmax, kdmax = 200., 3., 5000., 100.
    else:
        kp, kd, kpmax, kdmax = 30., 1., 500., 5.
    enc = lambda value, lo, hi: int((value - lo) / (hi - lo) * 65535)
    payload = struct.pack('>4H', enc(position, -4 * math.pi, 4 * math.pi), 32767,
                          enc(kp, 0., kpmax), enc(kd, 0., kdmax))
    frame = struct.pack('=IB3x8s', 0x80000000 | (1 << 24) | (32767 << 8) | mid, 8, payload)
    c.sock.send(frame)
    tx.append({'ns': time.monotonic_ns(), 'kind': 1, 'joint': mid, 'frame': frame.hex()})

def status():
    c.drain()
    s = c.latest_status.get(mid)
    if not s or time.monotonic_ns() - s['rx_ns'] > 80_000_000:
        raise RuntimeError('Target status stale')
    if s['fault_mask_raw'] or s['temperature_C'] >= 45:
        raise RuntimeError('Target fault or temperature limit')
    return s

try:
    c.poll(.1)
    baseline = [c.read(j, 0x7019, .05) for j in range(1, 8)]
    zero = {j: c.read(j, 0x7029, .05) for j in range(1, 8)}
    volts = {j: c.read(j, 0x701C, .05) for j in range(1, 8)}
    modes = {j: c.read(j, 0x7005, .05) for j in range(1, 8)}
    if any(v != 1 for v in zero.values()) or any(not 42 < v < 54 for v in volts.values()):
        raise RuntimeError('Zero-state or supply preflight failed')
    if any(v != 0 for v in modes.values()):
        raise RuntimeError('All joints must retain MIT mode')
    anchor = baseline[mid - 1]
    limit = 2.7 if mid == 1 else (1.57 if mid == 5 else 3.14)
    target_peak = anchor + a.direction * math.radians(amplitude)
    if abs(anchor) > .05 or not -limit <= target_peak <= limit:
        raise RuntimeError('Starting pose or target outside reviewed interval')
    result.update(baseline_all_motor_rad=baseline, anchor_motor_rad=anchor,
                  target_peak_rad=target_peak, preflight_zero=zero,
                  preflight_voltages_V=volts, preflight_mode_parameters=modes,
                  initial_periodic_status_required=False)
    for param in (0x700B, 0x7028):
        original[param] = c.read(mid, param, .03)
    result['original_parameters'] = {hex(k): v for k, v in original.items()}
    command(4)
    c.poll(.05)
    settled = [c.read(mid, 0x7019, .03) for _ in range(12)]
    if max(settled) - min(settled) > math.radians(.06):
        raise RuntimeError('Target not stationary before enable')
    anchor = sorted(settled)[len(settled) // 2]
    target_peak = anchor + a.direction * math.radians(amplitude)
    result['anchor_motor_rad'] = anchor
    result['target_peak_rad'] = target_peak
    for param, value in ((0x700B, .5), (0x7028, can_timeout_raw(100))):
        changed.append(param)
        write(param, value)
    result.update(torque_limit_Nm=.5, watchdog_raw=2000,
                  mit_kp=200. if mid == 1 else 30., mit_kd=3. if mid == 1 else 1.)
    guard = VelocityGuard(limit_deg_s=3., raw_limit_deg_s=3. if mid == 1 else 15.)
    for _ in range(30):
        q = c.read(mid, 0x7019, .02)
        v = c.read(mid, 0x701B, .02)
        guard.update(time.monotonic_ns(), q, v)
        time.sleep(.008)
    send_mit(anchor)
    gc.collect()
    gc.disable()
    command(3)
    send_mit(anchor)
    deadline = time.monotonic() + 1.
    while True:
        st = status()
        if st['mode_bits_raw'] == 2:
            result['enabled_status'] = dict(st)
            break
        if time.monotonic() > deadline:
            raise RuntimeError('Target did not enter Run mode')
    start = time.monotonic()
    previous = start
    announced = False
    total = duration + peak_hold + 3.
    for i in range(round(total * 100) + 1):
        time.sleep(max(0, start + i * .01 - time.monotonic()))
        now = time.monotonic()
        elapsed = now - start
        gap = now - previous
        previous = now
        if gap > .05:
            raise RuntimeError('Control freshness gap >50 ms')
        move_elapsed = elapsed
        if move_elapsed < duration / 2:
            fraction = move_elapsed / duration
        elif move_elapsed < duration / 2 + peak_hold:
            fraction = .5
        else:
            fraction = (move_elapsed - peak_hold) / duration
        fraction = min(max(fraction, 0.), 1.)
        target = anchor + a.direction * math.radians(amplitude) * .5 * (1 - math.cos(2 * math.pi * fraction))
        send_mit(target)
        q = c.read(mid, 0x7019, .02)
        v = c.read(mid, 0x701B, .02)
        voltage = c.read(mid, 0x701C, .02)
        iq = c.read(mid, 0x701A, .02)
        ns = time.monotonic_ns()
        st = dict(status())
        row = {'ns': ns, 'elapsed_s': elapsed, 'move_elapsed_s': move_elapsed,
               'command_rad': target, 'actual_rad': q, 'raw_velocity_rad_s': v,
               'voltage_V': voltage, 'iq_A': iq, 'gap_s': gap, 'status': st}
        row['velocity_guard'] = guard.update(ns, q, v)
        samples.append(row)
        if duration / 2 + 8 <= move_elapsed < duration / 2 + peak_hold:
            peak_samples.append(row)
        if not 42 <= voltage <= 54 or abs(iq) > (2. if mid == 1 else .75):
            raise RuntimeError('Supply or current outside envelope')
        envelope = amplitude + .6
        if abs(math.degrees(q - anchor)) > envelope or abs(math.degrees(q - target)) > envelope:
            raise RuntimeError('Position outside bounded test envelope')
        if elapsed > .1 and st['mode_bits_raw'] != 2:
            raise RuntimeError('Target left Run mode')
        if i % 10 == 0:
            neighbours = {j: c.read(j, 0x7019, .02) for j in range(1, 8) if j != mid}
            row['neighbours_rad'] = neighbours
            if any(abs(qj - baseline[j - 1]) > math.radians(.3) for j, qj in neighbours.items()):
                raise RuntimeError('Uncommanded joint displacement >0.3 deg')
        if not announced and move_elapsed >= duration / 2 + 8:
            event = {'event': 'PEAK_HOLD_READY', 'joint': mid, 'ns': ns,
                     'target_rad': target, 'remaining_hold_s': duration / 2 + peak_hold - move_elapsed}
            result['peak_capture_event'] = event
            print(json.dumps(event), flush=True)
            announced = True
    command(4)
    c.poll(.08)
    if status()['mode_bits_raw'] != 0:
        raise RuntimeError('Target failed to disable')
    result['completed_profile'] = True
    peak = a.direction * math.degrees(max((a.direction * (s['actual_rad'] - anchor) for s in samples), key=abs))
    final = math.degrees(samples[-1]['actual_rad'] - anchor)
    qs = [s['actual_rad'] for s in peak_samples]
    result.update(actual_amplitude_motor_deg=peak, final_error_motor_deg=final,
                  tracking_rmse_motor_deg=math.sqrt(sum(math.degrees(s['actual_rad'] - s['command_rad']) ** 2 for s in samples) / len(samples)),
                  peak_abs_current_A=max(abs(s['iq_A']) for s in samples),
                  maximum_feedback_gap_ms=max(s['gap_s'] for s in samples) * 1000,
                  peak_window={'samples': len(peak_samples), 'q_mean_rad': sum(qs) / len(qs),
                               'q_range_rad': max(qs) - min(qs),
                               'first_ns': peak_samples[0]['ns'], 'last_ns': peak_samples[-1]['ns']})
    result['success'] = (amplitude * .8 <= peak <= amplitude * 1.2 and abs(final) <= .15 and
                         len(peak_samples) >= 600 and result['peak_window']['q_range_rad'] < math.radians(.06))
except BaseException as exc:
    result['error'] = repr(exc)
finally:
    try:
        command(4)
        c.poll(.08)
        result['target_deliberately_left_disabled'] = status()['mode_bits_raw'] == 0
        for param in (0x7028, 0x700B):
            if param in changed:
                write(param, int(original[param]) if param == 0x7028 else original[param])
        result['parameters_restored'] = True
        result['restored_parameters'] = {hex(param): c.read(mid, param, .03) for param in changed}
    except BaseException as exc:
        result['restoration_error'] = repr(exc)
        try:
            command(4)
        except BaseException:
            pass
    if old_gc:
        gc.enable()
    result['samples'] = len(samples)
    result['commands_sent_by_test'] = len(tx)
    (a.out / 'samples.json').write_text(json.dumps(samples))
    (a.out / 'peak_samples.json').write_text(json.dumps(peak_samples))
    (a.out / 'tx_frames.json').write_text(json.dumps(tx))
    (a.out / 'rx_frames.json').write_text(json.dumps(c.capture))
    (a.out / 'summary.json').write_text(json.dumps(result, indent=2))
    c.close()
    lock.close()
    print(json.dumps(result), flush=True)
raise SystemExit(0 if result['success'] and result.get('parameters_restored') and result.get('target_deliberately_left_disabled') else 1)
