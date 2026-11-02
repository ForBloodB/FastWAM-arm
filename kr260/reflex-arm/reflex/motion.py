"""Very small J1-only test profile. No learned policy can enter this path."""
import math
from collections import deque
from statistics import median


def can_timeout_raw(milliseconds):
    """RS00/RS06 manual: 20,000 counts per second (50 us per count)."""
    if not 50<=milliseconds<=250:raise ValueError('Diagnostic watchdog must be 50..250 ms')
    return round(milliseconds*20)


class VelocityGuard:
    """Keep raw evidence; reject sustained velocity or encoder-confirmed motion.

    Isolated 701B spikes were observed with disabled motors and stationary encoder.
    A five-sample median does not replace encoder motion checks. The independent
    20/50 ms displacement checks and raw extreme/sustained checks remain active.
    This is a userspace diagnostic guard, not a certified speed limiter.
    """
    def __init__(self,limit_deg_s=3.,raw_limit_deg_s=3.):
        self.limit=math.radians(limit_deg_s)
        self.raw_limit=math.radians(raw_limit_deg_s)
        if not self.limit>0 or not self.limit<=self.raw_limit<=math.radians(15):
            raise ValueError('Invalid diagnostic velocity thresholds')
        self.history=deque()
        self.raw=deque(maxlen=5)
        self.over_since=None
        self.last_time=None
        self.trip_evidence=None

    def update(self,ns,position,velocity):
        if not all(math.isfinite(v) for v in (ns,position,velocity)):
            raise ValueError('Nonfinite velocity telemetry')
        if self.last_time is not None and not 0<ns-self.last_time<=50_000_000:
            raise ValueError('Velocity telemetry stale or nonmonotonic')
        self.last_time=ns
        self.raw.append(velocity);self.history.append((ns,position))
        while self.history and ns-self.history[0][0]>120_000_000:self.history.popleft()
        if abs(velocity)>math.radians(30):raise ValueError('Extreme raw speed > 30 deg/s')
        if abs(velocity)>self.raw_limit:
            if self.over_since is None:self.over_since=ns
            if ns-self.over_since>=20_000_000:raise ValueError('Sustained raw overspeed')
        else:self.over_since=None
        filtered=median(self.raw)
        if len(self.raw)>=5 and abs(filtered)>self.raw_limit:raise ValueError('Median velocity overspeed')
        measured=[]
        for window in (20_000_000,50_000_000):
            eligible=[r for r in self.history if ns-r[0]>=window]
            if eligible:
                t,p=eligible[-1]
                slope=(position-p)/((ns-t)/1e9);measured.append(slope)
                if abs(slope)>self.limit:
                    self.trip_evidence={'source':'position_displacement','from_ns':t,'to_ns':ns,
                        'minimum_window_ms':window/1e6,'actual_window_ms':(ns-t)/1e6,
                        'position_change_deg':math.degrees(position-p),
                        'encoder_speed_deg_s':math.degrees(slope),'limit_deg_s':math.degrees(self.limit)}
                    raise ValueError('Encoder displacement confirms overspeed')
        return {'median_rad_s':filtered,'encoder_rad_s':measured,
                'isolated_raw_outlier':abs(velocity)>self.limit}


def j1_test_target(initial, elapsed, duration=6., amplitude_deg=.5):
    if not math.isfinite(initial) or not 4<=duration<=10 or not 0<amplitude_deg<=.5:
        raise ValueError('Test limited to <= 0.5 degree and >= 4 seconds')
    fraction=max(0.,min(1.,elapsed/duration))
    return initial+math.radians(amplitude_deg)*.5*(1-math.cos(2*math.pi*fraction))


def repeated_diagnostic_target(mid,initial,elapsed,repetitions):
    """The validated J1/CSP-J7 excursions, always relative to one fixed anchor."""
    if mid not in (1,7) or not 1<=repetitions<=20 or not all(math.isfinite(v) for v in (initial,elapsed)):
        raise ValueError('Invalid repeated diagnostic profile')
    duration=6. if mid==1 else 12.;cycle_duration=duration+2.5
    elapsed=max(0.,min(elapsed,cycle_duration*repetitions))
    cycle=min(int(elapsed/cycle_duration),repetitions-1);local=elapsed-cycle*cycle_duration
    amplitude=.5 if mid==1 else 2.
    target=initial+math.radians(amplitude)*.5*(1-math.cos(2*math.pi*min(local,duration)/duration))
    return target,cycle,local


def check_j1_sample(position,velocity,initial,target,voltage):
    if not all(math.isfinite(x) for x in (position,velocity,initial,target,voltage)):
        raise ValueError('Nonfinite feedback')
    if not 42<=voltage<=54: raise ValueError('Supply outside preflight envelope')
    if abs(position-initial)>math.radians(1.) or abs(position-target)>math.radians(1.):
        raise ValueError('J1 position deviated > 1 degree')
    if abs(velocity)>math.radians(3.): raise ValueError('J1 measured speed > 3 deg/s')
