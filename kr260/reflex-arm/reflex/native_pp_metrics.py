"""Keep goal-step response error separate from stationary PP pose accuracy."""
import math


def wrist_tracking_metrics(rows,stationary_pp=False):
    if not rows:raise ValueError('No tracking rows')
    if any(not math.isfinite(row[key]) for row in rows for key in ('actual_rad','command_rad')):
        raise ValueError('Nonfinite tracking data')
    rmse=lambda values:math.sqrt(sum(math.degrees(row['actual_rad']-row['command_rad'])**2 for row in values)/len(values))
    whole=rmse(rows)
    result={'tracking_rmse_motor_deg':whole,'whole_goal_step_rmse_motor_deg':whole,
        'tracking_error_basis':'entire commanded reference'}
    if not stationary_pp:return result
    selected=[];windows={}
    for phase in ('peak','return'):
        values=[row for row in rows if row.get('pp_settled_phase')==phase]
        times=[row['ns'] for row in values]
        if len(values)<120 or times[-1]-times[0]<5_900_000_000 or any(
            not 0<b-a<=50_000_000 for a,b in zip(times,times[1:])):
            raise ValueError('Missing, short or stale PP stationary tracking window')
        windows[phase]={'samples':len(values),'duration_s':(times[-1]-times[0])/1e9,
            'rmse_motor_deg':rmse(values)}
        selected.extend(values)
    result.update(tracking_rmse_motor_deg=max(window['rmse_motor_deg'] for window in windows.values()),
        pp_combined_stationary_rmse_motor_deg=rmse(selected),
        tracking_error_basis='PP worst settled-window RMSE',pp_stationary_tracking_windows=windows)
    return result
