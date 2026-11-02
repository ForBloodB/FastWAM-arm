"""Quality gates for a stationary robot/image pair; no device APIs."""
import math
import numpy as np
from reflex.calibration import rigid


def assemble_pair(model,robot,board,window):
    if not robot.get('success'):raise ValueError('Robot stationary monitor failed')
    ranges=np.asarray(robot['q_range_rad'],float)
    q=np.asarray(robot['q_mean_rad'],float)
    if q.shape!=(7,) or ranges.shape!=(7,) or not np.isfinite(np.r_[q,ranges]).all():
        raise ValueError('Invalid joint window')
    if np.max(ranges)>=.001 or np.min(ranges)<0:raise ValueError('Robot moved during capture')
    if not window.get('camera_completed_before_monitor_end'):raise ValueError('Camera outside stationary window')
    if not window.get('camera_started_after_monitor_ready'):raise ValueError('Camera started before monitoring')
    if not board.get('success') or not math.isfinite(board['reprojection_rms_px']) or board['reprojection_rms_px']>.7:
        raise ValueError('Board not detected with sufficient quality')
    if np.any(q[:6]<model.limits[:,0]) or np.any(q[:6]>model.limits[:,1]):
        raise ValueError('Measured robot pose outside URDF limits')
    return {'T_base_gripper':model.fk(q[:6]).tolist(),
            'T_camera_board':rigid(board['T_camera_board']).tolist(),
            'reprojection_rms_px':board['reprojection_rms_px'],'q_rad':q.tolist(),
            'q_range_rad':ranges.tolist(),'tool_frame':model.tip,
            'pair_accepted':True,'camera_extrinsic_calibrated':False}
