import argparse
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import JointState
import tf2_ros
import tf_transformations

parser = argparse.ArgumentParser()
parser.add_argument('--out', required=True)
args = parser.parse_args()
rclpy.init()
node = Node('fastwam_real_state_capture')
executor = MultiThreadedExecutor(num_threads=2)
executor.add_node(node)
threading.Thread(target=executor.spin, daemon=True).start()
latest = None

def callback(msg):
    global latest
    if len(msg.position) >= 8:
        latest = msg

node.create_subscription(JointState, '/rebotarm/joint_states', callback, qos_profile_sensor_data)
buffer = tf2_ros.Buffer()
listener = tf2_ros.TransformListener(buffer, node)
deadline = time.monotonic() + 8
while latest is None and time.monotonic() < deadline:
    time.sleep(.02)
if latest is None:
    raise RuntimeError('No real joint state')
transform = buffer.lookup_transform('base_link', 'gripper_end', Time(), timeout=Duration(seconds=2))
t = transform.transform.translation
q = transform.transform.rotation
quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
quat /= np.linalg.norm(quat)
if quat[3] < 0:
    quat = -quat
vector_norm = float(np.linalg.norm(quat[:3]))
angle = 2 * math.atan2(vector_norm, float(quat[3]))
axis_angle = np.zeros(3) if vector_norm < 1e-12 else quat[:3] / vector_norm * angle
position = [t.x, t.y, t.z]
gripper = [float(latest.position[6]), float(latest.position[7])]
state = np.array(position + axis_angle.tolist() + gripper, dtype=np.float32)
result = {
    'source': 'KR260 official joint_states and TF; no simulation or placeholder',
    'state_layout': 'eef xyz m, eef axis-angle rad, two gripper joint positions m',
    'raw_state': state.tolist(),
    'joint_positions_rad': [float(v) for v in latest.position[:6]],
    'gripper_positions_m': gripper,
    'T_base_gripper': tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w]).tolist(),
}
result['T_base_gripper'][0][3] = t.x
result['T_base_gripper'][1][3] = t.y
result['T_base_gripper'][2][3] = t.z
Path(args.out).write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
executor.shutdown()
node.destroy_node()
rclpy.shutdown()
