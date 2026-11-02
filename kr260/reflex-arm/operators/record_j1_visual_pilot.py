"""Explicit one-degree pilot, existing protections and guaranteed J1 cleanup.

The board service must be stopped by the operator wrapper before invoking this.
No HTTP route invokes this file. No other axis is actuated.
"""
import json,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(root/'services/leader-follow'),str(root)]
from board_server import State

out=root/'runs/leader_web_20261031';state=State(out);rows=[]
try:
    time.sleep(1.5)
    state.follow.start(dict(axis=1,site_confirmed=True,gain=.1),pilot=1.)
    deadline=time.monotonic()+24
    while state.follow.thread.is_alive() and time.monotonic()<deadline:
        rows.append(dict(wall_ns=time.time_ns(),monotonic_ns=time.monotonic_ns(),follow=state.follow.snapshot()))
        time.sleep(.1)
finally:
    state.follow.stop('视觉复验收尾')
    if state.follow.thread:state.follow.thread.join(3)
    state.stop.set()
    result=state.follow.snapshot();session=result.get('session','visual_pilot_incomplete')
    (out/(session+'_visual_timeline.json')).write_text(json.dumps(rows))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    if state.follow.thread and state.follow.thread.is_alive():raise RuntimeError('J1线程未结束，请现场停止')
