"""Compatibility entry point for the corrected, explicitly scoped diagnostic."""
import argparse
from pathlib import Path
import subprocess
import sys

p=argparse.ArgumentParser()
p.add_argument('--execute-authorized-low-speed-test',action='store_true',required=True)
p.add_argument('--out',default='runs/j1_test_corrected')
a=p.parse_args()
raise SystemExit(subprocess.call([sys.executable,str(Path(__file__).with_name('repair_hardware.py')),
    'move-j1','--kp','200','--out',a.out,'--execute-authorized-test']))
