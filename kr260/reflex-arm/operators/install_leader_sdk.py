"""Install one verified, dependency-free ARM64 wheel in an isolated venv."""
from pathlib import Path,PurePosixPath
import hashlib,json,platform,subprocess,venv,zipfile

ROOT=Path('/home/ubuntu/reflex-arm')
WHEEL='motorbridge_smart_servo-0.0.4-cp39-abi3-manylinux_2_17_aarch64.manylinux2014_aarch64.whl'
EXPECTED='348cef6a647e5c7f9cc8e8ce1f3c806af4522e1087172bac2f8a1a0daa3592b6'
OUT=ROOT/'runs/leader_link_20261031'
ENV=ROOT/'.venv-leader'


def main():
    assert platform.machine()=='aarch64'
    wheel=OUT/'wheels'/WHEEL
    assert hashlib.sha256(wheel.read_bytes()).hexdigest()==EXPECTED
    venv.EnvBuilder(with_pip=False,symlinks=True).create(ENV)
    site=ENV/'lib/python3.10/site-packages'
    with zipfile.ZipFile(wheel) as z:
        metadata=z.read('motorbridge_smart_servo-0.0.4.dist-info/METADATA').decode()
        assert not any(line.startswith('Requires-Dist:') for line in metadata.splitlines())
        for info in z.infolist():
            p=PurePosixPath(info.filename)
            assert not p.is_absolute() and '..' not in p.parts
            assert p.parts[0] in ['motorbridge_smart_servo','motorbridge_smart_servo-0.0.4.dist-info']
            if info.is_dir():continue
            target=site/info.filename;assert not target.exists(),target
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(info))
    result=subprocess.run([str(ENV/'bin/python'),'-c',
        'import importlib.metadata as m;from motorbridge_smart_servo import FashionStarServo;print(m.version("motorbridge-smart-servo"))'],
        capture_output=True,text=True,check=True)
    receipt=dict(passed=True,version=result.stdout.strip(),wheel_sha256=EXPECTED,wheel=WHEEL,
        environment=str(ENV),method='Verified wheel extraction in stdlib venv without pip; no external dependencies in METADATA',
        global_python_unchanged=True,serial_opened=False,can_opened=False,motor_commands=0)
    (OUT/'sdk_install.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
