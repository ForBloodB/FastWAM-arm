"""Restore the single CH340 USB UART; never opens UART/CAN or sends commands."""
from pathlib import Path
import hashlib,json,os,subprocess,time

OUT=Path('/home/ubuntu/reflex-arm/runs/leader_link_20261031')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert os.geteuid()==0
    devices=[]
    for p in Path('/sys/bus/usb/devices').iterdir():
        try:
            if (p/'idVendor').read_text().strip()=='1a86' and (p/'idProduct').read_text().strip()=='7523':devices.append(p)
        except FileNotFoundError:pass
    assert len(devices)==1,devices
    device=devices[0];interfaces=[p for p in device.iterdir() if p.name.startswith(device.name+':')]
    assert len(interfaces)==1,interfaces
    interface=interfaces[0]
    vendor=Path('/usr/lib/udev/rules.d/85-brltty.rules')
    override=Path('/etc/udev/rules.d/85-brltty.rules')
    assert not override.exists(),'Preserve any prior administrator override'
    original=vendor.read_text()
    line='ENV{PRODUCT}=="1a86/7523/*", ENV{BRLTTY_BRAILLE_DRIVER}="bm", GOTO="brltty_usb_run"'
    assert original.count(line)==1
    updated=original.replace(line,'# ReFlex: CH340 is the FashionStar leader UART, not a Braille display.\n# '+line)
    OUT.mkdir(parents=True,exist_ok=True)
    receipt=OUT/'usb_repair.json';assert not receipt.exists()
    (OUT/'brltty_vendor_rules.original').write_bytes(vendor.read_bytes())
    state=dict(device_path=str(device),interface=interface.name,vid_pid='1a86:7523',
        original_vendor_sha256=sha(vendor),prior_override=False,
        brltty_before=subprocess.run(['systemctl','is-active','brltty-udev.service'],capture_output=True,text=True).stdout.strip(),
        source_sha256=sha(__file__),serial_motor_commands=0,can_motor_commands=0)
    override.write_text(updated);override.chmod(0o644)
    state['override_sha256']=sha(override)
    subprocess.run(['udevadm','control','--reload-rules'],check=True)
    subprocess.run(['systemctl','stop','brltty-udev.service'],check=True)
    try:
        if not (interface/'driver').exists():
            Path('/sys/bus/usb/drivers/ch341/bind').write_text(interface.name)
        assert (interface/'driver').resolve().name=='ch341'
        for _ in range(30):
            nodes=[]
            for tty in Path('/sys/class/tty').glob('ttyUSB*'):
                if interface.resolve() in (tty/'device').resolve().parents:nodes.append('/dev/'+tty.name)
            if nodes:break
            time.sleep(.1)
        assert len(nodes)==1,nodes
        state.update(passed=True,serial_nodes=nodes,driver='ch341',
            rollback='Verify override SHA, remove only /etc/udev/rules.d/85-brltty.rules, reload udev rules. Original vendor rule file was never modified. Restart brltty-udev only if its original use is restored.')
    except Exception as exc:
        state.update(passed=False,error=repr(exc));raise
    finally:
        receipt.write_text(json.dumps(state,indent=2))
        print(json.dumps(state),flush=True)


if __name__=='__main__':main()
