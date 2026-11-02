import json,math,struct,tempfile,time,unittest
from pathlib import Path
from leader_reader import Quality,decode,request
from follow_control import Target,Follow
from board_server import State

class InputTests(unittest.TestCase):
    def test_zero_remains_reliable_after_confirmation(self):
        q=Quality()
        for n in range(100):
            value,reliable,_=q.update(0.,n*.04)
            if n>=20:self.assertTrue(reliable)
    def test_near_zero_crossing_remains_continuous(self):
        q=Quality();q.update(.4,1.)
        self.assertTrue(q.update(.2,1.04)[1]);self.assertTrue(q.update(0.,1.08)[1])
    def test_power_reset_jump_latches_bad_until_reconnect(self):
        q=Quality();q.update(40.,1.)
        self.assertFalse(q.update(0.,1.04)[1]);self.assertFalse(q.update(40.,1.08)[1])
    def test_gap_requires_new_zero_confirmation(self):
        q=Quality();q.update(.4,1.);q.update(0.,1.04)
        self.assertFalse(q.update(0.,2.)[1])
    def test_query_only_and_packet_integrity(self):
        self.assertEqual(request(1),bytes([18,76,16,1,1,112]))
        p=bytes([5,28,16,7,1])+struct.pack('<i',-123)+b'\0\0';p+=bytes([sum(p)&255])
        self.assertEqual(decode(p,1),-12.3)
        for bad in (p[:-1]+b'\0',p[:-2],p):
            with self.assertRaises(ValueError):decode(bad,2)
        with self.assertRaises(ValueError):decode(p[:-1]+bytes([p[-1]^1]),1)

class ControlTests(unittest.TestCase):
    def test_relative_anchor_no_start_jump_and_hard_bounds(self):
        t=Target(90.,.2)
        self.assertEqual(t.update(90.,.02),.2)
        prev=.2
        for _ in range(200):
            value=t.update(100.,.02)
            self.assertLessEqual(abs(value-prev),math.radians(.005)+1e-12);prev=value
            self.assertLessEqual(abs(value-.2),math.radians(1.)+1e-12)
    def test_sign_units_and_faults(self):
        t=Target(5.,.5,-1,.5)
        self.assertLess(t.update(6.,.02),.5)
        for value,dt in [(51.,.02),(5.,.1),(float('nan'),.02),(5.,0.)]:
            with self.assertRaises(ValueError):t.update(value,dt)
    def test_only_j1_can_be_started_no_other_axes(self):
        with tempfile.TemporaryDirectory() as d:
            s=State(Path(d),hardware=False);s.follow.ready=True
            for axis in [0,2,3,4,5,6,7,True,'1']:
                with self.assertRaises(ValueError):s.follow.start(dict(axis=axis,site_confirmed=True))
            self.assertIsNone(s.follow.thread)
    def test_heartbeat_owner_and_expiry(self):
        with tempfile.TemporaryDirectory() as d:
            f=State(Path(d),hardware=False).follow;f.token='owner';f.lease=time.monotonic()-2
            with self.assertRaises(InterruptedError):f.check_owner()
            with self.assertRaises(ValueError):f.heartbeat('wrong')
            f.heartbeat('owner');f.check_owner();f.stop()
            with self.assertRaises(InterruptedError):f.check_owner()
    def test_no_saved_proof_no_hardware_start(self):
        with tempfile.TemporaryDirectory() as d:
            f=State(Path(d),hardware=False).follow
            with self.assertRaises(ValueError):f.start(dict(axis=1,site_confirmed=True))
    def test_bad_proof_does_not_release(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'j1_driver_verified.json').write_text(json.dumps(dict(passed=True,capability_hash='old')))
            self.assertFalse(State(Path(d),hardware=False).follow.ready)

if __name__=='__main__':unittest.main()
