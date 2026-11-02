import unittest,tempfile,threading,json,urllib.request,urllib.error
from pathlib import Path
from copy import deepcopy
from http.server import ThreadingHTTPServer
from engine import Engine,Problem
from board_server import State,handler


class Clock:
    def __init__(self):self.now=1.
    def __call__(self):return self.now


def sample(value=0.):return [dict(id=i,raw_deg=value,filtered_deg=value,reliable=True) for i in range(7)]


class Guards(unittest.TestCase):
    def setUp(self):self.clock=Clock();self.e=Engine(self.clock);self.e.ingest_leader(sample())
    def test_rate_and_travel_bound(self):
        self.e.capture();self.e.start()
        for _ in range(50):
            old=dict(self.e.target);self.clock.now+=.1;self.e.ingest_leader(sample(30.))
            self.assertTrue(all(abs(self.e.target[i]-old[i])<=.05+1e-9 for i in old))
        self.assertEqual(self.e.mode,'preview');self.assertTrue(all(abs(v)<=1. for v in self.e.target.values()))
    def test_stale_does_not_resume(self):
        self.e.capture();self.e.start();self.clock.now+=.6;self.e.snapshot();self.assertEqual(self.e.mode,'monitor')
        self.e.ingest_leader(sample(1.));self.assertEqual(self.e.mode,'monitor')
    def test_unreliable_stops(self):
        self.e.capture();self.e.start();r=sample();r[3]['reliable']=False;self.clock.now+=.1;self.e.ingest_leader(r)
        self.assertEqual(self.e.mode,'monitor')
    def test_reconfigure_invalidates_anchor(self):
        self.e.capture();self.e.start();rows=deepcopy(self.e.mapping);rows[2]['direction']=-1;self.e.update_mapping(rows)
        self.assertIsNone(self.e.reference);self.assertEqual(self.e.mode,'monitor')
        with self.assertRaises(Problem):self.e.start()
    def test_invalid_rows_leave_state_unchanged(self):
        for change in [lambda r:r[0].update(gain=float('nan')),lambda r:r[1].update(leader_id=0),lambda r:r.__setitem__(0,None),lambda r:r[0].update(speed_deg_s=100.)]:
            r=deepcopy(self.e.mapping);change(r)
            with self.assertRaises(Problem):self.e.update_mapping(r)
            self.assertEqual(self.e.revision,0)
    def test_without_follower_never_fakes_absolute_target(self):
        self.e.capture();self.e.start();self.clock.now+=.1;self.e.ingest_leader(sample(2.))
        self.assertTrue(all(r['target_absolute_deg'] is None for r in self.e.snapshot()['joints']))
    def test_reference_records_without_zero_writes(self):
        self.e.ingest_follower([dict(id=i,position_deg=float(i)) for i in range(1,8)])
        self.e.capture();self.assertEqual(self.e.reference['follower'][3],3.)
        self.assertEqual(self.e.snapshot()['motion_commands'],0)
    def test_single_axis_preview_does_not_require_unused_unreliable_axis(self):
        r=sample();r[1]['reliable']=False;self.e.ingest_leader(r)
        self.assertEqual(self.e.snapshot()['leader_online'],7);self.assertEqual(self.e.snapshot()['leader_reliable'],6)
        with self.assertRaises(Problem):self.e.capture()
        mapping=deepcopy(self.e.mapping)
        for row in mapping:row['enabled']=row['leader_id']==0
        self.e.update_mapping(mapping);self.e.capture();self.e.start()
        self.assertEqual(self.e.mode,'preview')
    def test_no_selected_axis_cannot_capture(self):
        mapping=deepcopy(self.e.mapping)
        for row in mapping:row['enabled']=False
        self.e.update_mapping(mapping)
        with self.assertRaises(Problem):self.e.capture()


class Routes(unittest.TestCase):
    def test_hardware_route_remains_rejected_and_api_readonly(self):
        with tempfile.TemporaryDirectory() as d:
            state=State(Path(d),hardware=False);state.engine.ingest_leader(sample())
            server=ThreadingHTTPServer(('127.0.0.1',0),handler(state));t=threading.Thread(target=server.serve_forever,daemon=True);t.start()
            base=f'http://127.0.0.1:{server.server_port}'
            try:
                req=urllib.request.Request(base+'/api/follow/start',data=b'{}',headers={'X-Reflex-Proxy':'leader-web'})
                with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(req)
                self.assertEqual(error.exception.code,409)
                status=json.load(urllib.request.urlopen(base+'/api/status'))
                self.assertFalse(status['hardware_follow_enabled']);self.assertEqual(status['motion_commands'],0)
                req=urllib.request.Request(base+'/api/reference',data=b'{}')
                with self.assertRaises(urllib.error.HTTPError) as error:urllib.request.urlopen(req)
                self.assertEqual(error.exception.code,403)
                csv=urllib.request.urlopen(base+'/api/export.csv').read().decode('utf-8-sig')
                self.assertIn('motion_commands',csv);self.assertEqual(len(csv.splitlines()),8)
            finally:server.shutdown();server.server_close();t.join()


if __name__=='__main__':unittest.main()
