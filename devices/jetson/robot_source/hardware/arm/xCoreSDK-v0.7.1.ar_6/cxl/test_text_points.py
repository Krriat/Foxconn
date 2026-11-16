"""Offline tests; do not require ROS or modify the real workflow."""
import json
import tempfile
import unittest
from pathlib import Path
import text_points as p


class PointTests(unittest.TestCase):
    def setUp(self):
        self.sources = p.load_sources(p.HERE/'text_points.sources.json')
        self.s = self.sources['part_left']
        self.payload = dict(detected=True, frame_id='base_link', position=dict(x=1,y=2,z=3),
                            stamp=dict(sec=10,nanosec=1), **{'class':'part1','confidence':0.9})

    def decode(self):
        return p.decode(self.s, {'data': json.dumps(self.payload)})

    def test_json_and_validity(self):
        self.assertEqual(self.decode()['xyz'], [1,2,3])
        self.payload['detected'] = False
        with self.assertRaises(ValueError): self.decode()

    def test_frame_and_nonfinite(self):
        self.payload['frame_id'] = 'camera'
        with self.assertRaises(ValueError): self.decode()
        self.payload['frame_id'] = 'base_link'
        self.payload['position']['x'] = float('nan')
        with self.assertRaises(ValueError): self.decode()

    def test_pose_and_arbitrary_field_mapping(self):
        msg = {'header': {'frame_id':'base_link','stamp':{'sec':10,'nanosec':0}},
               'pose': {'position': {'x':1,'y':2,'z':3}}}
        self.assertEqual(p.decode(self.sources['material_pose'], msg)['xyz'], [1,2,3])
        source = dict(self.s, xyz=['items.0.x','items.0.y','items.0.z'], scale=0.001)
        self.payload['items'] = [dict(x=1000,y=2000,z=3000)]
        self.assertEqual(p.decode(source, {'data':json.dumps(self.payload)})['xyz'], [1,2,3])

    def test_transform_and_template(self):
        self.s['transform'] = None
        with self.assertRaises(ValueError): p.make_pose(self.s, self.decode(), [0,0,1], [0,0,0])
        self.s.update(target_frame='left_external', transform=[[0,-1,0,1],[1,0,0,2],[0,0,1,3],[0,0,0,1]])
        self.assertEqual(p.make_pose(self.s, self.decode(), [0,0,1], [0,0,.1]), [-1,3,6.1,0,0,1])
        with self.assertRaises(ValueError): p.transform_matrix([[2,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])

    def test_preservation_overwrite_backup_and_conflict(self):
        text = '# keep\r\n[poses]\r\ntemplate 1 2 3 0.1 0.2 0.3 # keep\r\n[joints]\r\nq 0 0 0 0 0 0 0\r\n[steps]\r\nwait 1\r\n'
        updated = p.edit_pose(text, 'target', [4,5,6,.1,.2,.3], 'source=test')
        self.assertEqual(updated.split('[joints]')[1], text.split('[joints]')[1])
        self.assertIn('template 1 2 3 0.1 0.2 0.3 # keep\r\n', updated)
        with self.assertRaises(ValueError): p.edit_pose(updated, 'target', [0]*6, '')
        self.assertEqual(p.poses_in(p.edit_pose(updated, 'target', [0]*6, '', True))['target'], [0]*6)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'test.conf'
            path.write_bytes(text.encode())
            backup = p.save_config(path, text.encode(), updated)
            self.assertEqual(backup.read_bytes(), text.encode())
            self.assertEqual(path.read_bytes(), updated.encode())
            with self.assertRaises(ValueError): p.save_config(path, text.encode(), updated)


if __name__ == '__main__': unittest.main()
