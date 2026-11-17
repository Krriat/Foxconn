"""Exercise output arbitration and cancellation without touching any device."""
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from suction_io import Suction


class Function:
    def __init__(self, function):
        self.function = function
    def __call__(self, *args):
        return self.function(*args)


class FakeLibrary:
    def __init__(self, shared=None):
        self.reads, self.calls, self.values = 0, [], [False, False]
        self.shared = shared
        self.m050_config_default = Function(lambda config: None)
        self.m050_create = Function(lambda config, client, error: 0)
        self.m050_destroy = Function(lambda client: None)
        self.m050_read_do = Function(self.read)
        self.m050_set_do = Function(self.write)
    def read(self, *args):
        self.reads += 1
        raise AssertionError("Startup must reuse fresh ROS feedback, not open a second reader")
    def write(self, client, channel, closed, verified, error):
        if self.shared:
            with self.shared['lock']:
                self.shared['active'] += 1
                self.shared['max_active'] = max(self.shared['max_active'], self.shared['active'])
            time.sleep(.04)
        self.values[channel-1] = bool(closed)
        verified._obj.ch1, verified._obj.ch2 = map(int, self.values)
        self.calls.append((channel, closed))
        if self.shared:
            with self.shared['lock']:
                self.shared['active'] -= 1
        return 0


class SuctionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / 'gateway.lock'
        self.instances = []
    def tearDown(self):
        for instance in self.instances:
            instance.close()
        self.directory.cleanup()
    def create(self, library):
        with patch('suction_io.C.CDLL', return_value=library):
            instance = Suction({'robot_root':'/unused','io_timeout_ms':250}, [False, False], lock_path=self.path)
        self.instances.append(instance)
        return instance
    def wait(self, predicate):
        end = time.monotonic() + 1.0
        while time.monotonic() < end:
            if predicate():return
            time.sleep(.005)
        self.fail('output worker did not finish')
    def test_constructor_performs_no_device_read_or_write(self):
        library = FakeLibrary();instance = self.create(library)
        self.assertEqual(library.reads, 0)
        self.assertEqual(library.calls, [])
        self.assertEqual(instance.state()['verified'], [False,False])
    def test_constant_zero_is_valid_and_does_not_generate_writes(self):
        library = FakeLibrary();instance = self.create(library)
        instance.request('round', [False,False]);time.sleep(.08)
        self.assertEqual(library.calls, [])
        self.assertEqual(instance.state()['error'], '')
    def test_task_left_zero_one_zero_and_right_always_zero(self):
        library = FakeLibrary();instance = self.create(library)
        instance.request('round', [True,False])
        self.wait(lambda:instance.state()['writes']==1)
        instance.request('round', [False,False])
        self.wait(lambda:instance.state()['writes']==2)
        self.assertEqual(library.calls, [(1,1),(1,0)])
        self.assertEqual(instance.state()['verified'], [False,False])
    def test_stop_cancels_write_waiting_for_gateway_lock(self):
        import fcntl,os
        library = FakeLibrary();instance = self.create(library)
        descriptor=os.open(self.path,os.O_RDWR)
        try:
            fcntl.flock(descriptor,fcntl.LOCK_EX)
            instance.request('round',[True,False]);time.sleep(.05)
            self.assertTrue(instance.state()['pending_operation'])
            instance.suspend()
        finally:
            fcntl.flock(descriptor,fcntl.LOCK_UN);os.close(descriptor)
        self.wait(lambda:not instance.state()['pending_operation'])
        self.assertEqual(library.calls, [])
    def test_manual_stop_releases_only_left_and_waits_for_readback(self):
        library = FakeLibrary();instance = self.create(library)
        instance.request('round', [True,True])
        self.wait(lambda:instance.state()['verified']==[True,True])
        instance.suspend();instance.release_left('round')
        self.wait(lambda:instance.state()['settled'] and not instance.state()['pending_operation'])
        self.assertEqual(instance.state()['verified'], [False,True])
        self.assertEqual(library.calls, [(1,1),(2,1),(1,0)])
    def test_separate_clients_serialize_their_native_operations(self):
        shared={'lock':threading.Lock(),'active':0,'max_active':0}
        first,second=FakeLibrary(shared),FakeLibrary(shared)
        a,b=self.create(first),self.create(second)
        a.request('a',[True,False]);b.request('b',[True,False])
        self.wait(lambda:a.state()['writes']==1 and b.state()['writes']==1)
        self.assertEqual(shared['max_active'],1)


if __name__=='__main__':unittest.main()
