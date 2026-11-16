"""Regression for explicit suction release while a fault stop is draining."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class StopCleanup(unittest.TestCase):
    def test_manual_stop_releases_during_stopping_and_preserves_fault(self):
        source=Path(__file__).resolve().parents[1]/'two_stage_policy/executor_node.py'
        tree=ast.parse(source.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Executor')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='request_stop')
        scope={}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(source),'exec'),scope)
        releases=[]
        fake=SimpleNamespace(sequence_job=None,phase='STOPPING',mode='live',run_id='test',
            suction=SimpleNamespace(release_left=lambda run:releases.append(run)),
            shadow_suction=[True,False],stop_destination='FAULT')
        scope['request_stop'](fake,'Manual stop')
        self.assertEqual(releases,['test'])
        self.assertEqual(fake.shadow_suction,[False,False])
        self.assertEqual(fake.stop_destination,'FAULT')


if __name__=='__main__':unittest.main()
