"""Manual completion must not inherit the automatic brain's 90-second limit."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class ManualTimeout(unittest.TestCase):
    def setUp(self):
        path=Path(__file__).resolve().parents[1]/'two_stage_policy/executor_node.py'
        tree=ast.parse(path.read_text())
        cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Executor')
        method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='sequence_tick')
        scope={'time':SimpleNamespace(monotonic=lambda:1000.)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method],type_ignores=[])),str(path),'exec'),scope)
        self.tick=scope['sequence_tick']

    def test_manual_stage_has_no_completion_deadline(self):
        self.tick(SimpleNamespace(sequence_job={'completion_mode':'manual'},round_start=0.))

    def test_automatic_stage_retains_timeout(self):
        with self.assertRaises(TimeoutError):
            self.tick(SimpleNamespace(sequence_job={'completion_mode':'qwen_lora'},round_start=0.))


if __name__=='__main__':unittest.main()
