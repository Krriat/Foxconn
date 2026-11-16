#!/usr/bin/env python3
"""Check raw keyboard suction data before export; never connects to hardware.

Exit 0: structure/timing checks pass without quality warnings; 1: invalid data;
2: structurally valid but requires review (gaps, short chunks, missing suction
coverage, etc.). Passing cannot establish physical grasp/task success.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recording"))
import record_xcore_teach_suction as recorder
from xcore_teach_recording.quality import episode_quality


def check_dataset(root, chunk_size=50):
    root = Path(root)
    settings = json.loads((root/"schema.json").read_text())
    recorder.validate_schema(settings)
    paths = sorted(p for p in root.glob("episode_*") if p.is_dir())
    if not paths:
        raise ValueError("没有已保存的 episode；.partial 不参与训练")
    episodes, invalid = [], []
    action_values = {side: set() for side in recorder.SIDES}
    observed_values = {side: set() for side in recorder.SIDES}
    for path in paths:
        try:
            meta, arrays = recorder.validate_episode(path, settings, check_images=True)
            quality = episode_quality(meta, arrays, chunk_size)
            episodes.append(dict(episode=path.name, **quality))
            for side in recorder.SIDES:
                action_values[side].update(quality["suction"][side]["action_values"])
                observed_values[side].update(quality["suction"][side]["observation_values"])
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            invalid.append(dict(episode=path.name, error=str(exc)))
    warnings = []
    for side, label in (("left", "左"), ("right", "右")):
        if action_values[side] != {0, 1} or observed_values[side] != {0, 1}:
            warnings.append(f"数据集缺少{label}吸盘完整0/1动作或反馈覆盖")
    review = bool(warnings or any(e["warnings"] for e in episodes))
    status = "invalid" if invalid else "needs_review" if review else "checks_passed"
    return dict(root=str(root.resolve()), status=status, chunk_size=chunk_size,
                valid_episodes=episodes, invalid_episodes=invalid,
                dataset_warnings=warnings, task_success_requires_review=True,
                suction_coverage={side: dict(action=sorted(action_values[side]),
                                            observation=sorted(observed_values[side]))
                                  for side in recorder.SIDES})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.chunk_size < 1:
        parser.error("--chunk-size 必须是正整数")
    try:
        if args.report is not None and args.report.exists():
            raise FileExistsError("报告已存在，请选择新的 --report 路径")
        result = check_dataset(args.root, args.chunk_size)
        for episode in result["valid_episodes"]:
            print(f"{episode['episode']}: 格式/时间校验通过，{episode['frames']}帧，"
                  f"{episode['saved_fps']:.2f}/{episode['target_fps']:g}帧/秒，"
                  f"{len(episode['segment_frames'])}个连续片段，"
                  f"最长{max(episode['segment_frames'], default=0)}帧")
            for warning in episode["warnings"]:
                print("  需复核："+warning)
        for episode in result["invalid_episodes"]:
            print(f"{episode['episode']}: 无效：{episode['error']}")
        for warning in result["dataset_warnings"]:
            print("数据集需复核："+warning)
        print("结果："+result["status"]+"；任务成功与吸附效果需结合画面确认")
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
        return {"checks_passed": 0, "needs_review": 2, "invalid": 1}[result["status"]]
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
        print("数据检查失败："+str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
