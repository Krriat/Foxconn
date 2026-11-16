#!/usr/bin/env python3
"""Verify the completed independent pair and write portable task metadata."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = args.output.resolve()
    report = json.loads((root / 'report.json').read_text())
    assert report['state'] == 'complete' and report['deployment_unchanged']
    assert json.loads((root / 'deployed_before.json').read_text()) == json.loads((root / 'deployed_after.json').read_text())
    workspace = Path(__file__).resolve().parents[2]
    schema = json.loads((workspace / 'prepared_data/suction_20hz_v1/train/meta/keyboard_teach_source.json').read_text())['schema']
    hashes = {}
    lines = ['# 两个独立任务 RGB interACT 训练结果', '',
        '远盖和近盖分别独立训练：视觉骨干使用 ImageNet 初始化，交互与解码层随机初始化。',
        '三路 RGB，无需点云。输入和输出均为 16 维双吸盘布局，数据为 20 Hz。', '',
        '| 任务 | 实际训练步数 | 最佳步数 | 原混合模型误差 | 新模型误差 | 前10步误差（原→新） |',
        '|---|---:|---:|---:|---:|---:|']
    for stage, label in [('far', '远盖'), ('near', '近盖')]:
        result = report['results'][stage]
        manifest = json.loads((root / stage / 'manifest.json').read_text())
        assert manifest['arguments']['initialization'] == 'imagenet'
        assert result['best_step'] > 0
        assert manifest['parameters'] == manifest['trainable_parameters']
        assert result['inference']['checkpoint_reload_matches']
        folder = (root / stage / 'pretrained_model').resolve()
        assert digest(folder / 'model.safetensors') == result['model_sha256']
        assert digest(folder / 'normalization.json') == result['normalization_sha256']
        assert result['model_sha256'] != manifest['source_sha256']
        config = json.loads((folder / 'config.json').read_text())
        assert config['model_type'] == 'interact_rgb' and not config['use_point_cloud']
        assert config['state_layout'] == 'dual_arm_suction'
        updates = [json.loads(line) for line in (root / stage / 'training.jsonl').read_text().splitlines()]
        assert updates[0]['event'] == 'first_optimizer_update' and updates[0]['head_weight_max_delta'] > 0
        task_metadata = dict(stage=stage, fps=20, camera_order=['head', 'left_wrist', 'right_wrist'],
            state_names=schema['state_names'], action_names=schema['state_names'],
            joint_units='radians', suction_off=0, suction_on=1,
            method=manifest['method'], arguments=manifest['arguments'],
            normalization=manifest['normalization'], data=manifest['data'],
            executed_steps=result['executed_steps'], best_step=result['best_step'],
            validation=result['selected'], mixed_model_baseline=result['baseline'],
            limitation=manifest['validation_limits'], physical_validation_performed=False,
            model_sha256=result['model_sha256'])
        write(folder / 'task_manifest.json', task_metadata)
        hashes[stage] = {p.name: digest(p) for p in folder.iterdir() if p.is_file()}
        b, n = result['baseline'], result['selected']
        lines.append(f"| {label} | {result['executed_steps']} | {result['best_step']} | "
            f"{b['all']['joint_mae_deg']:.4f}° | {n['all']['joint_mae_deg']:.4f}° | "
            f"{b['first10']['joint_mae_deg']:.4f}° → {n['first10']['joint_mae_deg']:.4f}° |")
    assert hashes['far']['model.safetensors'] != hashes['near']['model.safetensors']
    write(root / 'artifact_checksums.json', hashes)
    lines += ['', '误差为完整验证集所有有效目标上 14 关节的平均绝对误差，FP32 推理，',
        '已反归一化为角度；前 10 步对应 20 Hz 数据的前 0.5 秒。',
        '验证数据用于权重选择，不是独立测试集。远盖仅 2 条验证示教，近盖 4 条。', '',
        '## 权重', '',
        f'- 远盖：`{root / "far/pretrained_model"}`',
        f'- 近盖：`{root / "near/pretrained_model"}`', '',
        '每个目录含 `model.safetensors`、`config.json`、`normalization.json`、`task_manifest.json`。',
        '复制模型时须同时保留这些文件，不能混用两路的归一化参数。', '',
        '## 验证范围', '',
        '已检查实际梯度更新、权重独立、保存重载一致、RGB-only 配置及源数据映射。',
        'ACT 两路权重、大脑 LoRA 权重及本机机器人配置文件的前后哈希一致。',
        '未替换运行中的 ACT，未调用机械臂或吸盘，也未做实机开盖测试。',
        '纯模型推理计时见 `report.json`，计时不含网络、图像解码和预处理。',
        '各任务的新模型可能不优于原混合模型，应以表内对照值判断，不能把完成训练等同于提升。', '',
        '训练代码与运行说明：`training_tools/interact_tasks/README.md`。', '']
    (root / 'README.md').write_text('\n'.join(lines))
    write(root / 'final_checks.json', dict(passed=True, distinct_task_weights=True,
        no_mixed_weights_initialization=True, full_parameter_updates_verified=True,
        checkpoint_reload_matches=True, normalization_audit=str(workspace / 'training_verification/interact_tasks_20261113/normalization_audit.json'),
        deployed_weights_and_config_unchanged=True, robot_commands_sent=0,
        physical_validation_performed=False))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
