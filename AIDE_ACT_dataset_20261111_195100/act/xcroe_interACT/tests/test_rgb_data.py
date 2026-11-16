import json
import sys
import time
import types

import numpy as np
import pytest
import torch

from xcore_interact.cameras import RosRgbdRig
from xcore_interact.data import InterACTWindowDataset, NormalizationStats, preprocess_observation
from xcore_interact.schema import (
    CameraConfig, HEAD_POINT_CLOUD_KEY, SUCTION_STATE_NAMES,
    dataset_features, infer_state_layout,
)


def test_rgb_camera_bundle_needs_no_depth_or_intrinsics():
    rig = RosRgbdRig(CameraConfig())
    now = time.monotonic()
    for key in ('head', 'left_wrist', 'right_wrist'):
        rig._store_frame(key, np.zeros((8, 8, 3), dtype=np.uint8), 100., now)
    rig._connected = True
    frame = rig.read(timeout_ms=1)
    assert frame.head_points is None
    assert rig._intrinsics is None
    assert frame.head_rgb.shape == (8, 8, 3)
    with pytest.raises(TimeoutError):
        rig.read(timeout_ms=1)  # A second read still requires fresh RGB frames.


def test_3d_camera_bundle_still_requires_depth():
    rig = RosRgbdRig(CameraConfig(use_point_cloud=True))
    now = time.monotonic()
    for key in ('head', 'left_wrist', 'right_wrist'):
        rig._store_frame(key, np.zeros((8, 8, 3), dtype=np.uint8), 100., now)
    with pytest.raises(TimeoutError, match='depth'):
        rig._wait_bundle(0., require_new=False)


def test_rgb_dataset_and_suction_schema_without_point_column(tmp_path, monkeypatch):
    features = dataset_features(CameraConfig())
    assert HEAD_POINT_CLOUD_KEY not in features
    for key in ('observation.state', 'action'):
        features[key]['names'] = list(SUCTION_STATE_NAMES)
    info = dict(fps=20, features=features)
    assert infer_state_layout(info) == 'dual_arm_suction'
    (tmp_path / 'meta').mkdir()
    (tmp_path / 'meta/info.json').write_text(json.dumps(info))

    class FakeLeRobotDataset:
        def __init__(self, *args, **kwargs):
            pass

        def __len__(self):
            return 1

        def __getitem__(self, index):
            item = {'observation.state': torch.arange(16.), 'action': torch.zeros(4, 16),
                    'action_is_pad': torch.tensor([False, False, True, True])}
            item.update({f'observation.images.{key}': torch.zeros(3, 8, 8)
                         for key in ('head', 'left_wrist', 'right_wrist')})
            return item

    module = types.ModuleType('lerobot.datasets.lerobot_dataset')
    module.LeRobotDataset = FakeLeRobotDataset
    monkeypatch.setitem(sys.modules, module.__name__, module)
    stats = NormalizationStats([0.] * 16, [1.] * 16, [0.] * 16, [1.] * 16)
    dataset = InterACTWindowDataset(tmp_path, 'local/test', chunk_size=4, image_size=16,
                                   stats=stats, state_layout='dual_arm_suction')
    item = dataset[0]
    assert 'points' not in item
    assert item['images'].shape == (3, 3, 16, 16)
    torch.testing.assert_close(item['state'], torch.arange(16.))
    with pytest.raises(ValueError, match='layout'):
        InterACTWindowDataset(tmp_path, 'local/test', chunk_size=4, image_size=16,
                              stats=stats, state_layout='bimanual_gripper')
    with pytest.raises(ValueError, match='point-cloud feature'):
        InterACTWindowDataset(tmp_path, 'local/test', chunk_size=4, image_size=16,
                              stats=stats, use_point_cloud=True)


def test_preprocessing_without_points():
    stats = NormalizationStats([0.] * 16, [1.] * 16, [0.] * 16, [1.] * 16)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    images, points, state = preprocess_observation(
        image, image, image, state=np.zeros(16, dtype=np.float32),
        stats=stats, image_size=16, device='cpu',
    )
    assert points is None
    assert images.shape == (1, 3, 3, 16, 16)
    assert state.shape == (1, 16)


def test_layout_validation_rejects_unknown_or_mismatched_names():
    features = dataset_features(CameraConfig())
    assert infer_state_layout({'features': features}) == 'bimanual_gripper'
    features['action']['names'] = list(SUCTION_STATE_NAMES)
    with pytest.raises(ValueError, match='must match'):
        infer_state_layout({'features': features})
    features['observation.state']['names'] = ['unknown'] * 16
    features['action']['names'] = ['unknown'] * 16
    with pytest.raises(ValueError, match='unsupported'):
        infer_state_layout({'features': features})


def _validator():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / 'scripts/dataset_tools/validate_dataset.py'
    spec = importlib.util.spec_from_file_location('validate_interact_dataset', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validation_accepts_rgb_and_requires_points_only_on_request():
    validator = _validator()
    info = {'features': dataset_features(CameraConfig())}
    assert validator.validate_schema(info) == []
    assert validator.validate_schema(info, use_point_cloud=True) == ['head point cloud shape must be [1024,6]']
    for key in ('observation.state', 'action'):
        info['features'][key]['names'] = list(SUCTION_STATE_NAMES)
    assert validator.validate_schema(info) == []
    info['features'] = dataset_features(CameraConfig(use_point_cloud=True))
    assert validator.validate_schema(info, use_point_cloud=True) == []


@pytest.mark.parametrize('suction_value,expected_status', [(1., 0), (2., 1)])
def test_audit_rgb_parquet_never_requests_points(tmp_path, monkeypatch, suction_value, expected_status):
    validator = _validator()
    features = dataset_features(CameraConfig())
    for key in ('observation.state', 'action'):
        features[key]['names'] = list(SUCTION_STATE_NAMES)
    (tmp_path / 'meta').mkdir()
    (tmp_path / 'meta/info.json').write_text(json.dumps({'features': features}))
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/episode.parquet').touch()
    vector = [0.] * 14 + [suction_value, 0.]

    class Column:
        def __init__(self, values):
            self.values = values

        def to_pylist(self):
            return self.values

    class Batch:
        num_rows = 20

        def column(self, index):
            return Column([0] * 20 if index == 0 else [vector] * 20)

    class ParquetFile:
        def __init__(self, path):
            pass

        def iter_batches(self, *, batch_size, columns):
            assert columns == ['episode_index', 'observation.state', 'action']
            yield Batch()

    package, parquet = types.ModuleType('pyarrow'), types.ModuleType('pyarrow.parquet')
    parquet.ParquetFile = ParquetFile
    package.parquet = parquet
    monkeypatch.setitem(sys.modules, 'pyarrow', package)
    monkeypatch.setitem(sys.modules, 'pyarrow.parquet', parquet)
    monkeypatch.setattr(sys, 'argv', ['validate_dataset.py', '--dataset-root', str(tmp_path)])
    assert validator.main() == expected_status
    report = json.loads((tmp_path / 'meta/validation_interact3d.json').read_text())
    assert not report['use_point_cloud']
    assert report['valid'] == (expected_status == 0)
