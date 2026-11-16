"""Standalone ACT runtime; full-precision original weights, no LeRobot/dataset."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from safetensors.torch import load_file


def joint_vector(message, side):
    if len(message.name) != len(message.position) or len(set(message.name)) != len(message.name):
        raise ValueError('Invalid JointState names or positions')
    mapping = dict(zip(message.name, message.position))
    values = np.array([mapping[f'{side}_joint_{i}'] for i in range(1, 8)], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite joint state')
    return values


class TensorRTForward:
    def __init__(self, path, expected_weights):
        sys.path.append('/usr/lib/python3.10/dist-packages')
        import tensorrt as trt
        metadata = json.loads(path.with_suffix('.engine.json').read_text())
        payload = path.read_bytes()
        if metadata['weights_sha256'] != expected_weights or hashlib.sha256(payload).hexdigest() != metadata['engine_sha256']:
            raise ValueError('TensorRT engine does not match reviewed ACT weights')
        if metadata['tensorrt_version'] != trt.__version__:
            raise ValueError('TensorRT engine version differs from runtime')
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(payload)
        self.context = self.engine.create_execution_context()
        expected = {'head': (1,3,480,640), 'left_wrist': (1,3,480,640),
                    'right_wrist': (1,3,480,640), 'state': (1,16), 'actions': (1,50,16)}
        assert self.engine.num_io_tensors == len(expected)
        for key, shape in expected.items():
            assert tuple(self.engine.get_tensor_shape(key)) == shape
            assert self.engine.get_tensor_dtype(key) == trt.float32
        self.output = torch.empty((1,50,16),device='cuda',dtype=torch.float32)
        self.stream = torch.cuda.Stream()

    def __call__(self, batch):
        inputs = dict(zip(('head','left_wrist','right_wrist'),batch['observation.images']))
        inputs['state'] = batch['observation.state']
        for name, value in inputs.items():
            assert value.is_contiguous() and value.is_cuda and value.dtype == torch.float32
            self.context.set_tensor_address(name,value.data_ptr())
            value.record_stream(self.stream)
        self.context.set_tensor_address('actions',self.output.data_ptr())
        current = torch.cuda.current_stream()
        self.stream.wait_stream(current)
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError('TensorRT execution failed')
        current.wait_stream(self.stream)
        return self.output, None


class ActEngine:
    def __init__(self, checkpoint, bundle, device='cuda', backend='pytorch', engine_path=None):
        checkpoint, bundle = Path(checkpoint), Path(bundle)
        self.device = device
        manifest = json.loads((checkpoint / 'artifact_checksums.json').read_text())
        for name, expected in manifest.items():
            if hashlib.sha256((checkpoint / name).read_bytes()).hexdigest() != expected:
                raise ValueError('ACT artifact checksum mismatch: ' + name)
        sys.path.insert(0, str(bundle))
        cfg = SimpleNamespace(**json.loads((checkpoint / 'config.json').read_text()))
        cfg.pretrained_backbone_weights = None
        cfg.image_features = {k: SimpleNamespace(**v) for k, v in cfg.input_features.items() if v['type'] == 'VISUAL'}
        cfg.robot_state_feature = SimpleNamespace(**cfg.input_features['observation.state'])
        cfg.env_state_feature = None
        cfg.action_feature = SimpleNamespace(**cfg.output_features['action'])
        self.keys = list(cfg.image_features)
        if self.keys != [f'observation.images.{c}' for c in ('head', 'left_wrist', 'right_wrist')]:
            raise ValueError('Unexpected ACT camera order')
        if backend == 'pytorch':
            from act_core import ACT
            self.model = ACT(cfg)
            self.model.load_state_dict({k.removeprefix('model.'): v for k, v in load_file(checkpoint / 'model.safetensors').items()}, strict=True)
            self.model.to(device).eval()
        elif backend == 'tensorrt':
            if str(device) != 'cuda':raise ValueError('TensorRT requires CUDA')
            self.model = TensorRTForward(Path(engine_path) if engine_path else bundle/(checkpoint.name+'.engine'),manifest['model.safetensors'])
        else:
            raise ValueError('Unknown ACT inference backend')
        self.stats = {k: v.to(device) for k, v in load_file(checkpoint / 'policy_preprocessor_step_3_normalizer_processor.safetensors').items()}
        self.out = {k: v.to(device) for k, v in load_file(checkpoint / 'policy_postprocessor_step_0_unnormalizer_processor.safetensors').items()}

    @torch.inference_mode()
    def predict(self, observation):
        batch = {}
        for key, image in zip(self.keys, observation['images']):
            if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                raise ValueError('Expected three RGB uint8 640x480 images')
            value = torch.from_numpy(image).permute(2, 0, 1).contiguous().float().div(255).unsqueeze(0).to(self.device)
            batch[key] = (value - self.stats[key+'.mean']) / (self.stats[key+'.std'] + 1e-8)
        value = torch.from_numpy(observation['state']).float().unsqueeze(0).to(self.device)
        batch['observation.state'] = (value-self.stats['observation.state.mean']) / (self.stats['observation.state.std'] + 1e-8)
        batch['observation.images'] = [batch[key] for key in self.keys]
        normalized = self.model(batch)[0].float()
        result = (normalized * (self.out['action.std'] + 1e-8) + self.out['action.mean']).cpu().numpy()[0]
        if result.shape != (50, 16) or not np.isfinite(result).all():
            raise ValueError('Expected finite 50x16 ACT chunk')
        return result


def load(config, checkpoint_override=None):
    if config['model'] != 'act':
        raise ValueError('Jetson runtime supports reviewed ACT stages only')
    threads = config.get('inference_cpu_threads', 2)
    if type(threads) is not int or not 1 <= threads <= 4:
        raise ValueError('Inference CPU threads must be 1..4')
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    cv2.setNumThreads(1)
    checkpoint = Path(checkpoint_override or config['act_stage_checkpoints']['far']['path'])
    sha = hashlib.sha256((checkpoint / 'model.safetensors').read_bytes()).hexdigest()
    if sha != config['checkpoint_sha256']['act']:
        raise ValueError('Unreviewed ACT checkpoint')
    engine = ActEngine(checkpoint, config['inference_bundle'], backend=config.get('act_backend','pytorch'))
    return SimpleNamespace(joint_vector=joint_vector), engine.predict, lambda: None, sha
