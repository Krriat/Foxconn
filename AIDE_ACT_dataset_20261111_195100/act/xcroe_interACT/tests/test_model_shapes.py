import torch

from xcore_interact.policy import InterACT3DConfig, InterACT3DPolicy


def test_interact3d_forward_and_backward():
    config = InterACT3DConfig(
        use_point_cloud=True,
        chunk_size=4,
        image_size=64,
        point_count=32,
        point_tokens=8,
        d_model=64,
        n_heads=4,
        dim_feedforward=128,
        encoder_blocks=1,
        pre_decoder_layers=1,
        post_decoder_layers=1,
        dropout=0.0,
        pretrained_backbone=False,
    )
    model = InterACT3DPolicy(config)
    images = torch.rand(1, 3, 3, 64, 64)
    points = torch.rand(1, 32, 6)
    points[..., 2] += 0.1
    state = torch.rand(1, 16)
    actions = torch.rand(1, 4, 16)
    result = model(images, points, state, actions, torch.zeros(1, 4, dtype=torch.bool))
    assert result["action"].shape == (1, 4, 16)
    assert torch.isfinite(result["loss"])
    result["loss"].backward()
    assert model.decoder.left_head.weight.grad is not None


def _rgb_config(**overrides):
    values = dict(chunk_size=4, image_size=64, d_model=64, n_heads=4,
                  dim_feedforward=128, encoder_blocks=1, pre_decoder_layers=1,
                  post_decoder_layers=1, dropout=0.0, pretrained_backbone=False)
    values.update(overrides)
    return InterACT3DConfig(**values)


def test_rgb_forward_backward_without_point_input():
    model = InterACT3DPolicy(_rgb_config())
    result = model(images=torch.rand(1, 3, 3, 64, 64), state=torch.rand(1, 16),
                   actions=torch.rand(1, 4, 16),
                   action_is_pad=torch.tensor([[False, False, True, True]]))
    assert model.geometry is None
    assert model.encoder.cls.shape[0] == 3
    assert result['action'].shape == (1, 4, 16)
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    assert model.decoder.left_head.weight.grad.abs().sum() > 0
    assert model.decoder.right_head.weight.grad.abs().sum() > 0


def test_suction_grouping_and_output_order():
    from unittest.mock import patch

    model = InterACT3DPolicy(_rgb_config(state_layout='dual_arm_suction')).eval()
    state = torch.tensor([[1., 2., 3., 4., 5., 6., 7.,
                           101., 102., 103., 104., 105., 106., 107., 1., 0.]])
    captured = {}
    left = model.encoder.left_state.register_forward_pre_hook(
        lambda _module, args: captured.update(left=args[0].detach().clone()))
    right = model.encoder.right_state.register_forward_pre_hook(
        lambda _module, args: captured.update(right=args[0].detach().clone()))
    by_arm = torch.arange(16.).reshape(1, 1, 16).expand(1, 4, 16)
    with patch.object(model.decoder, 'forward', return_value=by_arm):
        prediction = model.predict_action_chunk(torch.rand(1, 3, 3, 64, 64), state=state)
    left.remove()
    right.remove()
    torch.testing.assert_close(captured['left'].flatten(), torch.tensor([1., 2., 3., 4., 5., 6., 7., 1.]))
    torch.testing.assert_close(captured['right'].flatten(), torch.tensor([101., 102., 103., 104., 105., 106., 107., 0.]))
    torch.testing.assert_close(prediction[0, 0], torch.tensor([0., 1., 2., 3., 4., 5., 6., 8., 9., 10., 11., 12., 13., 14., 7., 15.]))


def test_rgb_checkpoint_round_trip(tmp_path):
    model = InterACT3DPolicy(_rgb_config(state_layout='dual_arm_suction')).eval()
    images, state = torch.rand(1, 3, 3, 64, 64), torch.rand(1, 16)
    with torch.no_grad():
        expected = model.predict_action_chunk(images, state=state)
    model.save_pretrained(tmp_path)
    restored = InterACT3DPolicy.from_pretrained(tmp_path).eval()
    assert not restored.config.use_point_cloud
    assert restored.config.state_layout == 'dual_arm_suction'
    with torch.no_grad():
        torch.testing.assert_close(restored.predict_action_chunk(images, state=state), expected)


def test_legacy_3d_checkpoint_round_trip(tmp_path):
    import json
    import pytest

    model = InterACT3DPolicy(_rgb_config(use_point_cloud=True, point_count=32, point_tokens=8)).eval()
    model.save_pretrained(tmp_path)
    path = tmp_path / 'config.json'
    config = json.loads(path.read_text())
    config.pop('use_point_cloud')
    config.pop('state_layout')
    path.write_text(json.dumps(config))
    restored = InterACT3DPolicy.from_pretrained(tmp_path).eval()
    assert restored.config.use_point_cloud
    assert restored.encoder.cls.shape[0] == 4
    with pytest.raises(ValueError, match='requires head point-cloud input'):
        restored.predict_action_chunk(torch.rand(1, 3, 3, 64, 64), state=torch.rand(1, 16))
    points = torch.rand(1, 32, 6)
    points[..., 2] += .1
    images, state = torch.rand(1, 3, 3, 64, 64), torch.rand(1, 16)
    with torch.no_grad():
        torch.testing.assert_close(restored.predict_action_chunk(images, points, state),
                                   model.predict_action_chunk(images, points, state))

