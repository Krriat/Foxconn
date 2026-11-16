#!/usr/bin/env python3
"""Synthetic raw capture -> LeRobot videos -> ACT forward/backward/checkpoint.

Requires the LeRobot 0.4.3 training environment. Uses temporary synthetic data
and a small CPU ACT model, no backbone download, ROS or hardware. This checks
format/label/boundary compatibility, not learned robot task performance.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"recording"))
import record_xcore_teach_suction as recorder


def smoke():
    import torch
    import torchvision
    from torch.utils.data import default_collate
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.utils import dataset_to_policy_features
    from lerobot.configs.types import FeatureType
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import make_act_pre_post_processors

    if not hasattr(torchvision.io, "VideoReader"):
        raise RuntimeError("LeRobot 0.4.3 的 pyav 后端需要带 VideoReader 的 torchvision 0.21/0.22；"
                           f"当前版本 {torchvision.__version__} 不兼容")
    torch.set_num_threads(2)
    torch.manual_seed(7)
    with tempfile.TemporaryDirectory(prefix="teach-act-smoke-") as directory:
        root=Path(directory)
        raw=root/"raw"
        settings=recorder.dataset_schema(recorder.parse_args(["--width","64","--height","64"]))
        recorder.prepare_root(raw,settings)
        writer=recorder.EpisodeWriter(raw,"SYNTHETIC_FORMAT_TEST",settings)
        expected=[]
        try:
            writer.begin_segment()
            for i in range(70):
                if i == 35:
                    writer.end_segment("synthetic control boundary")
                    writer.begin_segment()
                t=100.+i/30
                action=np.r_[np.linspace(.1,1.4,14)+.01*np.sin(i/10),
                             int(10 <= i < 30),int(20 <= i < 50)].astype(np.float32)
                state=action.copy()
                state[14:]=[int(11 <= i < 31),int(21 <= i < 51)]
                images={key:np.full((64,64,3),[20+i,50+j*40,180-i],np.uint8)
                        for j,key in enumerate(recorder.CAMERAS)}
                writer.add(recorder.Bundle(t,state,action,images,np.full(5,t),t-.005,np.full(2,t-.015)))
                expected.append(action.copy())
            writer.end_segment("synthetic stop")
            writer.save()
        finally:
            if not writer.closed:
                writer.discard()
        destination=root/"lerobot"
        count=recorder.export_dataset(raw,destination,"local/synthetic_suction_test")
        assert count == 2
        dataset=LeRobotDataset("local/synthetic_suction_test",root=destination,video_backend="pyav",
                              delta_timestamps={"action":[i/30 for i in range(10)]})
        assert dataset.num_frames == 70 and dataset.num_episodes == 2
        assert dataset.features["action"]["names"][-2:] == list(recorder.STATE_NAMES[-2:])
        for index in (0,10,20,29,34,35,49,50,69):
            sample=dataset[index]
            np.testing.assert_array_equal(sample["action"][0].numpy(),expected[index])
            for key in recorder.CAMERAS:
                assert sample["observation.images."+key].shape == (3,64,64)
        boundary=dataset[34]
        assert boundary["action_is_pad"].tolist() == [False]+[True]*9
        for index in range(10):
            np.testing.assert_array_equal(boundary["action"][index].numpy(),expected[34])
        features=dataset_to_policy_features(dataset.features)
        config=ACTConfig(device="cpu",chunk_size=10,n_action_steps=1,
                         input_features={k:v for k,v in features.items() if v.type != FeatureType.ACTION},
                         output_features={k:v for k,v in features.items() if v.type == FeatureType.ACTION},
                         pretrained_backbone_weights=None,dim_model=64,n_heads=4,dim_feedforward=128,
                         n_encoder_layers=1,n_decoder_layers=1,n_vae_encoder_layers=1,latent_dim=8)
        policy=ACTPolicy(config)
        preprocessor,postprocessor=make_act_pre_post_processors(config,dataset.meta.stats)
        batch=default_collate([dataset[10],dataset[20]])
        normalized=preprocessor(batch)
        assert normalized["observation.state"].shape == (2,16)
        assert normalized["action"].shape == (2,10,16)
        assert torch.isfinite(normalized["action"]).all()
        optimizer=torch.optim.AdamW(policy.get_optim_params(),lr=1e-4)
        losses=[]
        suction_gradient=0.
        policy.train()
        for _ in range(2):
            optimizer.zero_grad()
            loss,_=policy(normalized)
            assert torch.isfinite(loss)
            loss.backward()
            suction_gradient=float(policy.model.action_head.weight.grad[-2:].abs().sum())
            assert np.isfinite(suction_gradient) and suction_gradient > 0
            optimizer.step()
            losses.append(float(loss.detach()))
        policy.eval()
        with torch.no_grad():
            output=postprocessor(policy.select_action(normalized))
        assert output.shape == (2,16) and torch.isfinite(output).all()
        checkpoint=root/"checkpoint"
        policy.save_pretrained(checkpoint)
        reloaded=ACTPolicy.from_pretrained(checkpoint,local_files_only=True)
        assert reloaded.config.output_features["action"].shape == (16,)
        result=dict(result="pass",raw_frames=70,lerobot_episodes=count,
                    three_video_streams_decoded=True,state_shape=[2,16],action_chunk_shape=[2,10,16],
                    binary_suction_labels_preserved=True,action_chunks_do_not_cross_segments=True,
                    normalization_finite=True,training_steps=2,losses=losses,
                    suction_head_gradient_sum=suction_gradient,checkpoint_reloaded=True,
                    model_device="cpu",model_size="small smoke model; no task training",
                    synthetic_data_only=True,robot_or_DO_commands=False,temporary_dataset_removed=True,
                    torch_version=torch.__version__,torchvision_version=torchvision.__version__)
        del dataset,writer,policy,reloaded,optimizer
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error("output already exists")
    result=smoke()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
