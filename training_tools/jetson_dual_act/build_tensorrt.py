"""Build one target-Orin ACT engine; IEEE FP32 is the default release path."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import re

import tensorrt as trt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--stage',choices=('far','near'),required=True)
    p.add_argument('--mode',choices=('ieee','mixed'),default='ieee')
    args=p.parse_args();source=args.bundle/(args.stage+'.onnx')
    logger=trt.Logger(trt.Logger.WARNING)
    builder=trt.Builder(logger)
    network=builder.create_network(0)
    parser=trt.OnnxParser(network,logger)
    if not parser.parse_from_file(str(source)):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config=builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,512*1024*1024)
    if args.mode=='ieee':
        config.clear_flag(trt.BuilderFlag.TF32)
        config.clear_flag(trt.BuilderFlag.FP16)
        config.builder_optimization_level=1
        config.max_aux_streams=0
    else:
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
    cache_path=args.bundle/('act_ieee_timing.cache' if args.mode=='ieee' else 'act_accurate_timing.cache')
    cache=config.create_timing_cache(cache_path.read_bytes() if cache_path.exists() else b'')
    config.set_timing_cache(cache,ignore_mismatch=False)
    # Preserve sensitive image boundaries, every Transformer layer and heads.
    # Middle ResNet blocks can use FP16; all external I/O stays FP32.
    for i in range(network.num_layers):
        layer=network.get_layer(i)
        sensitive=('backbone' not in layer.name or '/layer4/' in layer.name or
                   bool(re.search(r'/backbone(?:_\d+)?/conv1/',layer.name)))
        floating=[j for j in range(layer.num_outputs)
                  if layer.get_output(j) is not None and layer.get_output(j).dtype==trt.float32]
        if args.mode=='mixed' and sensitive and floating and layer.type not in (trt.LayerType.CONSTANT,trt.LayerType.SHAPE):
            layer.precision=trt.float32
            for j in floating:layer.set_output_type(j,trt.float32)
    started=time.monotonic()
    serialized=builder.build_serialized_network(network,config)
    if serialized is None:raise RuntimeError('TensorRT build failed')
    cache_path.write_bytes(bytes(config.get_timing_cache().serialize()))
    target=args.bundle/(args.stage+'.engine')
    target.write_bytes(bytes(serialized))
    manifest=json.loads((args.bundle/'onnx_manifest.json').read_text())[args.stage]
    manifest.update(tensorrt_version=trt.__version__,precision=('IEEE FP32, FP16/TF32 disabled' if args.mode=='ieee' else
        'FP16 middle CNN; FP32 first/last CNN, Transformer and heads'),
        engine_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),build_seconds=time.monotonic()-started)
    target.with_suffix('.engine.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
