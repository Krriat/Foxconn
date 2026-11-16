#!/usr/bin/env python3
"""Bounded GPU benchmark with live telemetry; never changes hardware limits."""
import argparse
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time


def read(path, divisor=1):
    try:
        return float(Path(path).read_text().strip()) / divisor
    except (OSError, ValueError, TypeError):
        return None


def discover():
    rails, temperatures, events = {}, {}, {}
    for hwmon in Path('/sys/class/hwmon').glob('hwmon*'):
        for label_file in hwmon.glob('in*_label'):
            label = label_file.read_text().strip()
            if label in ('VDD_GPU_SOC', 'VDD_CPU_CV', 'VIN_SYS_5V0'):
                index = label_file.name[2:].split('_')[0]
                rails[label] = (hwmon / f'in{index}_input', hwmon / f'curr{index}_input')
        for path in hwmon.glob('oc*_event_cnt'):
            events[path.name] = path
    for zone in Path('/sys/class/thermal').glob('thermal_zone*'):
        try:
            temperatures[zone.joinpath('type').read_text().strip()] = zone / 'temp'
        except OSError:
            pass
    return rails, temperatures, events


def sample(paths):
    rails, temperatures, events = paths
    values = {}
    for label, (v_path, i_path) in rails.items():
        voltage, current = read(v_path, 1000), read(i_path, 1000)
        values[label] = dict(voltage_v=voltage, current_a=current,
            power_w=voltage*current if voltage is not None and current is not None else None)
    return dict(unix_time=time.time(), monotonic_seconds=time.monotonic(), rails=values,
        temperatures_c={k: read(v, 1000) for k, v in temperatures.items()},
        overcurrent_events={k: read(v) for k, v in events.items()},
        gpu_frequency_hz=read('/sys/class/devfreq/17000000.gpu/cur_freq'))


def interrupted(signum, frame):
    raise InterruptedError(f'runner received signal {signum}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--temperature-stop-c', type=float, default=85.)
    parser.add_argument('--minimum-voltage-ratio', type=float, default=.90)
    parser.add_argument('--max-seconds', type=float, default=600.)
    parser.add_argument('--iterations', type=int, default=300)
    args = parser.parse_args()
    root = args.root.resolve()
    result = root / 'results'
    result.mkdir(exist_ok=True)
    for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(signum, interrupted)
    paths = discover()
    initial_temperatures = sample(paths)['temperatures_c']
    if not any(value is not None for value in initial_temperatures.values()):
        raise RuntimeError('Cannot read temperatures; guarded test not started')
    baseline = []
    for _ in range(5):
        voltage = sample(paths)['rails'].get('VDD_GPU_SOC', {}).get('voltage_v')
        if voltage is not None:
            baseline.append(voltage)
        time.sleep(.2)
    if not baseline:
        raise RuntimeError('Cannot read supply voltage; guarded test not started')
    baseline_voltage = statistics.median(baseline)
    print(json.dumps(dict(event='guard_configuration', baseline_voltage_v=baseline_voltage,
        stop_below_voltage_v=baseline_voltage*args.minimum_voltage_ratio,
        temperature_stop_c=args.temperature_stop_c, max_seconds=args.max_seconds)), flush=True)
    outcome = dict(state='starting', baseline_voltage_v=baseline_voltage)
    process = None
    with (result / 'benchmark.log').open('w') as output, (result / 'guard_telemetry.jsonl').open('w') as telemetry:
        try:
            process = subprocess.Popen(['bash', str(root / 'run.sh'), '--warmup', '50',
                '--iterations', str(args.iterations)], stdout=output, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
            start, low_voltage_readings = time.monotonic(), 0
            while process.poll() is None:
                record = sample(paths)
                record.update(event='telemetry', elapsed_seconds=time.monotonic()-start, benchmark_pid=process.pid)
                line = json.dumps(record)
                print(line, flush=True)
                telemetry.write(line+'\n')
                telemetry.flush()
                temperatures = [v for v in record['temperatures_c'].values() if v is not None]
                voltage = record['rails'].get('VDD_GPU_SOC', {}).get('voltage_v')
                low_voltage_readings = (low_voltage_readings+1 if voltage is not None and
                    voltage < baseline_voltage*args.minimum_voltage_ratio else 0)
                if temperatures and max(temperatures) >= args.temperature_stop_c:
                    raise RuntimeError('Thermal stop threshold reached')
                if low_voltage_readings >= 2:
                    raise RuntimeError('Voltage dropped below 90% of idle baseline twice')
                if time.monotonic()-start > args.max_seconds:
                    raise TimeoutError('Benchmark duration limit reached')
                time.sleep(.5)
            outcome = dict(state='complete' if process.returncode == 0 else 'benchmark_failed',
                returncode=process.returncode, elapsed_seconds=time.monotonic()-start)
        except BaseException as error:
            outcome = dict(state='stopped', reason=repr(error))
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            (result / 'guard_outcome.json').write_text(json.dumps(outcome, indent=2)+'\n')
            print(json.dumps(dict(event='guard_outcome', **outcome)), flush=True)
    return 0 if outcome['state'] == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
