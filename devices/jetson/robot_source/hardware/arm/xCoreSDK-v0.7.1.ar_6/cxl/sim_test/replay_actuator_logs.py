#!/usr/bin/env python3
"""Replay original feedback through the mock guard; never loads the SDK.

This checks data/ordering and the stop latch, not physical stopping capability.
"""
import argparse
import csv
import itertools
from pathlib import Path
from common import read_json, write_json, digest
from protection import Protection

def replay(run):
    run=Path(run); meta=read_json(run/'metadata.json')
    known=meta.get('device_sample_identity_available',False)
    mismatches=[name for name,h in meta['raw_log_sha256'].items()
                if not (run/name).is_file() or digest(run/name)!=h]
    guard=Protection(); packets=0; observations_after_stop=0; incomplete_packets=0
    with (run/'samples.csv').open(newline='') as f:
        stream=csv.DictReader(f)
        for key,items in itertools.groupby(stream,lambda r:(r['trial_id'],r['sample_seq'],r['host_rx_ns'],r['arm'])):
            rows=list(items);packets+=1
            now=int(rows[0]['host_rx_ns'])
            if guard.state!='active': observations_after_stop+=1;continue
            if len(rows)!=7 or len(set(r['joint_name'] for r in rows))!=7:
                incomplete_packets+=1;guard.stop('incomplete_arm_packet',now);continue
            first=rows[0]
            try:
                q=[float(r['q_rad']) for r in rows];v=[float(r['qd_rad_s']) for r in rows]
            except ValueError: guard.stop('missing_feedback_value',now);continue
            mode=first['mode']
            if mode.startswith('read_only_idle_checked'): mode='normal'
            guard.feedback(int(first['sample_seq']) if known else None,
                           int(first['device_sample_ns']) if known and first['device_sample_ns'] else None,
                           now,q,v,mode,any(r['fault_flags'] or r['limit_flags'] for r in rows))
            if any(r.get('valid','1')=='0' for r in rows): guard.stop('recorded_invalid_sample',now)
    report=dict(offline_only=True,physical_stop_verified=False,device_identity_known=known,
                host_transport_sequence_not_used_as_device_sequence=not known,
                raw_hash_mismatches=mismatches,packets=packets,incomplete_packets=incomplete_packets,
                observations_after_stop=observations_after_stop,guard_state=guard.state,
                reason=guard.reason,events=guard.events,
                thresholds='mock guard defaults for regression only, not site-approved robot thresholds',
                limitation='No freshness proof without device ID; no physical execution, protection telemetry or stop confirmation')
    write_json(run/'replay_report.json',report)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    report=replay(a.run);print(a.run/'replay_report.json')
    return 1 if report['raw_hash_mismatches'] or report['reason'] else 0

if __name__=='__main__': raise SystemExit(main())
