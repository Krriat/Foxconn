"""Compare passive source/destination ROS logs without contacting hardware."""
import argparse
import collections
import json
import statistics
from pathlib import Path


def stats(v):
    v = sorted(v)
    if not v:
        return {'count': 0}
    return dict(count=len(v), median=statistics.median(v), p95=v[int(.95*(len(v)-1))],
                p99=v[int(.99*(len(v)-1))], maximum=v[-1], over_100=sum(x>100 for x in v),
                over_300=sum(x>300 for x in v))


def load(path):
    groups = collections.defaultdict(list)
    for line in path.open():
        r = json.loads(line)
        groups[r['topic']].append(r)
    return groups


def summary(groups):
    result = {}
    for topic, rows in groups.items():
        if not topic.startswith('/robot/') or not rows:
            continue
        duration = (rows[-1]['mono_ns']-rows[0]['mono_ns'])/1e9
        item = dict(messages=len(rows), duration_s=duration,
                    rate_hz=(len(rows)-1)/duration if duration else None,
                    receive_gap_ms=stats([(b['mono_ns']-a['mono_ns'])/1e6 for a,b in zip(rows,rows[1:])]))
        if 'stamp_ns' in rows[0]:
            stamps = [r['stamp_ns'] for r in rows]
            item.update(unique_stamps=len(set(stamps)), duplicate_stamps=len(stamps)-len(set(stamps)),
                        stamp_regressions=sum(b<a for a,b in zip(stamps,stamps[1:])),
                        source_stamp_gap_ms=stats([(b-a)/1e6 for a,b in zip(stamps,stamps[1:])]),
                        decode_errors=sum(r.get('decode_ok') is False for r in rows),
                        invalid_joints=sum(r.get('valid') is False for r in rows))
        elif '/status/robot_' in topic:
            item['first_state'] = rows[0]['data']
            item['last_state'] = rows[-1]['data']
        elif topic.endswith('/diagnostics'):
            first = {s['name']:s for s in rows[0]['data']}
            item['hardware'] = {}
            for s in rows[-1]['data']:
                v, before = s['values'], first[s['name']]['values']
                item['hardware'][s['name']] = dict(last=s, samples_delta=int(v['samples'])-int(before['samples']),
                                                   failures_delta=int(v['failures'])-int(before['failures']))
        elif topic.endswith('/policy/status'):
            item['phases'] = dict(collections.Counter(r['data'].get('phase') for r in rows))
        result[topic] = item
    return result


def compare(source, destination, topic):
    a, b = source.get(topic, []), destination.get(topic, [])
    if not a or not b:
        return dict(comparable=False, source_messages=len(a), destination_messages=len(b),
                    reason='No source or destination samples; a loss percentage cannot be inferred.')
    # Match exact acquisition stamps; trim boundaries to exclude discovery/start/stop skew.
    lo=max(min(r['stamp_ns'] for r in a),min(r['stamp_ns'] for r in b))+2_000_000_000
    hi=min(max(r['stamp_ns'] for r in a),max(r['stamp_ns'] for r in b))-2_000_000_000
    sa={r['stamp_ns']:r for r in a if lo<=r['stamp_ns']<=hi}
    sb={r['stamp_ns']:r for r in b if lo<=r['stamp_ns']<=hi}
    if not sa:
        return dict(comparable=False, reason='Insufficient overlapping samples')
    missing=sorted(sa.keys()-sb.keys())
    local_missing=sorted(sb.keys()-sa.keys())
    paired_deltas=[(sb[k]['wall_ns']-sa[k]['wall_ns'])/1e6 for k in sa.keys()&sb.keys()]
    return dict(comparable=True, common_window_s=(hi-lo)/1e9, source_unique=len(sa), destination_unique=len(sb),
                source_stamps_not_at_destination=len(missing), missing_stamps=missing,
                missing_fraction=len(missing)/len(sa), destination_stamps_not_at_source_observer=len(local_missing),
                paired_receipt_delta_ms=stats(paired_deltas),
                clock_note='Cross-host wall-clock deltas include clock offset; frame matching does not depend on clock synchronization.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    a=p.parse_args()
    logs={h:load(a.directory/f'{h}_initial.jsonl') for h in ('pc','jetson','umie')}
    routes={f'/robot/upperlimb/joint_states/{s}_arm':'jetson' for s in ('left','right')}
    routes.update({f'/robot/sensors/cameras/{c}/color/image_raw/compressed':
                   'jetson' if c=='head' else 'umie' for c in ('head','left_wrist','right_wrist')})
    report=dict(scope='Read-only arm feedback, RGB and relay telemetry; no actuator commands.',
                streams={h:summary(g) for h,g in logs.items()},
                source_to_pc={t:compare(logs[h],logs['pc'],t) for t,h in routes.items()},
                limitations=['Camera hardware sequence numbers are not published; source/destination timestamp matching checks delivered ROS messages.',
                             'The configured 30-Hz capture to 20-Hz publication is intentional downsampling, not inferred transport loss.',
                             'An unavailable stream is reported unavailable, not as a measured packet loss percentage.'])
    stable={h:{t:[r for r in rows if r['mono_ns']>=g['started'][0]['mono_ns']+30_000_000_000]
               for t,rows in g.items()} for h,g in logs.items()}
    report['after_initial_30_seconds']=dict(
        streams={h:summary(g) for h,g in stable.items()},
        source_to_pc={t:compare(stable[h],stable['pc'],t) for t,h in routes.items()})
    output=a.directory/'stream_transport_report.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'report':str(output),'source_to_pc':report['source_to_pc']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
