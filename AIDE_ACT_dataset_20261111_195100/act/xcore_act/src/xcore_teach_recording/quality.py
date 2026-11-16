"""Read-only training diagnostics; task success still needs human review."""
import numpy as np


def episode_quality(meta, arrays, chunk_size=50):
    """Summarize a structurally validated keyboard-suction episode.

    Warnings never discard or change a demonstration. Constant suction labels
    may be intentional; report their coverage across the dataset separately.
    """
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    n, fps = meta["frames"], float(meta["fps"])
    lengths = [s["end"]-s["start"] for s in meta["segments"]]
    stamp = arrays["ros_timestamp"]
    span = float(stamp[-1]-stamp[0]) if n > 1 else 0.
    summary = meta.get("recording_summary", {})
    elapsed = float(summary.get("elapsed_seconds", max(span+1/fps, n/fps)))
    saved_fps = n/max(elapsed, 1/fps)
    warnings = []
    local = meta.get("capture_diagnostics",{}).get("local_joint_feedback")
    if local:
        initial = (local.get("at_record_start") or {}).get("streams",{})
        current = local.get("current",{})
        if current.get("error"):
            warnings.append("本机关节反馈故障："+str(current["error"]))
        for stream, counts in current.get("streams",{}).items():
            missing = counts.get("missing_sequences",0)-initial.get(stream,{}).get("missing_sequences",0)
            if missing>0:
                warnings.append(f"本机关节数据包序号缺失{missing}个（{stream}），请复核源采样连续性")
    if saved_fps < .9*fps:
        warnings.append(f"整段保存 {saved_fps:.2f}/{fps:g} 帧/秒，低于目标90%；检查等待和间断")
    full_chunks = sum(max(0, length-chunk_size+1) for length in lengths)
    if not full_chunks:
        warnings.append(f"没有长度达到 {chunk_size} 帧的连续片段，动作块会大量填充")
    trailing = [gap for gap in meta.get("capture_gaps", []) if gap.get("after_frame") == n]
    trailing_seconds = max((max(0., float(gap.get("end_elapsed_seconds") or elapsed)-
                                   float(gap["start_elapsed_seconds"])) for gap in trailing), default=0.)
    if trailing_seconds > .5:
        warnings.append(f"录制末尾 {trailing_seconds:.2f} 秒没有新帧，需确认任务是否录全")
    returns = [e for e in meta.get("control_events", []) if e.get("key") == "b"]
    if returns and returns[-1]["after_frame"] >= n:
        warnings.append("最后一次 B 回位请求后没有保存新帧，未录到该次回位")
    suction = {}
    for side, col in (("left", 14), ("right", 15)):
        observed, requested = arrays["state"][:, col], arrays["action"][:, col]
        suction[side] = dict(
            observation_values=[int(v) for v in np.unique(observed)],
            action_values=[int(v) for v in np.unique(requested)],
            observation_transitions=int(np.count_nonzero(np.diff(observed))),
            action_transitions=int(np.count_nonzero(np.diff(requested))),
            final_observed=int(observed[-1]), final_requested=int(requested[-1]))
        if observed[-1] != requested[-1]:
            warnings.append(f"{'左' if side == 'left' else '右'}吸盘末帧目标尚未由读回确认")
    if not any(v["action_transitions"] for v in suction.values()):
        warnings.append("本条没有录到吸盘开关指令变化；需确认是否覆盖所需操作")
    for side, offset in (("left", 0), ("right", 7)):
        if np.all(arrays["state"][:, offset:offset+7] == 0):
            warnings.append(f"{'左' if side == 'left' else '右'}臂七个关节全程为0，需核实是否与实际姿态一致")
    return dict(status="needs_review" if warnings else "checks_passed",
                frames=n, target_fps=fps, saved_fps=saved_fps,
                source_span_seconds=span,
                source_span_fps=(n-1)/span if span > 0 else None,
                segment_frames=lengths, chunk_size=chunk_size,
                full_chunk_start_frames=full_chunks,
                trailing_no_frame_seconds=trailing_seconds,
                suction=suction, warnings=warnings,
                task_success_requires_review=True)
