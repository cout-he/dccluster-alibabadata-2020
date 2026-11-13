"""Analyze Alibaba PAI GPU v2020 raw tables with pandas and NumPy."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
DAY = 86400
UTC8 = 28800
WINDOWS = (5, 15, 60, 300, 900)
KEY = ['job_name', 'task_name']
QUANTILES = (0, .25, .5, .75, .9, .95, .99, .999, 1)
QLABELS = ('min', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99', 'p999', 'max')


def read(root, name, columns=None, **kwargs):
    names = (root / f'{name}.header').read_text().strip().split(',')
    return pd.read_csv(root / f'{name}.csv', names=names, usecols=columns, **kwargs)


def describe(values):
    clean = values[np.isfinite(values)]
    result = {'n': len(clean)}
    if len(clean):
        result.update(zip(QLABELS, np.quantile(clean, QUANTILES).tolist()))
        result.update(mean=float(clean.mean()), zero=float(np.mean(clean == 0)))
    return result


def intervals(times):
    diff = np.diff(np.sort(times))
    return describe(diff) | {f'lt{v}': float(np.mean(diff < v)) for v in (5, 15, 60)}


def add_events(target, starts, ends, weights=None):
    if weights is None:
        np.add.at(target, starts, 1)
        np.add.at(target, ends, -1)
    else:
        np.add.at(target, starts, weights)
        np.add.at(target, ends, -weights)


def encode_times(series):
    values = series.dropna().to_numpy()
    assert np.all(values == values.astype(np.int64))
    return values.astype(np.int64)


def time_slice(frame, left, right):
    return frame.start_time.lt(right) & frame.end_time.gt(left) & frame.end_time.gt(frame.start_time)


def window_profile(per_sec):
    metrics = {}
    baseline = per_sec.reshape(-1, 900).sum(axis=1)
    for width in WINDOWS:
        blocks = per_sec.reshape(-1, width).sum(axis=1)
        vals = describe(blocks)
        if width < 900:
            p = width / 900
            expected = np.repeat(baseline * p, 900 // width)
            expected_var = expected * (1 - p)
            conditional_empty = float(np.mean((1 - p) ** baseline))
            within_block_dispersion = float(np.sum((blocks - expected)**2) / expected_var.sum())
        else:
            conditional_empty = None
            within_block_dispersion = None
        vals.update(
            fano=float(blocks.var() / blocks.mean()),
            poisson_empty=float(np.exp(-blocks.mean())),
            within_block_empty=conditional_empty,
            within_block_dispersion=within_block_dispersion,
            singleton=float(np.mean(blocks == 1)),
            multi=float(np.mean(blocks >= 2)),
            peak_offset=int(blocks.argmax() * width),
        )
        assert blocks.sum() == per_sec.sum()
        metrics[str(width)] = vals
    return metrics


def concurrent_profile(series):
    summary = describe(series)
    for width in WINDOWS:
        buckets = series.reshape(-1, width)
        mean = buckets.mean(axis=1)
        peak = buckets.max(axis=1)
        positive = mean > 0
        summary[str(width)] = {
            'mean': describe(mean), 'peak': describe(peak),
            'peak_to_mean': describe(peak[positive] / mean[positive]),
            'sample_max': float(buckets[:, 0].max()),
        }
    return summary


def main(root, out):
    t = read(root, 'pai_task_table', KEY + ['start_time', 'end_time', 'status', 'inst_num', 'plan_cpu', 'plan_gpu', 'plan_mem'])
    idx = pd.MultiIndex.from_frame(t[KEY])
    assert idx.is_unique
    times = encode_times(t.start_time)
    left = (int(times.min()) + UTC8) // DAY * DAY + DAY - UTC8
    right = (int(times.max()) + UTC8) // DAY * DAY - UTC8
    span = right - left
    in_window = (times >= left) & (times < right)
    core = times[in_window]
    unique_core = np.unique(core)
    arrivals = np.bincount(core - left, minlength=span)
    job = read(root, 'pai_job_table', ['start_time'])
    jt = encode_times(job.start_time)
    jt = jt[(jt >= left) & (jt < right)]
    job_arrivals = np.bincount(jt - left, minlength=span)
    metrics = {
        'sample': {'task_rows': len(t), 'valid_start': len(times), 'core_tasks': len(core), 'core_jobs': len(jt),
                   'left': left, 'right': right, 'days': span//DAY, 'unique_seconds': len(unique_core)},
        'gaps': intervals(core), 'gaps_full': intervals(times), 'gaps_distinct': intervals(unique_core),
        'job_gaps': intervals(jt), 'window': window_profile(arrivals), 'job_window': window_profile(job_arrivals),
    }
    hours = arrivals.reshape(-1, 24, 3600).sum(axis=2)
    weekdays = (np.arange(span // DAY) + (left + UTC8) // DAY + 3) % 7
    hour_by_weekday = np.stack([hours[weekdays == i].mean(axis=0) for i in range(7)])
    minutes = arrivals.reshape(-1, 15, 60).sum(axis=2)
    block_sum = minutes.sum(axis=1)
    positive = block_sum > 0
    share = minutes[positive].max(axis=1) / block_sum[positive]
    five = arrivals.reshape(-1, 5).sum(axis=1)
    top_n = int(np.ceil(len(five) * .01))
    longest_pos = int(np.argmax(np.diff(unique_core)))
    busy = block_sum >= 30
    highest_share_block = np.flatnonzero(busy)[np.argmax(minutes[busy].max(axis=1) / block_sum[busy])]
    metrics['burst'] = {
        'peak5': int(five.max()), 'peak5_offset': int(five.argmax()*5),
        'top1_share': float(np.partition(five, -top_n)[-top_n:].sum()/len(core)),
        'largest_1min_share': describe(share),
        'largest_1min_share_busy': describe(minutes[busy].max(axis=1)/block_sum[busy]),
        'busy_windows': int(busy.sum()),
        'busy_half_in_1min': float(np.mean(minutes[busy].max(axis=1)*2 >= block_sum[busy])),
        'example_offset': int(highest_share_block*900),
        'example_minutes': minutes[highest_share_block].tolist(),
        'longest_gap': int(np.diff(unique_core)[longest_pos]),
        'longest_gap_offset': int(unique_core[longest_pos] - left),
        'peak_hour': int(hours.sum(axis=0).argmax()),
        'low_hour': int(hours.sum(axis=0).argmin()),
        'hour_ratio': float(hours.sum(axis=0).max()/hours.sum(axis=0).min()),
        'daily': describe(hours.sum(axis=1)),
        'empty_5_within_nonempty_15min': float(np.mean(five.reshape(-1,180)[positive] == 0)),
        'empty_5_phase': [float(np.mean(arrivals[i:span-5+i].reshape(-1,5).sum(axis=1)==0)) for i in range(5)],
    }
    # Check whether the unusually sparse first six complete days drive the result.
    later = arrivals[6*DAY:]
    metrics['later_days'] = {
        'task_count': int(later.sum()),
        'gaps': intervals(core[core >= left+6*DAY]),
        'window': window_profile(later),
    }

    task_active = np.zeros(span+1, dtype=np.int32)
    valid_task = time_slice(t, left, right)
    a = t.loc[valid_task, 'start_time'].clip(lower=left).to_numpy(np.int64) - left
    b = t.loc[valid_task, 'end_time'].clip(upper=right).to_numpy(np.int64) - left
    add_events(task_active, a, b)
    expected_task_seconds = int((b-a).sum())
    task_series = np.cumsum(task_active, dtype=np.int32)[:-1]
    assert task_series.sum(dtype=np.int64) == expected_task_seconds
    del task_active
    plan = t[['plan_cpu','plan_gpu']].to_numpy()
    cpu = np.zeros(span+1, dtype=np.int64)
    gpu = np.zeros_like(cpu)
    inst = np.zeros(span+1, dtype=np.int32)
    extra = np.zeros_like(inst)
    extra_cpu = np.zeros_like(cpu)
    extra_gpu = np.zeros_like(gpu)
    cpu_used = np.zeros(span+1)
    gpu_used = np.zeros(span+1)
    cpu_covered = np.zeros_like(inst)
    gpu_covered = np.zeros_like(inst)
    ns = len(t)
    scheduled = np.zeros(ns, dtype=np.int32)
    sensor_count = np.zeros_like(scheduled)
    fields = ['cpu_usage', 'gpu_wrk_util', 'avg_mem', 'avg_gpu_wrk_mem', 'network_read_gib', 'bandwidth_gb']
    task_values = np.zeros((ns,len(fields)))
    metric_count = np.zeros((ns,len(fields)), dtype=np.int32)
    resource_valid = np.zeros(2,dtype=np.int64)
    expected = np.zeros(3,dtype=np.int64)
    sensor_join_n = end_missing_n = instance_n = valid_n = 0
    s = read(root, 'pai_sensor_table', ['worker_name','cpu_usage','gpu_wrk_util','avg_mem','avg_gpu_wrk_mem','read','write']).set_index('worker_name', verify_integrity=True)
    s['network_read_gib'] = s['read'] / 2**30
    s['bandwidth_gb'] = (s.pop('read')+s.pop('write')) / 2**30
    for part in read(root, 'pai_instance_table', KEY + ['worker_name','machine','start_time','end_time','status'], chunksize=250_000):
        instance_n += len(part)
        host = part.machine.notna()
        known = time_slice(part,left,right) & host
        unknown = host & part.status.eq('Running') & part.end_time.isna() & part.start_time.lt(right)
        cols = KEY + ['worker_name','start_time','end_time']
        x = part.loc[known,cols]
        u = part.loc[unknown,cols]
        pos = idx.get_indexer(pd.MultiIndex.from_frame(x[KEY]))
        pos_extra = idx.get_indexer(pd.MultiIndex.from_frame(u[KEY]))
        assert (pos >= 0).all() and (pos_extra >= 0).all()
        a = x.start_time.clip(lower=left).to_numpy(np.int64) - left
        b = x.end_time.clip(upper=right).to_numpy(np.int64) - left
        a_extra = u.start_time.clip(lower=left).to_numpy(np.int64) - left
        b_extra = np.full(len(u),span,dtype=np.int64)
        end_missing_n += len(u)
        valid_n += len(x)
        add_events(inst,a,b)
        add_events(extra,a_extra,b_extra)
        expected[0] += (b-a).sum()
        for col, (target, extension) in enumerate(((cpu,extra_cpu),(gpu,extra_gpu))):
            wanted = plan[pos,col]
            ok = np.isfinite(wanted) & (wanted >= 0)
            resource_valid[col] += ok.sum()
            weights = np.rint(wanted[ok]*100).astype(np.int64)
            assert np.allclose(wanted[ok]*100,weights,rtol=0,atol=1e-6)
            add_events(target,a[ok],b[ok],weights)
            expected[col+1] += np.dot(b[ok]-a[ok],weights)
            alt = plan[pos_extra,col]
            valid_alt = np.isfinite(alt) & (alt >= 0)
            add_events(extension,a_extra[valid_alt],b_extra[valid_alt],np.rint(alt[valid_alt]*100).astype(np.int64))
        loc = s.index.get_indexer(x.worker_name)
        sensor_found = loc >= 0
        sensor_join_n += sensor_found.sum()
        np.add.at(scheduled,pos,1)
        np.add.at(sensor_count,pos[sensor_found],1)
        values = s.iloc[loc[sensor_found]][fields].to_numpy()
        for col in range(len(fields)):
            valid = np.isfinite(values[:,col]) & (values[:,col]>=0)
            selected = pos[sensor_found][valid]
            vals = values[valid,col]
            np.add.at(task_values[:,col],selected,vals)
            np.add.at(metric_count[:,col],selected,1)
            if col < 2:
                events = cpu_used if col == 0 else gpu_used
                coverage = cpu_covered if col == 0 else gpu_covered
                sa, sb = a[sensor_found][valid], b[sensor_found][valid]
                add_events(events,sa,sb,vals/100)
                add_events(coverage,sa,sb)
    del s, idx, plan
    inst = np.cumsum(inst,dtype=np.int32)[:-1]
    cpu = np.cumsum(cpu)[:-1]
    gpu = np.cumsum(gpu)[:-1]
    assert np.array_equal([inst.sum(dtype=np.int64),cpu.sum(),gpu.sum()],expected)
    cpu = cpu/10000
    gpu = gpu/10000
    extension = {'instance': np.cumsum(extra,dtype=np.int32)[:-1],
                 'cpu_request': np.cumsum(extra_cpu)[:-1]/10000,
                 'gpu_request': np.cumsum(extra_gpu)[:-1]/10000}
    cpu_used = np.maximum(0,np.cumsum(cpu_used)[:-1])
    gpu_used = np.maximum(0,np.cumsum(gpu_used)[:-1])
    cpu_covered = np.cumsum(cpu_covered,dtype=np.int32)[:-1]
    gpu_covered = np.cumsum(gpu_covered,dtype=np.int32)[:-1]
    machine = read(root, 'pai_machine_spec', ['cap_cpu','cap_gpu'])
    series = {'task': task_series,'instance': inst,'cpu_request': cpu,'gpu_request': gpu,'cpu_monitored': cpu_used,'gpu_monitored': gpu_used}
    assert all(v.min() >= 0 for v in series.values())
    metrics['concurrent'] = {key: concurrent_profile(v) for key,v in series.items()}
    metrics['concurrency_scope'] = {
        'task_intervals': int(valid_task.sum()), 'instance_rows': instance_n, 'usable_instance_intervals': valid_n,
        'sensor_matched': int(sensor_join_n), 'cpu_known': int(resource_valid[0]), 'gpu_known': int(resource_valid[1]),
        'cpu_cover_time': float(cpu_covered.sum()/inst.sum()), 'gpu_cover_time': float(gpu_covered.sum()/inst.sum()),
        'missing_end_running_scenario_intervals': end_missing_n,
        'extend_running': {k: describe(series[k]+v) for k,v in extension.items()},
        'gpu_inventory': int(machine.cap_gpu.sum()), 'cpu_inventory': int(machine.cap_cpu.sum()),
        'seconds_gpu_request_gt_inventory': int((gpu>machine.cap_gpu.sum()).sum()),
        'seconds_cpu_request_gt_inventory': int((cpu>machine.cap_cpu.sum()).sum()),
    }
    print('Concurrency and resource series complete', flush=True)

    duration = (t.end_time.to_numpy() - t.start_time.to_numpy()) / 60
    inside = t.start_time.between(left,right,inclusive='left').to_numpy()
    complete = inside & (duration > 0) & t.status.eq('Terminated').to_numpy()
    monitored = complete & (sensor_count > 0)
    fully = monitored & (scheduled == sensor_count) & (scheduled == t.inst_num.to_numpy())
    task_values[:,0:2] /= 100
    task_values[metric_count==0] = np.nan
    duration_groups = [duration < 1, (duration >= 1) & (duration < 15), (duration >= 15) & (duration < 60), duration >= 60]
    complete_requests = t[['plan_cpu','plan_gpu','plan_mem']].mul(t.inst_num,axis=0).to_numpy()
    complete_requests[:,0:2] /= 100
    shape = {
        'positive_duration': describe(duration[complete]),
        'duration_groups': [int((complete & g).sum()) for g in duration_groups],
        'monitored_tasks': int(monitored.sum()), 'full_recorded_sensor_tasks': int(fully.sum()),
        'duration_groups_sensor': [int((monitored & g).sum()) for g in duration_groups],
        'size': {'single': int((complete & t.inst_num.eq(1)).sum()),'multi': int((complete & t.inst_num.gt(1)).sum())},
        'observed': {}, 'planned': {},
    }
    grid = np.linspace(0,1,301)
    plot_values = {}
    for index, key in enumerate(fields):
        value = task_values[:,index]
        mask = monitored & np.isfinite(value)
        full_mask = fully & np.isfinite(value)
        shape['observed'][key] = describe(value[mask]) | {
            'rho_duration': float(spearmanr(duration[mask],value[mask]).statistic),
            'full_rho_duration': float(spearmanr(duration[full_mask],value[full_mask]).statistic),
            'short': describe(value[mask & duration_groups[0]]),
            'long': describe(value[mask & duration_groups[-1]]),
            'single': describe(value[mask & t.inst_num.eq(1).to_numpy()]),
            'multi': describe(value[mask & t.inst_num.gt(1).to_numpy()]),
        }
        plot_values[key] = np.quantile(value[mask],grid)
    for index, key in enumerate(('cpu_request','gpu_request','memory_request')):
        value = complete_requests[:,index]
        mask = complete & np.isfinite(value)
        shape['planned'][key] = describe(value[mask]) | {
            'rho_duration': float(spearmanr(duration[mask],value[mask]).statistic),
            'single': describe(value[mask & t.inst_num.eq(1).to_numpy()]),
            'multi': describe(value[mask & t.inst_num.gt(1).to_numpy()]),
        }
    valid_class = monitored & np.isfinite(task_values[:,0]) & np.isfinite(task_values[:,1])
    cpu_heavy = task_values[:,0] >= 4
    gpu_heavy = task_values[:,1] >= 1
    shape['classes'] = {
        'n': int(valid_class.sum()),
        'cpu_only': int((valid_class & cpu_heavy & ~gpu_heavy).sum()),
        'gpu_only': int((valid_class & gpu_heavy & ~cpu_heavy).sum()),
        'both': int((valid_class & cpu_heavy & gpu_heavy).sum()),
        'neither': int((valid_class & ~cpu_heavy & ~gpu_heavy).sum()),
    }
    metrics['heterogeneity'] = shape

    block_gpu = gpu.reshape(-1,900)
    # Show a 15-minute window with a large GPU request peak-to-mean ratio,
    # restricting the example to windows whose peak is at least the 95th percentile.
    peak = block_gpu.max(axis=1)
    candidates = np.flatnonzero(peak >= np.quantile(peak,.95))
    ratios = peak[candidates]/block_gpu[candidates].mean(axis=1)
    selected = int(candidates[np.argmax(ratios)])
    gpu_peak_index = int(gpu.argmax())
    metrics['resource_example'] = {
        'offset': selected*900, 'ratio': float(ratios.max()),
        'gpu_peak_offset': gpu_peak_index,
        'tasks_at_gpu_peak': int(task_series[gpu_peak_index]),
        'instances_at_gpu_peak': int(inst[gpu_peak_index]),
        'cpu_at_gpu_peak': float(cpu[gpu_peak_index]),
        'gpu_monitored_at_gpu_peak': float(gpu_used[gpu_peak_index]),
        'arrivals_gpu_spearman_5s': float(spearmanr(five,gpu.reshape(-1,5).mean(axis=1)).statistic),
        'arrivals_gpu_spearman_15m': float(spearmanr(block_sum,block_gpu.mean(axis=1)).statistic),
    }
    clip = slice(selected*900,(selected+1)*900)
    peak_bin = five.argmax() // 180
    peak_clip = slice(peak_bin*180,(peak_bin+1)*180)
    q = np.arange(0,61)
    gap = np.diff(np.sort(core))
    gap_unique = np.diff(unique_core)
    block_data = {f'{k}_{op}': getattr(v.reshape(-1,900),op)(axis=1)
                  for k,v in series.items() for op in ('mean','max')}
    np.savez_compressed(
        out/'figure_data.npz', q=q, gap_cdf=np.searchsorted(np.sort(gap),q,side='right')/len(gap),
        distinct_cdf=np.searchsorted(np.sort(gap_unique),q,side='right')/len(gap_unique),
        hour_by_weekday=hour_by_weekday, daily=hours.sum(axis=1),
        quantiles=grid, duration=np.quantile(duration[complete],grid),
        example_minutes=minutes[highest_share_block], peak_five=five[peak_clip],
        detail_task=task_series[clip], detail_instances=inst[clip], detail_gpu=gpu[clip],
        detail_cpu=cpu[clip], detail_gpu_monitored=gpu_used[clip],
        **plot_values, **block_data,
    )
    (out/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print('Wrote metrics.json and figure_data.npz',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=HERE.parent/'alibaba-gpu-v2020')
    args = parser.parse_args()
    main(args.data,HERE)
