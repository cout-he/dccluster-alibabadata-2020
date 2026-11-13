"""Reproducible SustainCluster trace audit; writes only into this analysis folder.

Run from any directory: python analysis/workload_profile/workload_analysis.py
Dependencies: numpy, pandas, scipy, matplotlib, pyyaml. No original module is edited.
Arrival times reconstructed from the trace are launch-time proxies, not proven submissions.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import spearmanr
import yaml

FIELDS = ['job_name', 'start_time', 'end_time', 'start_dt', 'duration_min',
          'cpu_usage', 'gpu_wrk_util', 'avg_mem', 'avg_gpu_wrk_mem',
          'bandwidth_gb', 'weekday_name', 'weekday_num']
NUMERIC = ['start_time', 'end_time', 'duration_min', 'cpu_usage', 'gpu_wrk_util',
           'avg_mem', 'avg_gpu_wrk_mem', 'bandwidth_gb', 'weekday_num']
RESOURCES = ['tasks', 'cpu_usage', 'gpu_wrk_util', 'avg_mem', 'avg_gpu_wrk_mem']
LABELS = {'duration_min': '运行时间（分钟）', 'cpu_usage': 'CPU使用量（100 = 1核）',
          'gpu_wrk_util': 'GPU使用量（100 = 1 GPU等效量）', 'avg_mem': '主存使用量（GB）',
          'avg_gpu_wrk_mem': '显存使用量（GB）', 'bandwidth_gb': '网络量估计（GiB）',
          'tasks': '并发记录数', 'cpu_cores': 'CPU（核等效量）',
          'gpu_equiv': 'GPU（等效量）', 'memory_gb': '主存（GB）'}
OFFICIAL = 'https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2020/README.md'
DAY = 86400.


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(frame, path):
    pd.DataFrame(frame).to_csv(path, index=False, encoding='utf-8-sig', float_format='%.10g')


def describe(x):
    x = np.asarray(x, dtype=float)
    good = x[np.isfinite(x)]
    if not len(good):
        return {'count': 0, 'missing': len(x)}
    q = np.quantile(good, [.25, .5, .75, .9, .95, .99])
    return dict(count=len(good), missing=len(x)-len(good), minimum=good.min(),
                maximum=good.max(), mean=good.mean(), median=q[1], p25=q[0],
                p50=q[1], p75=q[2], p90=q[3], p95=q[4], p99=q[5],
                std=good.std(ddof=1) if len(good)>1 else 0.,
                zero_fraction=(good==0).mean(), negative_fraction=(good<0).mean())


def arrival_counts(starts, width, horizon):
    """UTC-midnight anchored, half-open bins; empty bins included."""
    starts = np.asarray(starts)
    assert np.all((starts >= 0) & (starts < horizon))
    return np.bincount((starts // width).astype(int), minlength=int(math.ceil(horizon/width)))


def event_sweep(starts, duration_s, weights, horizon, periodic=False):
    """Exact piecewise-constant half-open demand, clipped to [0, horizon).

    Same-time starts and ends are combined BEFORE observing the right-hand value.
    Periodic mode adds all prior copies that can overlap the observation window.
    Return segment left edges, demands, and actual segment lengths in seconds.
    """
    starts = np.asarray(starts, dtype=float)
    duration_s = np.asarray(duration_s, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim == 1:
        weights = weights[:, None]
    assert len(starts) == len(duration_s) == len(weights)
    assert np.isfinite(starts).all() and np.isfinite(duration_s).all()
    assert (duration_s > 0).all() and np.isfinite(weights).all() and (weights >= 0).all()
    assert horizon > 0 and np.all((starts >= 0) & (starts < horizon))
    copies = int(math.ceil(duration_s.max(initial=0)/horizon)) if periodic else 0
    ss, ee, ww = [], [], []
    for back in range(copies+1):
        a, b = starts-back*horizon, starts-back*horizon+duration_s
        keep = (b>0) & (a<horizon)
        ss.append(np.maximum(a[keep], 0))
        ee.append(np.minimum(b[keep], horizon))
        ww.append(weights[keep])
    s, e, w = np.concatenate(ss), np.concatenate(ee), np.concatenate(ww)
    ev = np.concatenate(([0., horizon], s, e))
    delta = np.vstack((np.zeros((2, weights.shape[1])), w, -w))
    order = np.argsort(ev, kind='stable')
    times, first = np.unique(ev[order], return_index=True)
    values = np.cumsum(np.add.reduceat(delta[order], first, axis=0), axis=0)[:-1]
    # Only remove floating-point cancellation noise, never positive observations.
    values[np.abs(values) < 1e-8] = 0.
    assert values.min(initial=0) > -1e-6
    return times[:-1], values, np.diff(times)


def weighted_summary(times, values, seconds):
    """Quantile = inverse time CDF, not a quantile of event rows."""
    order = np.argsort(values, kind='stable')
    cdf = np.cumsum(seconds[order]) / seconds.sum()
    q = values[order][np.searchsorted(cdf, [.5,.9,.95,.99], side='left')]
    return dict(mean=float(np.dot(values,seconds)/seconds.sum()), p50=q[0],
                p90=q[1], p95=q[2], p99=q[3], maximum=values.max(initial=0),
                peak_seconds=times[int(np.argmax(values))],
                zero_time_fraction=seconds[values==0].sum()/seconds.sum())


def source_audit(root):
    """Extract actual literal call arguments, failing rather than silently assuming defaults."""
    tree = ast.parse((root/'simulation/cluster_manager.py').read_text(encoding='utf-8-sig'))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id=='extract_tasks_from_row']
    assert len(calls)==1, 'Review changed extraction call chain before running.'
    args = {k.arg: ast.literal_eval(k.value) for k in calls[0].keywords
            if k.arg in ['scale','task_scale','group_size']}
    assert args['scale']==1 and args['group_size']==1, 'Augmentation/grouping needs separate modeling.'
    return args


def load_and_audit(root, out, quality):
    d = pd.read_pickle(root/'data/workload/alibaba_2020_dataset/result_df_full_year_2020.pkl')
    assert isinstance(d,pd.DataFrame) and list(d.columns)==['interval_15m','tasks_matrix']
    d = d.sort_values('interval_15m').reset_index(drop=True)
    assert str(d.interval_15m.dt.tz)=='UTC'
    assert d.interval_15m.is_unique
    assert d.interval_15m.eq(d.interval_15m.dt.floor('15min')).all()
    assert all(isinstance(x,np.ndarray) and x.ndim==2 and x.shape[1]==12 for x in d.tasks_matrix)
    origin = d.interval_15m.min()
    period = pd.Timedelta(days=49)
    horizon = period.total_seconds()
    blocks = ((d.interval_15m-origin)//period).astype(int)
    base = d[blocks==0].copy()
    ref = {int((r.interval_15m-origin).total_seconds()): r.tasks_matrix
           for r in base.itertuples()}
    cycle_rows=[]
    for k, block in d.groupby(blocks):
        same_position=True
        equal_count=0
        for r in block.itertuples():
            rel = int((r.interval_15m-origin-k*period).total_seconds())
            if rel not in ref:
                same_position=False
                continue
            # DataFrame.equals considers colocated NaN equal; compares every cell and order.
            equal_count += pd.DataFrame(r.tasks_matrix).equals(pd.DataFrame(ref[rel]))
        complete = len(block)==len(base) and same_position
        cycle_rows.append(dict(cycle=int(k), start=str(origin+k*period),
                               end_exclusive=str(origin+(k+1)*period),
                               interval_rows=len(block), records=sum(map(len,block.tasks_matrix)),
                               identical_matrix_rows=equal_count, same_positions=complete,
                               exactly_equal_to_base=complete and equal_count==len(base)))
    cycles=pd.DataFrame(cycle_rows)
    assert cycles.exactly_equal_to_base.all(), '49-day repetition not confirmed; reassess sample definition.'
    write_csv(cycles,out/'cycle_comparison.csv')
    n=sum(map(len,d.tasks_matrix))
    quality.extend([dict(scope='full',field='records',check='total',value=n),
                    dict(scope='full',field='interval_15m',check='rows',value=len(d)),
                    dict(scope='full',field='interval_15m',check='nonempty_rows',value=sum(len(a)>0 for a in d.tasks_matrix)),
                    dict(scope='full',field='interval_15m',check='missing_15min_slots',value=int((d.interval_15m.max()-origin).total_seconds()/900+1)-len(d)),
                    dict(scope='full',field='cycles',check='identical_49day_blocks',value=len(cycles)),
                    dict(scope='full',field='interval_15m',check='duplicate_timestamps',value=d.interval_15m.duplicated().sum()),
                    dict(scope='full',field='interval_15m',check='coverage_start',value=str(origin)),
                    dict(scope='full',field='interval_15m',check='coverage_end_exclusive',value=str(d.interval_15m.max()+pd.Timedelta(minutes=15)))])
    a=pd.DataFrame(np.concatenate(base.tasks_matrix.to_list()),columns=FIELDS)
    for col in NUMERIC:
        a[col]=pd.to_numeric(a[col],errors='coerce')
    a['start_dt']=pd.to_datetime(a.start_dt,utc=True)
    a['interval_15m']=pd.DatetimeIndex(np.repeat(base.interval_15m.to_numpy(),base.tasks_matrix.map(len)))
    a['offset_seconds']=(a.start_dt-a.start_dt.dt.floor('15min')).dt.total_seconds()
    a['arrival_exact']=a.interval_15m+pd.to_timedelta(a.offset_seconds,unit='s')
    a['arrival_sim']=a.interval_15m
    a['task_span_min']=(a.end_time-a.start_time)/60
    a['duration_minus_span_min']=a.duration_min-a.task_span_min
    # NaN-aware full/base checks: verified identical cycles imply exact multiplicity.
    for col in FIELDS:
        vals=a[col]
        checks={'missing':int(vals.isna().sum())}
        if col in NUMERIC:
            x=vals.to_numpy(dtype=float)
            checks.update(zero=int((x==0).sum()),negative=int((x<0).sum()),
                          infinity=int(np.isinf(x).sum()))
            lo,hi=np.nanquantile(x,[.25,.75]); spread=hi-lo
            checks['outside_1p5_IQR_flag_only']=int(((x<lo-1.5*spread)|(x>hi+1.5*spread)).sum())
        for check,value in checks.items():
            quality.append(dict(scope='base',field=col,check=check,value=value))
            quality.append(dict(scope='full',field=col,check=check,value=value*len(cycles)))
    features=['duration_min','cpu_usage','gpu_wrk_util','avg_mem','avg_gpu_wrk_mem','bandwidth_gb']
    for scope,mult in [('base',1),('full',len(cycles))]:
        quality.extend([dict(scope=scope,field='records',check='total',value=len(a)*mult),
                        dict(scope=scope,field='job_name',check='unique_jobs',value=a.job_name.nunique()),
                        dict(scope=scope,field='job_name',check='duplicate_rows_beyond_first',value=len(a)*mult-a.job_name.nunique()),
                        dict(scope=scope,field='all_12_fields',check='duplicate_rows_beyond_first',value=len(a)*mult-len(a[FIELDS].drop_duplicates())),
                        dict(scope=scope,field='six_numeric_features',check='duplicate_rows_beyond_first',value=len(a)*mult-len(a[features].drop_duplicates()))])
    a['start_dt_minus_unix_s']=(a.start_dt-pd.to_datetime(a.start_time,unit='s',utc=True)).dt.total_seconds()
    a['weekday_num_mismatch']=a.weekday_num.ne(a.arrival_exact.dt.weekday)
    a['weekday_name_mismatch']=a.weekday_name.ne(a.arrival_exact.dt.day_name())
    a['inner_weekday_num_mismatch']=a.weekday_num.ne(a.start_dt.dt.weekday)
    # Outer bins must equal the original floor after one constant epoch translation.
    a['bin_epoch_translation_s']=(a.interval_15m-a.start_dt.dt.floor('15min')).dt.total_seconds()
    for col in ['weekday_num_mismatch','weekday_name_mismatch','inner_weekday_num_mismatch']:
        quality.append(dict(scope='base',field=col,check='true_rows',value=int(a[col].sum())))
        quality.append(dict(scope='full',field=col,check='true_rows',value=int(a[col].sum())*len(cycles)))
    for col in ['start_dt_minus_unix_s','bin_epoch_translation_s']:
        for value,count in a[col].value_counts(dropna=False).items():
            quality.append(dict(scope='base',field=col,check=f'offset_{value}_seconds_rows',value=count))
    assert (a.offset_seconds>=0).all() and (a.offset_seconds<900).all()
    assert a.bin_epoch_translation_s.nunique()==1
    assert np.isfinite(a[features].to_numpy(dtype=float)).all()
    assert (a.duration_min>0).all() and (a[features[1:]]>=0).all().all()
    return d,a,origin,horizon,cycles


def origins_for_seed(a,configs,seed):
    """Exact probabilities and RandomState inverse-CDF sampling used by np.random.choice.

    Seed describes an isolated assignment RNG. Full simulator interleaves unrelated
    np.random calls, so this is rule-faithful Monte Carlo, not a full-run seed replay.
    """
    hours=a.arrival_sim.dt.hour.to_numpy()
    minute=a.arrival_sim.dt.minute.to_numpy()/60
    score=np.column_stack([c.get('population_weight',.1)*np.where(
        (((hours+minute+c.get('timezone_shift',0))%24)>=8)&
        (((hours+minute+c.get('timezone_shift',0))%24)<20),1.,.3) for c in configs])
    probabilities=score/score.sum(axis=1,keepdims=True)
    rng=np.random.RandomState(seed)
    return (rng.random_sample(len(a))[:,None]>probabilities.cumsum(axis=1)).sum(axis=1)


def validate_origin_rule(root,a,configs):
    """Compare analytical sampling with actual unmodified repository function."""
    from types import SimpleNamespace
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(root))
    from utils.workload_utils import assign_task_origins, extract_tasks_from_row
    sample=a.iloc[:256]
    # Multiple actual bins ensure all relevant local hours are covered separately.
    for hour in range(24):
        stamp=pd.Timestamp('2020-01-01',tz='UTC')+pd.Timedelta(hours=hour)
        tasks=[SimpleNamespace(origin_dc_id=None) for _ in range(256)]
        state=np.random.get_state()
        np.random.seed(137)
        assign_task_origins(tasks,configs,stamp)
        np.random.set_state(state)
        z=sample.copy(); z['arrival_sim']=stamp
        got=origins_for_seed(z,configs,137)
        np.testing.assert_array_equal([t.origin_dc_id for t in tasks],[configs[k]['dc_id'] for k in got])
    row=pd.Series({'tasks_matrix':a.iloc[:2][FIELDS].to_numpy()})
    tasks=extract_tasks_from_row(row,current_time_utc=a.arrival_sim.iloc[0],task_scale=5,scale=1,group_size=1)
    for i,t in enumerate(tasks):
        np.testing.assert_allclose([t.cores_req,t.gpu_req,t.mem_req,t.bandwidth_gb],
                                   [a.cpu_usage.iloc[i]*.05,a.gpu_wrk_util.iloc[i]*.05,
                                    a.avg_mem.iloc[i]*5,a.avg_gpu_wrk_mem.iloc[i]])


def mdtable(frame, columns=None, maxrows=30):
    if columns is not None:
        frame=frame[columns]
    frame=frame.head(maxrows)
    def cell(x):
        if isinstance(x,(float,np.floating)):
            return f'{x:,.4g}'
        return str(x).replace('|','/')
    return '| '+' | '.join(map(str,frame.columns))+' |\n| '+' | '.join(['---']*len(frame.columns))+' |\n'+''.join(
        '| '+' | '.join(cell(v) for v in row)+' |\n' for row in frame.itertuples(index=False,name=None))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',type=int,default=10,help='Number of isolated origin RNG seeds, starting at zero')
    parser.add_argument('--skip-figures',action='store_true')
    ns=parser.parse_args()
    assert ns.seeds>=2
    out=Path(__file__).resolve().parent
    root=out.parent.parent
    figdir=out/'figures'; figdir.mkdir(exist_ok=True)
    protected=['data/workload/alibaba_2020_dataset/result_df_full_year_2020.pkl',
               'utils/workload_utils.py','rl_components/task.py',
               'data/workload/alibaba_2020_dataset/extract_dataset_from_dfas.py',
               'data/workload/alibaba_2020_dataset/alibaba_utils.py',
               'data/workload/alibaba_2020_dataset/alibaba_analysis.ipynb',
               'simulation/cluster_manager.py','configs/env/datacenters.yaml','configs/env/sim_config.yaml',
               'envs/task_scheduling_env.py','data/network_cost/network_delay.py']
    before={p:sha256(root/p) for p in protected}
    args=source_audit(root)
    configs=yaml.safe_load((root/'configs/env/datacenters.yaml').read_text(encoding='utf8'))['datacenters']
    sim=yaml.safe_load((root/'configs/env/sim_config.yaml').read_text(encoding='utf8'))['simulation']
    quality=[]
    print('Auditing all 49-day blocks...',flush=True)
    full,a,origin,horizon,cycles=load_and_audit(root,out,quality)
    print(f'Confirmed {len(cycles)} identical cycles; {len(a):,} base-cycle records.',flush=True)
    write_csv(quality,out/'data_quality.csv')
    validate_origin_rule(root,a,configs)
    metrics=['duration_min','cpu_usage','gpu_wrk_util','avg_mem','avg_gpu_wrk_mem','bandwidth_gb',
             'task_span_min','duration_minus_span_min']
    taskrows=[dict(scenario='raw',metric=m,**describe(a[m])) for m in metrics]
    scenarios=[('task_scale_1',1),('task_scale_2',2),('task_scale_5',5),('actual_config',args['task_scale'])]
    mapping={'cpu_cores':('cpu_usage',.01),'gpu_equiv':('gpu_wrk_util',.01),'memory_gb':('avg_mem',1.)}
    for name,scale in scenarios:
        for metric,(col,factor) in mapping.items():
            taskrows.append(dict(scenario=name,task_scale=scale,metric=metric,**describe(a[col]*factor*scale)))
    write_csv(taskrows,out/'task_statistics.csv')
    corrcols=metrics[:6]
    pearson=a[corrcols].corr(method='pearson'); spearman=a[corrcols].corr(method='spearman')
    pearson.rename_axis('metric').reset_index().to_csv(out/'correlation_pearson.csv',index=False,encoding='utf-8-sig')
    spearman.rename_axis('metric').reset_index().to_csv(out/'correlation_spearman.csv',index=False,encoding='utf-8-sig')
    # Explicit empirical rank rules, without invented utilization thresholds.
    ranks=a[['cpu_usage','gpu_wrk_util','avg_mem']].rank(method='average',pct=True)
    dominant=ranks.idxmax(axis=1)
    ties=ranks.eq(ranks.max(axis=1),axis=0).sum(axis=1)>1
    dominant[ties]='tie'
    composition=dominant.value_counts().rename_axis('dominant_relative_rank').reset_index(name='records')
    composition['fraction']=composition.records/len(a)
    write_csv(composition,out/'resource_composition.csv')
    q1,q3=a.duration_min.quantile([.25,.75])
    runtimeclasses=pd.DataFrame([
        dict(class_name='short_empirical_Q1',rule=f'duration <= {q1}',records=int((a.duration_min<=q1).sum())),
        dict(class_name='middle_Q1_Q3',rule=f'{q1} < duration <= {q3}',records=int(((a.duration_min>q1)&(a.duration_min<=q3)).sum())),
        dict(class_name='long_empirical_Q3',rule=f'duration > {q3}',records=int((a.duration_min>q3).sum()))])
    runtimeclasses['fraction']=runtimeclasses.records/len(a)
    write_csv(runtimeclasses,out/'runtime_classes.csv')
    # Data-size audit: same resource demands, different network size / cost / energy.
    bwrows=[]
    for name,col in [('current_index_8','avg_gpu_wrk_mem'),('correct_index_9','bandwidth_gb')]:
        bwrows.append(dict(mapping=name,field=col,total_network_value=a[col].sum(),
                           transmission_energy_kwh_if_all_remote=a[col].sum()*.06,**describe(a[col])))
    write_csv(bwrows,out/'bandwidth_mapping_comparison.csv')
    ratio=a.avg_gpu_wrk_mem/a.bandwidth_gb
    write_csv([dict(metric='current_to_correct_per_record_ratio',**describe(ratio)),
               dict(metric='aggregate_ratio',mean=a.avg_gpu_wrk_mem.sum()/a.bandwidth_gb.sum()),
               dict(metric='current_underestimates_fraction',mean=float((ratio<1).mean())),
               dict(metric='current_overestimates_fraction',mean=float((ratio>1).mean()))],out/'bandwidth_mapping_ratios.csv')
    starts={mode:(a[f'arrival_{mode}']-origin).dt.total_seconds().to_numpy() for mode in ['exact','sim']}
    arrivals=[]; intervals=[]; distributions=[]; profiles=[]; count_cache={}
    for mode,ss in starts.items():
        for width in [5,15,60,300,600,900]:
            counts=arrival_counts(ss,width,horizon)
            count_cache[mode,width]=counts
            assert counts.sum()==len(a)
            stat=describe(counts)
            stat.update(fano=counts.var(ddof=1)/counts.mean(),cv=counts.std(ddof=1)/counts.mean(),
                        peak_to_mean=counts.max()/counts.mean())
            arrivals.append(dict(arrival_mode=mode,bin_seconds=width,bins=len(counts),**stat))
            vals,freq=np.unique(counts,return_counts=True)
            distributions.extend(dict(arrival_mode=mode,bin_seconds=width,tasks_in_bin=int(v),
                                      bin_count=int(f),fraction=f/len(counts)) for v,f in zip(vals,freq))
        gaps=np.diff(np.sort(ss))
        g=describe(gaps); cv=gaps.std(ddof=1)/gaps.mean()
        intervals.append(dict(arrival_mode=mode,**g,cv=cv,burstiness=(cv-1)/(cv+1),
                              below_5s=(gaps<5).mean(),below_15s=(gaps<15).mean(),below_60s=(gaps<60).mean(),
                              simultaneous_fraction=(gaps==0).mean()))
        dates=a[f'arrival_{mode}']
        for kind,key,exposure in [('hour',dates.dt.hour,np.full(24,49)),
                                  ('weekday',dates.dt.weekday,np.full(7,7))]:
            values=key.value_counts().reindex(range(len(exposure)),fill_value=0)
            profiles.extend(dict(arrival_mode=mode,profile=kind,key=k,records=int(v),
                                 exposure_days=int(exposure[k]),mean_per_exposure_day=v/exposure[k])
                            for k,v in values.items())
        daily=dates.dt.floor('D').value_counts().sort_index().reindex(pd.date_range(origin,periods=49,freq='D'),fill_value=0)
        profiles.extend(dict(arrival_mode=mode,profile='date',key=str(k.date()),records=int(v),
                             exposure_days=1,mean_per_exposure_day=v) for k,v in daily.items())
    write_csv(arrivals,out/'arrival_statistics.csv'); write_csv(intervals,out/'interarrival_statistics.csv')
    write_csv(distributions,out/'arrival_count_distribution.csv'); write_csv(profiles,out/'arrival_calendar_profiles.csv')
    write_csv(pd.DataFrame({'time_utc':pd.date_range(origin,periods=int(horizon/900),freq='15min'),
                            'arrival_count':count_cache['exact',900]}),out/'arrival_timeseries_15min.csv')
    raw=np.column_stack([np.ones(len(a)),a[['cpu_usage','gpu_wrk_util','avg_mem','avg_gpu_wrk_mem']].to_numpy()])
    duration=a.duration_min.to_numpy()*60
    concurrency=[]; sweep_cache={}; integrity=[]
    print('Sweeping exact and batch arrivals, cold-start and periodic steady state...',flush=True)
    for mode,ss in starts.items():
        for boundary in ['cold_start','periodic']:
            t,v,w=event_sweep(ss,duration,raw,horizon,periodic=boundary=='periodic')
            sweep_cache[mode,boundary]=(t,v,w)
            expected=np.einsum('ij,i->j',raw,duration if boundary=='periodic' else np.minimum(duration,horizon-ss))
            observed=np.einsum('ij,i->j',v,w)
            np.testing.assert_allclose(observed,expected,rtol=2e-9,atol=1e-3)
            integrity.append(dict(check=f'integral_{mode}_{boundary}',max_relative_error=float(np.max(np.abs(observed-expected)/np.maximum(1,expected)))))
            for j,col in enumerate(RESOURCES):
                stat=weighted_summary(t,v[:,j],w)
                stat['peak_time_utc']=str(origin+pd.Timedelta(seconds=stat['peak_seconds']))
                concurrency.append(dict(arrival_mode=mode,boundary=boundary,scenario='raw',metric=col,**stat))
            for name,scale in scenarios:
                for metric,(col,factor) in mapping.items():
                    j=RESOURCES.index(col)
                    stat=weighted_summary(t,v[:,j]*factor*scale,w)
                    stat['peak_time_utc']=str(origin+pd.Timedelta(seconds=stat['peak_seconds']))
                    capkey={'cpu_cores':'total_cores','gpu_equiv':'total_gpus','memory_gb':'total_mem'}[metric]
                    current_cap=sum(c[capkey] for c in configs)
                    demand=v[:,j]*factor*scale
                    stat['current_capacity']=current_cap
                    stat['current_mean_utilization']=stat['mean']/current_cap
                    stat['over_current_capacity_time_fraction']=w[demand>current_cap].sum()/w.sum()
                    stat['over_80pct_current_capacity_time_fraction']=w[demand>.8*current_cap].sum()/w.sum()
                    concurrency.append(dict(arrival_mode=mode,boundary=boundary,scenario=name,metric=metric,**stat))
            # Export display means AND exact maxima in each 15-min display window.
            ts=segment_bins(t,v,w,900,horizon)
            frame=pd.DataFrame({'time_utc':pd.date_range(origin,periods=len(ts),freq='15min')})
            for j,col in enumerate(RESOURCES):
                frame[col+'_mean']=ts[:,j,0]
                frame[col+'_max']=ts[:,j,1]
            write_csv(frame,out/f'concurrency_timeseries_{mode}_{boundary}.csv')
    cc=pd.DataFrame(concurrency)
    write_csv(cc,out/'concurrency_statistics.csv')
    globalrows=[]
    caps={k:sum(c[key] for c in configs) for k,key in [('cpu_cores','total_cores'),('gpu_equiv','total_gpus'),('memory_gb','total_mem')]}
    for row in cc[(cc.boundary=='periodic')&(cc.scenario!='raw')].to_dict('records'):
        for q in ['p95','p99','maximum']:
            for util in [.7,.8,.9]:
                cap=row[q]/util
                globalrows.append(dict(arrival_mode=row['arrival_mode'],boundary='periodic',scenario=row['scenario'],
                                       metric=row['metric'],quantile=q,target_utilization=util,demand=row[q],
                                       required_capacity=cap,ceil_capacity=math.ceil(cap),current_capacity=caps[row['metric']],
                                       reserve_vs_demand=cap-row[q],current_minus_required=caps[row['metric']]-cap,
                                       current_demand_ratio=row[q]/caps[row['metric']]))
    write_csv(globalrows,out/'global_capacity.csv')
    print(f'Assigning origins with {ns.seeds} seeds and sweeping per-DC demand...',flush=True)
    dcseed=[]; originrows=[]
    for seed in range(ns.seeds):
        assigned=origins_for_seed(a,configs,seed)
        for k,c in enumerate(configs):
            selected=assigned==k
            originrows.append(dict(seed=seed,dc_id=c['dc_id'],location=c['location'],records=int(selected.sum()),fraction=selected.mean()))
            for mode,ss in starts.items():
                t,v,w=event_sweep(ss[selected],duration[selected],raw[selected],horizon,periodic=True)
                for metric,(col,factor) in mapping.items():
                    for name,scale in scenarios:
                        stat=weighted_summary(t,v[:,RESOURCES.index(col)]*factor*scale,w)
                        capkey={'cpu_cores':'total_cores','gpu_equiv':'total_gpus','memory_gb':'total_mem'}[metric]
                        demand=v[:,RESOURCES.index(col)]*factor*scale
                        stat['current_mean_utilization']=stat['mean']/c[capkey]
                        stat['over_current_capacity_time_fraction']=w[demand>c[capkey]].sum()/w.sum()
                        dcseed.append(dict(seed=seed,dc_id=c['dc_id'],location=c['location'],arrival_mode=mode,
                                           scenario=name,metric=metric,**stat))
        print(f'  origin seed {seed} completed',flush=True)
    ds=pd.DataFrame(dcseed); origins=pd.DataFrame(originrows)
    write_csv(ds,out/'datacenter_seed_statistics.csv'); write_csv(origins,out/'datacenter_origin_counts.csv')
    dcrows=[]
    capkeys={'cpu_cores':'total_cores','gpu_equiv':'total_gpus','memory_gb':'total_mem'}
    for keys,g in ds.groupby(['dc_id','location','arrival_mode','scenario','metric'],sort=False):
        dc,loc,mode,scenario,metric=keys
        conf=next(c for c in configs if c['dc_id']==dc)
        for q in ['p95','p99','maximum']:
            for util in [.7,.8,.9]:
                demand=g[q]; capacity=demand/util
                dcrows.append(dict(dc_id=dc,location=loc,arrival_mode=mode,scenario=scenario,metric=metric,
                                   quantile=q,target_utilization=util,seeds=ns.seeds,
                                   demand_mean=demand.mean(),demand_min=demand.min(),demand_max=demand.max(),demand_std=demand.std(ddof=1),
                                   required_capacity_mean=capacity.mean(),required_capacity_min=capacity.min(),required_capacity_max=capacity.max(),
                                   ceil_capacity_seed_worst=math.ceil(capacity.max()),
                                   current_capacity=conf[capkeys[metric]],
                                   current_minus_seed_worst=conf[capkeys[metric]]-capacity.max(),
                                   reserve_vs_mean_demand=capacity.mean()-demand.mean()))
    dcframe=pd.DataFrame(dcrows); write_csv(dcframe,out/'datacenter_capacity.csv')
    write_csv(configs,out/'datacenter_current_config.csv')
    # Configured 2023 seven-day window uses 2020 month/day lookup, no warm-start backlog.
    configured_window(full,sim,args,mapping,out)
    summary=dict(records_full=sum(map(len,full.tasks_matrix)),records_base=len(a),cycles=len(cycles),
                 interval_rows_full=len(full),interval_rows_base=len(full)//len(cycles),
                 start=str(origin),end_exclusive=str(full.interval_15m.max()+pd.Timedelta(minutes=15)),
                 task_scale=args['task_scale'],scale=args['scale'],group_size=args['group_size'],
                 gpu_zero_fraction=float((a.gpu_wrk_util==0).mean()),
                 gpu_positive_fraction=float((a.gpu_wrk_util>0).mean()),
                 gpu_top_decile_threshold=float(a.gpu_wrk_util.quantile(.9)),
                 gpu_top_decile_fraction=float((a.gpu_wrk_util>=a.gpu_wrk_util.quantile(.9)).mean()),
                 bandwidth_aggregate_ratio=float(a.avg_gpu_wrk_mem.sum()/a.bandwidth_gb.sum()),
                 weekday_mismatch_fraction=float(a.weekday_num_mismatch.mean()),
                 span_missing=int(a.task_span_min.isna().sum()),
                 offset_mean=float(a.offset_seconds.mean()),offset_max=float(a.offset_seconds.max()),
                 input_hashes=before,seeds=list(range(ns.seeds)),official_source=OFFICIAL,
                 versions={'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__})
    if not ns.skip_figures:
        print('Rendering Chinese PNG/PDF/SVG figures...',flush=True)
        make_figures(a,cc,ds,origins,count_cache,starts,sweep_cache,origin,horizon,out,configs)
    report(a,full,cycles,cc,pd.DataFrame(globalrows),dcframe,origins,pd.DataFrame(arrivals),
           pd.DataFrame(intervals),pearson,spearman,summary,sim,composition,runtimeclasses,out)
    after={p:sha256(root/p) for p in protected}
    assert after==before, 'Protected input changed during analysis.'
    summary['protected_inputs_unchanged']=True
    summary['integral_checks']=integrity
    summary['script_sha256']=sha256(Path(__file__))
    (out/'analysis_manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print('COMPLETE: all protected inputs unchanged; all event integral checks passed.',flush=True)


def segment_bins(t,v,w,width,horizon):
    """Exact integral means and peaks for display bins (no point-sampling aliasing)."""
    n=int(math.ceil(horizon/width)); result=np.zeros((n,v.shape[1],2))
    for i,(left,seconds) in enumerate(zip(t,w)):
        right=left+seconds
        first=int(left//width); last=min(n-1,int(np.nextafter(right,-np.inf)//width))
        for b in range(first,last+1):
            overlap=min(right,(b+1)*width)-max(left,b*width)
            if overlap>0:
                result[b,:,0]+=v[i]*overlap
                result[b,:,1]=np.maximum(result[b,:,1],v[i])
    lengths=np.minimum(width,horizon-np.arange(n)*width)
    result[:,:,0]/=lengths[:,None]
    return result


def configured_window(full,sim,args,mapping,out):
    month=sim.get('month',1); day=sim.get('init_day',1); hour=sim.get('init_hour',0)
    start=pd.Timestamp(year=2020,month=month,day=day,hour=hour,tz='UTC')
    duration_days=sim.get('duration_days',7); h=duration_days*DAY
    selected=full[(full.interval_15m>=start)&(full.interval_15m<start+pd.Timedelta(days=duration_days))]
    a=pd.DataFrame(np.concatenate(selected.tasks_matrix.to_list()),columns=FIELDS)
    bins=pd.DatetimeIndex(np.repeat(selected.interval_15m.to_numpy(),selected.tasks_matrix.map(len)))
    inner=pd.to_datetime(a.start_dt,utc=True)
    offset=(inner-inner.dt.floor('15min')).dt.total_seconds().to_numpy()
    raw=a[['cpu_usage','gpu_wrk_util','avg_mem']].to_numpy(dtype=float)
    rows=[]
    for mode,sh in [('sim',0),('exact',offset)]:
        ss=(bins-start).total_seconds().to_numpy()+sh
        t,v,w=event_sweep(ss,a.duration_min.to_numpy(dtype=float)*60,raw,h)
        for j,(metric,(_,factor)) in enumerate(mapping.items()):
            rows.append(dict(arrival_mode=mode,start_lookup_2020=str(start),duration_days=duration_days,
                             selected_records=len(a),boundary='cold_start',scenario='actual_config',metric=metric,
                             **weighted_summary(t,v[:,j]*factor*args['task_scale'],w)))
    write_csv(rows,out/'configured_window_statistics.csv')


def make_figures(a,cc,ds,origins,count_cache,starts,sweeps,origin,horizon,out,configs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib import font_manager
    candidates=['Microsoft YaHei','SimHei','Noto Sans CJK SC','Source Han Sans SC']
    available={f.name for f in font_manager.fontManager.ttflist}
    chosen=next((f for f in candidates if f in available),None)
    if chosen is None:
        raise RuntimeError('A Chinese font is required; install Microsoft YaHei / Noto Sans CJK SC.')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':[chosen],
                         'axes.unicode_minus':False,'font.size':9,
                         'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
                         'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
                         'figure.dpi':120,'savefig.dpi':300})
    palette=['#216B8A','#BB6544','#678C6B','#806A9A','#81785A']
    figdir=out/'figures'; evidence=[]
    def new(title,xlabel,ylabel,wide=False):
        fig,ax=plt.subplots(figsize=(180/25.4,105/25.4 if not wide else 90/25.4))
        fig.subplots_adjust(left=.15,right=.96,bottom=.22,top=.84)
        ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
        ax.grid(axis='y',alpha=.17,lw=.6)
        return fig,ax
    def save(fig,name,claim,data):
        fig.canvas.draw()
        fig.savefig(figdir/f'{name}.png',facecolor='white',dpi=300)
        fig.savefig(figdir/f'{name}.pdf',facecolor='white')
        fig.savefig(figdir/f'{name}.svg',facecolor='white')
        evidence.append(dict(figure=name,claim=claim,source_data=data,font=chosen,
                             alignment='single-panel; not applicable',width_mm=180))
        plt.close(fig)
    for metric in ['duration_min','cpu_usage','gpu_wrk_util','avg_mem','avg_gpu_wrk_mem','bandwidth_gb']:
        x=a[metric].to_numpy(dtype=float)
        fig,ax=new(LABELS[metric]+'分布',LABELS[metric],'任务记录占比')
        ax.hist(x,bins=75,weights=np.ones(len(x))/len(x),color=palette[0],edgecolor='white',linewidth=.25)
        save(fig,metric+'_hist','保留完整范围、包含零值的单任务分布','task_statistics.csv; PKL base cycle')
        fig,ax=new(LABELS[metric]+'累积分布',LABELS[metric],'累积比例')
        ax.plot(np.sort(x),np.arange(1,len(x)+1)/len(x),color=palette[0],lw=1.5)
        ax.set_ylim(0,1.02)
        save(fig,metric+'_cdf','完整样本经验CDF','task_statistics.csv; PKL base cycle')
    time=pd.date_range(origin,periods=int(horizon/900),freq='15min')
    fig,ax=new('基础周期任务到达数量','时间（UTC，2020年）','每15分钟记录数',True)
    ax.plot(time,count_cache['exact',900],color=palette[0],lw=.7)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    save(fig,'arrival_timeseries','49天范围的到达波动','arrival_timeseries_15min.csv')
    fig,ax=new('时间粒度与到达批量峰值','时间粒度','每时隙到达峰值')
    widths=[5,15,60,300,600,900]; xx=np.arange(6)
    for idx,mode in enumerate(['exact','sim']):
        ax.plot(xx,[count_cache[mode,w].max() for w in widths],marker='o',color=palette[idx],
                label=['恢复秒级偏移','仿真批量到达'][idx])
    ax.set_xticks(xx,['5秒','15秒','1分钟','5分钟','10分钟','15分钟'])
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.55,.01),ncol=2,frameon=False)
    save(fig,'arrival_granularity','批量到达改变细粒度峰值','arrival_statistics.csv')
    fig,ax=new('各时间粒度的到达数量分布','每时隙到达数量','累积比例')
    for idx,width in enumerate(widths):
        x=np.sort(count_cache['exact',width]); ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',
                                                    lw=1,label=f'{width}秒')
    ax.set_xscale('symlog',linthresh=1)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.55,.01),ncol=6,frameon=False)
    save(fig,'arrival_count_cdfs','含空时隙的多粒度CDF','arrival_count_distribution.csv')
    hour=a.arrival_exact.dt.hour.value_counts().reindex(range(24),fill_value=0)/49
    fig,ax=new('任务到达的UTC小时分布','UTC小时','平均每小时到达记录数')
    ax.bar(hour.index,hour.values,color=palette[0]);ax.set_xticks(range(0,24,3))
    save(fig,'arrival_hour','每个小时除以49个观测日','arrival_calendar_profiles.csv')
    week=a.arrival_exact.dt.weekday.value_counts().reindex(range(7),fill_value=0)/7
    fig,ax=new('合成2020年日期的星期分布','星期（非原始采集日历）','平均每日到达记录数')
    ax.bar(np.arange(7),week.values,color=palette[0]);ax.set_xticks(range(7),['一','二','三','四','五','六','日'])
    save(fig,'arrival_weekday','星期由重建日期重新计算，不能作原始行为解释','arrival_calendar_profiles.csv')
    daily=a.arrival_exact.dt.floor('D').value_counts().sort_index()
    fig,ax=new('任务每日到达数量','日期（UTC）','每日记录数')
    ax.plot(daily.index,daily.values,color=palette[0]);ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    save(fig,'arrival_daily','49天的逐日异质性','arrival_calendar_profiles.csv')
    for mode in ['exact','sim']:
        gaps=np.diff(np.sort(starts[mode]));x=np.sort(gaps)
        fig,ax=new('相邻到达时间间隔：'+('秒级偏移' if mode=='exact' else '仿真批量'),'到达间隔（秒）','累积比例')
        ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',color=palette[0]);ax.set_xscale('symlog',linthresh=1)
        save(fig,f'interarrival_{mode}_cdf','包含同时到达的零间隔','interarrival_statistics.csv')
    for metric,col,factor in [('cpu','cpu_usage',.05),('gpu','gpu_wrk_util',.05),('memory','avg_mem',5),
                              ('gpu_memory','avg_gpu_wrk_mem',1),('tasks','tasks',1)]:
        fig,ax=new('理论并发需求：'+{'cpu':'CPU','gpu':'GPU','memory':'主存',
                                    'gpu_memory':'显存（原始规模）','tasks':'运行任务'}[metric],
                   '时间（UTC，2020年）',
                   {'cpu':'CPU（核等效量）','gpu':'GPU等效量','memory':'主存（GB）',
                    'gpu_memory':'原始显存（GB，未被仿真约束）','tasks':'并发记录数'}[metric],True)
        for idx,mode in enumerate(['exact','sim']):
            frame=pd.read_csv(out/f'concurrency_timeseries_{mode}_periodic.csv')
            # Each point is the exact maximum over its 15-minute display interval.
            ax.plot(time,frame[col+'_max']*factor,color=palette[idx],lw=.8,alpha=.85,
                    label=['秒级偏移','15分钟批量'][idx])
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.55,.01),ncol=2,frameon=False)
        save(fig,f'concurrency_{metric}','周期稳态；展示每15分钟内精确峰值，统计表按持续秒数加权',
             'concurrency_timeseries_*_periodic.csv')
    fig,ax=new('运行时间与资源的秩相关','变量','变量')
    cols=['duration_min','cpu_usage','gpu_wrk_util','avg_mem','avg_gpu_wrk_mem']
    mat=a[cols].corr(method='spearman').to_numpy()
    image=ax.imshow(mat,cmap='RdBu_r',vmin=-1,vmax=1,aspect='equal')
    short=['运行时间','CPU','GPU','主存','显存']
    ax.set_xticks(range(5),short);ax.set_yticks(range(5),short);ax.grid(False)
    for i in range(5):
        for j in range(5):
            ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=8,
                    color='white' if abs(mat[i,j])>.65 else 'black')
    ax.set_position([.28,.22,.36,.62])
    cax=fig.add_axes([.71,.22,.025,.62])
    fig.colorbar(image,cax=cax,label='Spearman ρ')
    save(fig,'resource_correlation','原始资源与持续时间的单调关系','correlation_spearman.csv')
    fig,ax=new('GPU使用量与显存使用量','GPU使用量（100 = 1 GPU等效量）','显存使用量（GB）')
    h=ax.hexbin(a.gpu_wrk_util,a.avg_gpu_wrk_mem,gridsize=50,bins='log',mincnt=1,cmap='Blues')
    ax.set_position([.15,.22,.60,.62])
    cax=fig.add_axes([.81,.22,.025,.62])
    fig.colorbar(h,cax=cax,label='网格记录数（对数色标）')
    save(fig,'gpu_vs_gpu_memory','使用全量基础周期记录，未删去零GPU记录','PKL base cycle')
    for metric in ['cpu_cores','gpu_equiv','memory_gb']:
        g=ds[(ds.arrival_mode=='sim')&(ds.scenario=='actual_config')]
        g=g[g.metric==metric].groupby('dc_id').p99.agg(['mean','min','max'])
        xx=np.arange(len(g))
        fig,ax=new('来源负载P99：'+LABELS[metric],'数据中心（任务来源）',LABELS[metric])
        ax.bar(xx,g['mean'],color=palette[0],yerr=np.vstack([g['mean']-g['min'],g['max']-g['mean']]),capsize=3)
        ax.set_xticks(xx,[c['location'] for c in configs]);ax.tick_params(axis='x',labelrotation=15)
        save(fig,f'datacenter_{metric}','柱为十种子均值，误差线为种子最小至最大值','datacenter_seed_statistics.csv')
    fig,ax=new('任务来源地分配','数据中心（任务来源）','基础周期记录数')
    g=origins.groupby('dc_id').records.agg(['mean','min','max'])
    ax.bar(np.arange(len(g)),g['mean'],color=palette[0],
           yerr=np.vstack([g['mean']-g['min'],g['max']-g['mean']]),capsize=3)
    ax.set_xticks(range(len(g)),[c['location'] for c in configs]);ax.tick_params(axis='x',labelrotation=15)
    save(fig,'datacenter_origins','十种子均值和完整波动范围；来源不等于执行位置','datacenter_origin_counts.csv')
    write_csv(evidence,out/'figure_manifest.csv')
    (out/'figure_qa_notes.md').write_text('''# 绘图与可追溯性\n\n所有图由 Python/matplotlib 根据完整49天样本生成，包含零值和极端值。每图独立单面板；面板间对齐不适用。图宽180毫米；字体8–10pt，PNG为300dpi，另附PDF和可编辑文字SVG。分布图不截尾；密度图使用全量点的六边形汇总。\n\n显存曲线保持原始规模，因为仿真没有显存资源约束。CPU/GPU/主存曲线采用当前task_scale=5。时间曲线显示每15分钟内精确最大值，绝非对事件稀疏采样；主表的均值与分位数均按持续秒数计算。\n\n中心柱状图柱为各随机种子结果的均值，误差线为全部种子的最小至最大值；不是置信区间，不代表采集不确定性。星期图使用合成2020日历，不能作为真实采集日历证据。全部图的变量、证据目的与源数据对应见figure_manifest.csv。\n''',encoding='utf8')


def report(a,full,cycles,cc,gcap,dcap,origins,arrivals,gaps,pearson,spearman,s,sim,composition,runtimeclasses,out):
    actual=cc[(cc.scenario=='actual_config')&(cc.boundary=='periodic')]
    simrows=actual[actual.arrival_mode=='sim']
    caps=gcap[(gcap.scenario=='actual_config')&(gcap.arrival_mode=='sim')&
              (gcap['quantile']=='p99')&(gcap.target_utilization==.8)]
    dcsel=dcap[(dcap.scenario=='actual_config')&(dcap.arrival_mode=='sim')&
               (dcap['quantile']=='p99')&(dcap.target_utilization==.8)]
    table=dcsel.pivot(index='location',columns='metric',values='ceil_capacity_seed_worst').reset_index()
    stats=pd.read_csv(out/'task_statistics.csv')
    raw=stats[(stats.scenario=='raw')&stats.metric.isin(['duration_min','cpu_usage','gpu_wrk_util','avg_mem','avg_gpu_wrk_mem','bandwidth_gb'])]
    dcbalance=dcsel[['location','metric','demand_mean','current_capacity','current_minus_seed_worst']]
    configwindow=pd.read_csv(out/'configured_window_statistics.csv')
    ranges=gcap[(gcap.scenario=='actual_config')&(gcap.arrival_mode=='sim')].groupby('metric').required_capacity.agg(['min','max']).reset_index()
    share=origins.groupby('location').agg(records_mean=('records','mean'),records_min=('records','min'),records_max=('records','max'),fraction_mean=('fraction','mean')).reset_index()
    maxbatch=int(arrivals[(arrivals.arrival_mode=='sim')&(arrivals.bin_seconds==900)].maximum.iloc[0])
    text=f'''# SustainCluster GPU工作负载系统分析报告

分析日期：2026-11-13。统计对象是当前本地文件及当前源码，不假定与远端仓库版本完全一致。原始PKL、源码、配置均未改写，SHA-256及运行环境见 `analysis_manifest.json`。

## 1. 关键结论与使用建议

完整PKL含 **{s['records_full']:,}** 条记录、**{s['interval_rows_full']:,}** 个非空15分钟片；覆盖 **{s['start']} 至 {s['end_exclusive']}（右端不含）**，不是恰好一个2020年。逐行逐字段比较确认 **{s['cycles']} 个完整49天周期完全相同**；每周期 **{s['records_base']:,}** 条记录、{s['interval_rows_base']:,} 个非空片。独立特征统计只使用第一个周期；不按Job去重，也没有删除异常记录。

当前资源映射规模为5，使用15分钟批量到达、49天周期稳态、P99负载与80%目标利用率时，整个集群的容量估计如下。它们是仿真中的可分资源等效量，GPU不是能直接采购的物理卡数。

{mdtable(caps,['metric','demand','required_capacity','ceil_capacity','current_capacity'])}

该配置可作为“允许约1%时间超过P99负载”的实验起点。若要覆盖样本内所有峰值，请使用 `global_capacity.csv` 的 `maximum` 行；保留目标利用率余量不能替代排队仿真、容灾设计与整卡装箱约束。跨资源P99可能不在同一时刻，不等同于联合99%服务保证。

按本地来源执行、各中心分别使用十个种子中最坏P99/0.8向上取整时，建议起点如下（来源局部执行情景）：

{mdtable(table)}

这些是来源局部执行的参照情景，不是全局调度后的目的地负载。允许跨中心调度时，应先使用全局容量表，再依据链路、成本、碳强度、SLA与可迁移比例确定每个目的地的容量。以CL-SIC为例，P99/80%起点为392 GPU等效容量，超过当前300；其他中心有余量不能自动保证它的本地任务可执行。

## 2. 数据来源、预处理和单位核验

官方数据说明：[Alibaba Cluster Trace GPU 2020]({OFFICIAL})。官方区分 `plan_cpu/plan_gpu/plan_mem` 申请量与 `cpu_usage/gpu_wrk_util/avg_mem` 使用量；传感器数据记录实例生命周期均值。官方描述原始记录有Job→Task→Instance层级，当前PKL已丢失task_name、worker_name、inst_num及计划资源，无法还原每条记录的唯一Task身份和真实申请量。

本地 `alibaba_analysis.ipynb` 将传感器资源按 `(job_name, task_name)` 求和，再与Task信息合并；因此PKL资源是任务聚合后的实例平均使用量之和，不是瞬时测量或计划申请。求和还可能受连接重复影响；缺少原始表时无法从PKL证明没有重复连接。当前报告将每行称作“任务记录”。

| 索引 | 字段 | 含义及单位 |
| --- | --- | --- |
| 0 | job_name | Job标识，不能唯一标识Task |
| 1 | start_time | 脱敏Task启动时间（秒）；不是已验证的用户提交时间 |
| 2 | end_time | 脱敏Task结束时间（秒）；部分缺失 |
| 3 | start_dt | 当前预处理产生的UTC时间，保留1970脱敏日期 |
| 4 | duration_min | 同一Task下实例有效运行时间均值/60（分钟） |
| 5 | cpu_usage | 已用CPU百分数，100代表1核等效量 |
| 6 | gpu_wrk_util | 已用GPU百分数，100代表1 GPU等效量，可超过100 |
| 7 | avg_mem | 按实例聚合的平均主存用量，官方单位GB |
| 8 | avg_gpu_wrk_mem | 按实例聚合的平均显存用量，官方单位GB |
| 9 | bandwidth_gb | 本地派生的网络量估计，除以1024³，数值单位实际是GiB |
| 10 | weekday_name | 从1970内部时间派生的英文星期 |
| 11 | weekday_num | 从1970内部时间派生的星期，周一=0 |

官方对GB字段的十进制/二进制细节没有充分说明，本报告保持官方“GB”标签，不额外假定其换算。`Task`将主存标作GB，网络成本函数也按GB计费；GiB派生量与GB价目可能另有约7.37%的单位基准差异（相同字节量按十进制GB计应乘1.073741824）。CPU/GPU等效量不能替代CPU型号、GPU型号、整卡数量或显存规格。

### 2.1 duration_min为何与时间戳跨度不同

`alibaba_utils.get_dfia()` 对实例正开始/结束时间计算runtime，并按 `(job_name, task_name)` 求平均；`get_dfa()` 使用 `duration_min = runtime_i / 60`。PKL的 `start_time/end_time` 则来自Task表，因此 `(end_time-start_time)/60` 是Task起止跨度；实例启动延迟、多实例异步运行及失败/重试会使两者不同。`task_span_min` 缺失 {s['span_missing']:,} 条；不据此删除资源与duration有效的记录。

{mdtable(stats[(stats.scenario=='raw')&stats.metric.isin(['duration_min','task_span_min','duration_minus_span_min'])],['metric','count','mean','p50','p95','p99','minimum','maximum'])}

报告的区间使用用户指定的duration_min；若改为Task跨度会改变并发量，且无法为缺失结束时间的行确定跨度，不做无依据补值。

### 2.2 网络量派生过程还有身份连接风险

notebook先按 `(job_name, task_name)` 汇总 `(read + write)/1024³`，但随后只按 `job_name` 与已经丢失task_name的CSV合并。这可能使同一Job中的多任务发生多对多连接。PKL不足以定量确认究竟新增了多少行；完整相同行为零不表示没有此类连接问题。官方传感器read/write也经过生命周期汇总，当前派生量不应解释为真实输入数据集大小或链路速率；精确传输量语义待原始采样与网络字段进一步核实。

## 3. 实际调用链与实现审计

`ClusterManager.get_tasks_for_timestep()` 将当前年份替换为2020，以 `interval_15m == adjusted_time` 精确取行，调用 `extract_tasks_from_row(scale=1, task_scale=5, group_size=1, current_time_utc=current_time)`。`arrival_time=current_time_utc`，没有使用内部start_dt。同一片所有记录被视为同刻到达。若改变仿真步长却保持精确查找、不补发经过的片，可能漏任务；当前15分钟默认值与文件分箱一致。

`TaskSchedulingEnv` 还将 `self.time_step` 硬编码为15分钟，因此仅修改YAML的timestep_minutes并不会自动改变该环境的调度粒度。全局任务时间来自传入的start_time，中心reset还可能随机化自己的初始日期和小时；本次不把配置文字值当成一次已运行的完整仿真轨迹。

实际转换：CPU=`task_scale*cpu_usage/100`，GPU=`task_scale*gpu_wrk_util/100`，主存=`task_scale*avg_mem`。`scale`控制复制与随机扰动，`group_size`控制元任务合并；当前两者均为1，所以分析没有复制/合并。显存字段没有成为仿真独立资源约束，不能根据本报告宣称显存可行性已经验证。

**确认索引错误：** `bandwidth_gb=float(task_data[8])` 读取的是显存；按实际12列应读索引9。当前映射不随task_scale缩放。两种映射不改变本报告CPU/GPU/主存需求，但会改变网络成本、传输能耗，以及任何使用该字段的迁移/延迟机制。

{mdtable(pd.read_csv(out/'bandwidth_mapping_comparison.csv'),['mapping','field','mean','p50','p99','maximum','zero_fraction','total_network_value','transmission_energy_kwh_if_all_remote'])}

全周期总量比（索引8/索引9）为 **{s['bandwidth_aggregate_ratio']:.4f}**。它是总体比，不能当作每条记录的统一误差倍数，逐记录比值分布见 `bandwidth_mapping_ratios.csv`。表中的能耗仅为所有任务都迁移一次、使用仓库0.06 kWh/GB系数的假想对照，不能解释为实际运行能耗。

当前配置年为{sim['year']}，默认{sim['month']}月{sim['init_day']}日{sim['init_hour']}时、{sim['duration_days']}天、{sim['timestep_minutes']}分钟步长。训练/评估入口可能另传日期或随机重置；本报告容量主结论针对49天基础模式，不冒充某次训练轨迹。默认日期对应的七天冷启动窗口另见：

{mdtable(configwindow,['arrival_mode','metric','selected_records','mean','p95','p99','maximum'])}

## 4. 数据完整性和重复扩展

`data_quality.csv` 包含全部12字段缺失值、零值、负值、非有限值和1.5×IQR异常标记（标记不删除）。资源与duration的可计算性在运行时强制检查；非法值会中止分析，不静默插补。`end_time`缺失允许保留，只影响Task跨度比较。

{mdtable(cycles,['cycle','interval_rows','records','identical_matrix_rows','exactly_equal_to_base'])}

每个49天窗口理论4704片，实际{len(full)//len(cycles)}片，缺片{4704-len(full)//len(cycles)}片；当作零到达片补入计数分母。所有周期的相对时隙位置、矩阵形状、记录顺序和每个字段都比较一致（相同位置NaN视为相同）。该比较保留矩阵内记录，属于识别重复采样窗口，不是对任务去重。完整样本的缺失/异常数量由已逐矩阵验证的周期倍数准确外推。

基础周期有 {a.job_name.nunique():,} 个不同Job、{a.job_name.duplicated().sum():,} 条超出每个Job首条的记录；全部12字段完全重复 {a[FIELDS].duplicated().sum():,} 条。仅六个数值特征相同的重复超出首条 {a[['duration_min','cpu_usage','gpu_wrk_util','avg_mem','avg_gpu_wrk_mem','bandwidth_gb']].duplicated().sum():,} 条，不代表可以删除。全年绝不能按1,300,144个独立观测计算置信区间或随机划分训练测试集；建议按基础周期连续时间段划分并检查同Job/重复特征泄漏。

### 4.1 内外时间轴与星期检查

恢复公式严格为 `offset=start_dt-start_dt.floor('15min')`；`arrival_exact=interval_15m+offset`。这只是保留预处理秒级偏移的“启动时间代理”，不是恢复被脱敏的2020真实日期或提交时间。`arrival_sim=interval_15m`。

所有内部时间比将start_time直接作为Unix秒解析的UTC时间少28800秒。预处理先解析为无时区时间、再localize到上海、最后转UTC，造成该偏移；官方示例是将秒解析成带时区的Unix时间，这与当前实现不同。8小时是15分钟整数倍，所以上述秒级偏移仍可用，但日内UTC相位与原始采集对应关系存在问题。

内部1970年1月26日周一被平移到2020年1月1日周三；旧星期与重建日期不一致比例 **{s['weekday_mismatch_fraction']:.2%}**。所有星期统计由重建2020时间重新计算；仅表示合成日历，对真实周末/工作日行为不能下结论。完整文件超出年末来自整块复制后未裁剪，不能把2021段当成独立新增数据。

## 5. 单任务运行时间、资源及异质性

{mdtable(raw,['metric','minimum','mean','p50','p75','p90','p95','p99','maximum','std','zero_fraction'])}

运行时与CPU/GPU/主存的Pearson和Spearman相关都保存在独立CSV。Spearman表如下（值表示秩相关，不能推出因果）：

{mdtable(spearman.rename_axis('metric').reset_index(),['metric','duration_min','cpu_usage','gpu_wrk_util','avg_mem','avg_gpu_wrk_mem'])}

短/中/长任务仅按基础样本Q1、Q3划分，便于描述，不声称是领域固定阈值：

{mdtable(runtimeclasses)}

GPU使用量为零的记录占 **{s['gpu_zero_fraction']:.2%}**。较高GPU使用量定义为本样本经验前10%，阈值为 **{s['gpu_top_decile_threshold']:.4g}** 原始单位（除100后为GPU等效量），占比 **{s['gpu_top_decile_fraction']:.2%}**；这不是“GPU利用率超过某硬件阈值”的定义，多实例求和允许远大于100。

CPU/GPU/主存异质性采用各字段在全样本内的经验百分位名次（平均名次、含零值并列），比较每条记录哪种资源相对本资源分布更靠前：

{mdtable(composition)}

以上是“相对资源偏重”的描述，不是物理CPU/GPU瓶颈分类。未引入未经依据的密集型阈值，也未拟合聚类。大量GPU零值与显存非零记录可能来自生命周期统计、角色或聚合口径，不能仅凭GPU利用率把它们当作无显卡要求任务。GPU-显存关系另有全量六边形密度图，实际物理可行性仍需GPU申请与型号字段。

数据预处理先去除小于15分钟的任务，再对duration、CPU、主存、网络量顺序执行IQR过滤；当前数据持续时间上限和资源尾部因此受到筛选。短任务缺失、尾部被裁剪意味着容量数字只适用于此加工轨迹，不能推到Alibaba原始生产全量集群。

## 6. 到达特征、突发性和调度粒度

统计完整49天窗口，按UTC午夜锚定半开时隙，包含无任务时隙。均值、中位数和分位数是“每片记录数”，不是每秒速率。5/15秒、1/5/10/15分钟两个到达轴共12组结果：

{mdtable(arrivals,['arrival_mode','bin_seconds','mean','p50','p90','p95','p99','maximum','zero_fraction','fano','peak_to_mean'])}

相邻到达间隔包含不同记录同时到达的零间隔，按排序后的每条记录计算；不是仅对不同时间戳求差：

{mdtable(gaps,['arrival_mode','mean','p50','p90','p95','p99','below_5s','below_15s','below_60s','simultaneous_fraction','burstiness'])}

Fano=计数样本方差/均值，CV=样本标准差/均值，间隔突发指标B=(CV−1)/(CV+1)。Fano显著高于1表示比均匀泊松计数更过度离散；B和Fano只是描述，非平稳日内趋势、任务批处理也会造成高值，不能据此证明长程依赖或泊松假设检验结论。

15分钟视角每片最多 **{maxbatch}** 条记录。批量到达将时隙内真实偏移压到同一点，制造大量零间隔；同一片里的排队顺序、链路竞争和短时资源峰值被改变。平均提前量 **{s['offset_mean']:.2f}秒**，最大 **{s['offset_max']:.0f}秒**。任务最短15分钟是预处理选择，因此当前数据适合15分钟能耗/碳信号粗粒度实验；若研究秒级排队、SLA及突发调度，15分钟不足，建议以保留偏移的1分钟事件驱动模型作主要对照，并做5秒、15秒、5分钟等敏感性实验。仅改配置步长不能自动恢复秒级任务。

小时图按49个观测日求每小时均值；星期图按每星期7个日期求日均；每日序列不额外平滑。文件提供全部计数分布，图中的CDF包括空片。

## 7. 理论并发资源需求（本分析核心）

每条记录假定自到达开始、资源保持PKL平均值、运行duration_min分钟，区间为 `[arrival, arrival+duration)`。同一时刻结束和开始统一合并增量后观察，避免同刻假峰。CPU/GPU/主存/显存与并发数都使用事件扫描；统计分位数为逆持续时间CDF，均值按各状态持续秒数加权，绝不对事件行等权平均。

主情景是已确认49天模式持续重复的周期稳态，补入上一周期仍在运行的任务，以消除左边界空系统偏差。49天只计一个周期的时间，不增加独立样本数。另报从空系统开始的cold_start情景，并将超出右边界的运行截断。区间积分与逐记录“资源×有效运行秒数”交叉核验通过；周期稳态下资源时间积分等于所有任务完整资源时间积分。

当前task_scale=5的周期稳态结果：

{mdtable(actual,['arrival_mode','metric','mean','p50','p90','p95','p99','maximum','peak_time_utc'])}

与当前全局容量相比的理论平均利用率和超载时长比例：

{mdtable(simrows,['metric','current_capacity','current_mean_utilization','over_current_capacity_time_fraction','over_80pct_current_capacity_time_fraction'])}

这说明CPU与主存在此理论模型中配置较宽裕，GPU相对更紧张；局部中心可能与全局结论不同。低平均值不能证明应按均值缩容，必须同时检查尾部需求和目的地调度约束。

原始单位与scale=1/2/5/current_config全表在 `concurrency_statistics.csv`；并发数和原始显存另列于raw场景。task_scale=5与actual_config数值应完全相同，是显式配置校验。规模缩放1→2→5下，CPU/GPU/主存线性变化；没有任务数量或持续时间缩放。

曲线使用每15分钟显示窗口内的精确最大值；CSV也提供精确时间积分平均值。它们用于显示，不参与上述连续时间分位数计算。该需求是“无限服务台、无等待、常资源”的理论需求，既非原生产集群瞬时使用量，也非SustainCluster调度和传输后实际占用。任务只在离散步更新、资源分配与归还、等待和迁移等差异需要运行完整调度器另验。

## 8. 容量建议、余量与资源失衡

`global_capacity.csv` 对两个到达轴、四个scale标签、P95/P99/maximum和70%/80%/90%目标利用率完整交叉计算：容量=Dq/u；余量=Dq×(1/u−1)。90/80/70%分别对应相对负载11.11%/25%/42.86%余量。表中保留浮点等效量和向上取整值，主存取整仅为配置便利。

当前规模、仿真批量、全部分位数和利用率组合得到的全局容量包络如下（低端=P95/0.9，高端=最大负载/0.7，不是置信区间）：

{mdtable(ranges)}

P95适用于允许更多排队、关注平均能耗的场景；P99适用于通常负载可用性实验；最大值适用于此有限样本内无容量超载的比较基线。即使使用样本最大值也不保证未来生产负载无超载，因为当前轨迹经过截尾过滤且无独立全年季节性。

不同task_scale的建议见完整表。若从当前5降为1，在同到达/边界/分位数/利用率下三类资源容量全部除5；升为2则为当前的2/5。GPU等效量可分是当前仿真假设；物理部署还须加上按卡型装箱、显存与每任务申请量限制。不能仅对小数GPU容量向上取整就声称真实可调度。

## 9. 地理分布式任务来源与配置平衡

`assign_task_origins()` 使用 `population_weight × activity`，当地小时8≤hour<20时activity=1，否则0.3，然后归一化概率独立抽样；时区使用配置的固定timezone_shift，不自动处理夏令时。恢复偏移不会跨越原15分钟时隙，因此本分析两个轴使用同一组来源抽样便于比较。向量抽样已与原始函数在24个UTC小时、每小时256次抽样逐项核验一致。

十个隔离的RandomState种子0–9，按原记录顺序抽样。全仿真还会穿插其他随机调用，所以这些种子用于来源分配敏感性，并不声称重放某次完整RL仿真。周期稳态的入边界任务沿用同一来源抽样，这反映同一模式周期性来源的建模假设。种子波动是Monte Carlo波动范围，非总体置信区间。

{mdtable(share)}

按中心局部执行的P99与当前配置比较如下（余量列为当前容量−十种子最坏P99/0.8）：

{mdtable(dcbalance)}

中心容量无需简单按人口比例分配；时区活动与CPU/GPU/内存组合使各中心的资源峰值不同。建议根据 `datacenter_capacity.csv` 按资源独立配置，优先查看负余量资源。对全局可迁移工作负载，应把以上来源容量看作“禁止迁移时的参考”，不要把所有中心P99或最大值求和当成同一时刻的全球峰值；全球表来自全体记录统一事件扫描，更适合资源池化基线。

## 10. 对研究设计的具体含义与局限

1. **任务有什么特点？** 加工后的任务至少15分钟，平均{a.duration_min.mean():.2f}分钟，中位数{a.duration_min.median():.2f}分钟；资源分布偏斜且GPU零值占{s['gpu_zero_fraction']:.2%}。CPU/GPU是多实例平均使用量之和，不能当成每卡利用率或申请量。保留相同Job的所有记录。
2. **CPU/GPU/内存需多少？** 首选本报告第1节的P99/80%全局与来源局部执行表，再用最大值/70%作保守实验边界。数字只属于当前task_scale=5的理论等效资源；1/2规模按线性比例换算。实际配置失衡及余量逐中心逐资源给出，不直接改datacenters.yaml。
3. **有无突发？** 到达统计与间隔表显示时隙计数过度离散与批量现象；仿真轴会额外制造集中到达，因此实验必须并列两个轴，不能把仿真零间隔全解释成真实提交突发。
4. **15分钟合理吗？** 对本轨迹的长任务与能源信号可作粗基线；对排队、迁移、细粒度SLA不充分，且小于15分钟的任务已经被筛除。建议增加保留秒偏移的1分钟/事件驱动对照。
5. **中心如何分配？** 按本地执行假设使用多种子每中心资源分位数；若允许迁移，以全局需求和链路约束共同优化目的地，不用来源概率直接替代执行负载。
6. **有哪些实现问题？** 网络字段错位；网络估计存在只按Job合并风险及GiB/GB差异；预处理UTC有8小时偏移；旧星期与合成日历不符；全年实际392天且8周期重复；资源均值和实例平均duration组合不保真于瞬时并发；缺少task_name与原始申请/显卡型号/实例数；IQR过滤削弱尾部；仿真无独立显存约束。分组功能当前关闭，若未来开启，还需复核“注释称最早截止、实现取max”等语义，不属于本次已运行场景。

## 11. 复现与交付文件

在仓库根目录运行：

```powershell
python analysis/workload_profile/workload_analysis.py --seeds 10
python -m unittest discover -s analysis/workload_profile -p test_analysis.py
```

依赖：numpy、pandas、scipy、matplotlib、pyyaml及中文字体（Microsoft YaHei等）。脚本从自身位置定位仓库；输入结构、原始矩阵列宽、周期一致性、资源有限性、调用参数、来源规则和事件积分都设有检查。`--skip-figures`可重算表格而保留已存在的图片；默认实际生成全部图片。`analysis_manifest.json`记录输入哈希、脚本哈希、版本、种子和验证结果。

必需CSV：`data_quality.csv`、`task_statistics.csv`、`arrival_statistics.csv`、`concurrency_statistics.csv`、`global_capacity.csv`、`datacenter_capacity.csv`。额外结果包括周期比较、时间间隔、计数分布、每日/小时/星期、两种相关矩阵、网络映射对比、各中心种子明细和完整15分钟显示时间序列。

所有CSV为UTF-8 BOM，可用Excel打开。统计比例用0–1表示；std为样本标准差（ddof=1）；单任务和到达计数分位数使用numpy线性插值；连续并发分位数使用按持续秒数加权的经验逆CDF。峰值时间取最早达到最大值的事件时刻，表中有明确到达轴与边界情景。所有图在 `figures/`，包含PNG、PDF及SVG；证据与源数据见 `figure_manifest.csv`。
'''
    text+='\n## 12. 主要图表\n\n所有图均可在figures目录取得PDF/SVG矢量版本；下图为PNG预览。完整31张图及源数据见figure_manifest.csv。\n\n'
    plots=[('duration_min_hist','任务运行时间分布'),('duration_min_cdf','任务运行时间CDF'),
           ('cpu_usage_hist','CPU使用量分布'),('gpu_wrk_util_hist','GPU使用量分布'),
           ('avg_mem_hist','主存使用量分布'),('avg_gpu_wrk_mem_hist','显存使用量分布'),
           ('bandwidth_gb_hist','正确网络字段分布'),('arrival_timeseries','任务到达时间序列'),
           ('arrival_granularity','不同时间粒度的到达峰值比较'),('arrival_hour','UTC小时分布'),
           ('arrival_weekday','合成2020年星期分布'),('concurrency_cpu','CPU理论并发需求'),
           ('concurrency_gpu','GPU理论并发需求'),('concurrency_memory','主存理论并发需求'),
           ('resource_correlation','资源及持续时间相关性'),('datacenter_gpu_equiv','各中心GPU来源需求比较')]
    for filename,title in plots:
        text+=f'### {title}\n\n![{title}](figures/{filename}.png)\n\n'
    (out/'workload_analysis_report.md').write_text(text,encoding='utf8')


if __name__=='__main__':
    main()
