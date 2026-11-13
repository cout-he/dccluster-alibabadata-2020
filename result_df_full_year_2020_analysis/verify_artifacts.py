"""Cross-check exported results against independent identities and protected inputs."""
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from workload_analysis import sha256


def main():
    out=Path(__file__).resolve().parent
    root=out.parent.parent
    manifest=json.loads((out/'analysis_manifest.json').read_text(encoding='utf8'))
    checks=[]
    def check(name,condition):
        assert condition,name
        checks.append(name)
    for name,digest in manifest['input_hashes'].items():
        check('protected SHA256: '+name,sha256(root/name)==digest)
    check('script SHA256',sha256(out/'workload_analysis.py')==manifest['script_sha256'])
    cycles=pd.read_csv(out/'cycle_comparison.csv')
    check('all 49-day cycles equal',cycles.exactly_equal_to_base.all())
    check('record multiplicity',cycles.records.sum()==manifest['records_full']==len(cycles)*manifest['records_base'])
    ar=pd.read_csv(out/'arrival_statistics.csv')
    check('12 arrival modes and widths',len(ar)==12)
    check('empty arrival bins included',(ar['count']==ar.bins).all())
    for row in ar.itertuples():
        check(f'arrival conservation: {row.arrival_mode}/{row.bin_seconds}',
              np.isclose(row.mean*row.bins,manifest['records_base'],rtol=1e-8))
    left=ar[ar.arrival_mode=='exact'].set_index('bin_seconds').loc[900]
    right=ar[ar.arrival_mode=='sim'].set_index('bin_seconds').loc[900]
    check('15-minute arrivals identical',all(left[k]==right[k] for k in ['mean','p99','maximum','zero_fraction']))
    c=pd.read_csv(out/'concurrency_statistics.csv')
    for mode in ['exact','sim']:
        for boundary in ['cold_start','periodic']:
            g=c[(c.arrival_mode==mode)&(c.boundary==boundary)]
            for metric in ['cpu_cores','gpu_equiv','memory_gb']:
                v=g[g.metric==metric].set_index('scenario')
                for q in ['mean','p50','p90','p95','p99','maximum']:
                    check(f'scale linearity: {mode}/{boundary}/{metric}/{q}',
                          np.isclose(v.loc['task_scale_2',q],2*v.loc['task_scale_1',q],rtol=1e-8)
                          and np.isclose(v.loc['task_scale_5',q],5*v.loc['task_scale_1',q],rtol=1e-8)
                          and v.loc['task_scale_5',q]==v.loc['actual_config',q])
    p=c[(c.scenario=='actual_config')&(c.boundary=='periodic')].set_index(['arrival_mode','metric'])
    for metric in ['cpu_cores','gpu_equiv','memory_gb']:
        check('periodic mean invariant to time offsets: '+metric,
              np.isclose(p.loc['exact',metric]['mean'],p.loc['sim',metric]['mean'],rtol=1e-9))
    globalcap=pd.read_csv(out/'global_capacity.csv')
    check('global capacity complete cross product',len(globalcap)==2*4*3*3*3)
    check('global capacity formula',np.allclose(globalcap.required_capacity,
                                               globalcap.demand/globalcap.target_utilization,rtol=1e-8))
    check('global capacity ceiling',np.array_equal(globalcap.ceil_capacity,np.ceil(globalcap.required_capacity)))
    dc=pd.read_csv(out/'datacenter_capacity.csv')
    check('per-DC capacity complete cross product',len(dc)==5*2*4*3*3*3)
    check('per-DC capacity formula',np.allclose(dc.required_capacity_max,
                                              dc.demand_max/dc.target_utilization,rtol=1e-8))
    os=pd.read_csv(out/'datacenter_origin_counts.csv')
    check('origin record conservation',os.groupby('seed').records.sum().eq(manifest['records_base']).all())
    ds=pd.read_csv(out/'datacenter_seed_statistics.csv')
    # Different per-DC distributions need not share P99; means MUST add exactly.
    sums=ds.groupby(['seed','arrival_mode','scenario','metric'])['mean'].sum()
    for (seed,mode,scenario,metric),value in sums.items():
        expected=c[(c.arrival_mode==mode)&(c.boundary=='periodic')&
                   (c.scenario==scenario)&(c.metric==metric)]['mean'].iloc[0]
        check(f'origin resource mean conservation: {seed}/{mode}/{scenario}/{metric}',
              np.isclose(value,expected,rtol=1e-8))
    check('event-integral errors below 1e-9',all(x['max_relative_error']<1e-9 for x in manifest['integral_checks']))
    figs=pd.read_csv(out/'figure_manifest.csv')
    check('31 requested and supplementary figures',len(figs)==31)
    for name in figs.figure:
        for extension in ['png','pdf','svg']:
            check('figure exists: '+name+'.'+extension,(out/'figures'/f'{name}.{extension}').stat().st_size>1000)
    report=(out/'workload_analysis_report.md').read_text(encoding='utf8')
    for value in ['50984','1127','45013','392','1,300,144','162,518']:
        check('report result exists: '+value,value in report)
    check('Chinese report length',len(report)>10000)
    for name in ['data_quality.csv','task_statistics.csv','arrival_statistics.csv',
                 'concurrency_statistics.csv','global_capacity.csv','datacenter_capacity.csv']:
        check('required table nonempty: '+name,len(pd.read_csv(out/name))>0)
    result={'checks_passed':len(checks),'all_passed':True,'checks':checks}
    (out/'verification_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(f'PASS: {len(checks)} artifact, conservation, scaling, capacity and SHA256 checks.')


if __name__=='__main__':
    main()
