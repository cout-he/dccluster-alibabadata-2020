# Alibaba PAI GPU v2020 workload analysis

This repository contains the original Alibaba PAI GPU v2020 trace and a
reproducible analysis of task arrivals, burstiness, workload heterogeneity,
and resource concurrency.

- [Analysis report (Chinese)](alibaba-gpu-v2020-analysis/阿里GPU任务到达与调度粒度报告.md)
- [Figures](alibaba-gpu-v2020-analysis/figures)
- [Summary metrics](alibaba-gpu-v2020-analysis/metrics.json)

The original CSV files, archives, and headers are credited to Alibaba and the
trace authors. They are redistributed unchanged under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
[Official release and checksums](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2020) ·
[Upstream license](https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2020/LICENSE) ·
[Weng et al., NSDI 2022](https://www.usenix.org/conference/nsdi22/presentation/weng)

The raw CSV files and archives use Git LFS. After cloning, run
`git lfs pull` to fetch both forms of the dataset. With NumPy, pandas, SciPy,
and Matplotlib installed, run the following from the repository root:

```sh
python alibaba-gpu-v2020-analysis/analyze.py
python alibaba-gpu-v2020-analysis/plot.py
```
