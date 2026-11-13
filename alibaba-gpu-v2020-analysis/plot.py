"""Render the analysis figures from the aggregated source data."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
BLUE = '#185b89'
ORANGE = '#aa522f'
GRAY = '#7b858e'
INK = '#192733'
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Microsoft YaHei', 'Arial', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 10, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'axes.spines.right': False, 'axes.spines.top': False,
    'axes.linewidth': .7, 'axes.edgecolor': GRAY,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK, 'ytick.color': INK,
    'legend.frameon': False, 'legend.fontsize': 9,
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
})

FILES = ['01_到达间隔与窗口','02_日内周期与突发','03_任务异质性','04_全程并发与资源','05_十五分钟内的资源峰值']


def axes(rows, cols, size):
    fig, ax = plt.subplots(rows,cols,figsize=size,layout='constrained',squeeze=False)
    fig.get_layout_engine().set(w_pad=.15,h_pad=.12,wspace=.04,hspace=.05)
    return fig, ax.ravel()


def label(ax, title, x, y):
    ax.set(title=title, xlabel=x, ylabel=y)
    ax.title.set_position((0,1.0))
    ax.title.set_ha('left')
    ax.set_axisbelow(True)
    ax.grid(axis='y',color='#dce3e9',linewidth=.65)


def arrival(m, d):
    fig, (ax, bx) = axes(1,2,(11,3.65))
    ax.step(d['q'],d['gap_cdf'],where='post',color=BLUE,label='所有任务')
    ax.step(d['q'],d['distinct_cdf'],where='post',color=ORANGE,label='每秒只保留一次到达')
    ax.axvline(5,color=GRAY,lw=.8,ls=':')
    ax.set(xlim=(0,60),ylim=(0,1.04))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc='lower right')
    label(ax,'a  相邻到达间隔的累积分布','到达间隔（秒）','累积比例')
    window = m['window']
    positions = np.arange(5)
    bx.bar(positions,[x['zero'] for x in window.values()],color=BLUE,width=.5,label='实际空窗口')
    bx.plot(positions,[x['poisson_empty'] for x in window.values()],marker='o',color=ORANGE,label='同均值泊松基线')
    bx.plot(positions[:4],[x['within_block_empty'] for x in list(window.values())[:4]],marker='x',ls='--',color=GRAY,label='固定每 15 分钟总量的基线')
    bx.set_xticks(positions,['5 秒','15 秒','1 分钟','5 分钟','15 分钟'])
    bx.set_ylim(0,.82)
    bx.yaxis.set_major_formatter(PercentFormatter(1))
    fig.legend(*bx.get_legend_handles_labels(),loc='outside upper center',ncol=3)
    label(bx,'b  空窗口比例随粒度变化','固定窗口宽度','空窗口比例')
    return fig


def burst(m,d):
    fig,(ax,bx,cx,dx)=axes(2,2,(11,6.7))
    daily=d['daily']
    ax.plot(np.arange(1,len(daily)+1),daily,color=BLUE)
    label(ax,'a  日到达量随追踪日期变化','完整自然日序号','任务数 / 日')
    hourly=d['hour_by_weekday']
    left=m['sample']['left']
    day_of_week = (np.arange(len(daily))+(left+28800)//86400+3)%7
    weekday_sizes = np.bincount(day_of_week,minlength=7)
    bx.plot(np.arange(24),np.average(hourly[:5],axis=0,weights=weekday_sizes[:5]),color=BLUE,label='工作日')
    bx.plot(np.arange(24),np.average(hourly[5:],axis=0,weights=weekday_sizes[5:]),color=ORANGE,label='周末')
    bx.set_xticks(range(0,24,3))
    bx.legend(loc='upper right')
    label(bx,'b  北京时间的日内轮廓','北京时间（小时）','平均任务数 / 小时')
    cx.bar(np.arange(1,16),d['example_minutes'],color=BLUE,width=.78)
    cx.set_xticks([1,3,6,9,12,15])
    label(cx,'c  集中到达的 15 分钟示例','窗口内第几分钟','任务数 / 分钟')
    y=d['peak_five']
    dx.bar(np.arange(len(y))*5,y,width=5,align='edge',color=BLUE,linewidth=0)
    label(dx,'d  含全期最高 5 秒到达量的窗口','窗口内时间（秒）','任务数 / 5 秒')
    return fig


def heterogeneity(m,d):
    fig, grid_axes=plt.subplots(2,3,figsize=(12,6.5))
    fig.subplots_adjust(left=.073,right=.98,bottom=.095,top=.93,wspace=.42,hspace=.45)
    ax=grid_axes.ravel()
    specs=[
        ('duration','a  执行时长','分钟'),
        ('cpu_usage','b  CPU 使用','vCPU 核'),
        ('gpu_wrk_util','c  GPU 使用','GPU 等效算力'),
        ('avg_mem','d  内存使用','GB'),
        ('avg_gpu_wrk_mem','e  显存使用','GB'),
        ('network_read_gib','f  网络输入','GiB'),
    ]
    for plot,(name,title,unit) in zip(ax,specs):
        p = d['quantiles']
        value = d[name]
        stats = m['heterogeneity']['positive_duration'] if name == 'duration' else m['heterogeneity']['observed'][name]
        shown = p < .99
        plot.plot(np.r_[p[shown],.99],np.r_[value[shown],stats['p99']],color=BLUE)
        plot.set_xlim(0,1)
        plot.set_ylim(bottom=0)
        plot.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        plot.set_xticks([0,.25,.5,.75,.99])
        label(plot,title,'样本分位','数值（'+unit+'）')
    return fig


def concurrency(m,d):
    fig, plots=axes(2,2,(11,6.8))
    days=np.arange(len(d['task_mean']))/96
    panels=[
        ('task','a  并发任务','任务数'),
        ('instance','b  并发运行实例','实例数'),
        ('cpu_request','c  在跑实例的 CPU 申请','vCPU 核'),
        ('gpu_request','d  在跑实例的 GPU 申请','GPU 张数'),
    ]
    for ax,(key,title,unit) in zip(plots,panels):
        mean,peak=d[key+'_mean'],d[key+'_max']
        ax.fill_between(days,mean,peak,color='#9fc1d7',alpha=.6,label='区间均值至区间峰值')
        ax.plot(days,mean,lw=.7,color=BLUE,label='15 分钟平均')
        ax.xaxis.set_major_locator(MaxNLocator(7,integer=True))
        label(ax,title,'距首个完整日开始（天）',unit)
    fig.legend(*plots[0].get_legend_handles_labels(),loc='outside upper center',ncol=2)
    return fig


def detail(m,d):
    fig,(ax,bx,cx,dx)=axes(2,2,(11,6.7))
    time=np.arange(900)/60
    for a,key,title,unit in [
        (ax,'task','a  同时运行的任务','任务数'),
        (bx,'instances','b  同时运行的实例','实例数'),
        (cx,'cpu','c  同时运行实例的 CPU 申请','vCPU 核'),
        (dx,'gpu','d  同时运行实例的 GPU 申请','GPU 张数'),
    ]:
        vals=d['detail_'+key]
        a.plot(time,vals,color=BLUE,label='逐秒重建')
        a.axhline(vals.mean(),color=ORANGE,ls='--',lw=1.1,label='15 分钟平均')
        a.set_xlim(0,15)
        label(a,title,'窗口内时间（分钟）',unit)
    fig.legend(*dx.get_legend_handles_labels(),loc='outside upper center',ncol=2)
    return fig


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-scripts',type=Path)
    args=parser.parse_args()
    if args.audit_scripts:
        sys.path.insert(0,str(args.audit_scripts))
        from audit_panel_alignment import require_matplotlib_panel_alignment
    metrics=json.loads((HERE/'metrics.json').read_text(encoding='utf-8'))
    with np.load(HERE/'figure_data.npz') as data:
        for file,create in zip(FILES,(arrival,burst,heterogeneity,concurrency,detail)):
            fig=create(metrics,data)
            if args.audit_scripts:
                require_matplotlib_panel_alignment(fig,json_out=HERE/'qa'/(file+'.alignment.json'),strict=True)
            base=HERE/'figures'/file
            fig.savefig(base.with_suffix('.pdf'))
            fig.savefig(base.with_suffix('.svg'))
            fig.savefig(base.with_suffix('.png'),dpi=300)
            plt.close(fig)


if __name__=='__main__':
    main()
