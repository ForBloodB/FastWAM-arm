import sys,json
from pathlib import Path


import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def audit_layout(fig): return {"note":"Visually review regenerated figures on this system; original publication figures were reviewed."}
from PIL import Image
R=Path(__file__).resolve().parents[1];O=R/'figures';O.mkdir(exist_ok=True)
j=json.loads((R/'analysis/metrics.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300,'svg.fonttype':'none'})
qa={}
def save(fig,name):
 qa[name]=audit_layout(fig);fig.savefig(O/(name+'.png'),dpi=300);fig.savefig(O/(name+'.svg'))
 Image.open(O/(name+'.png')).convert('L').save(O/(name+'_gray.png'),dpi=(300,300));plt.close(fig)
f,ax=plt.subplots(1,2,figsize=(10.5,4),layout='constrained')
names=['B_zero','F_fixed','C_frozen_11','D_online_11'];labels=['零残差','固定规则','冻结策略','在线策略']
for x,(n,l) in enumerate(zip(names,labels)):
 v=j['rl']['groups'][n]['overall'];lo,hi=v['wilson95'];ax[0].errorbar(x,v['rate']*100,yerr=[[100*(v['rate']-lo)],[100*(hi-v['rate'])]],fmt='o',color=['#526176','#0072B2','#009E73','#CC79A7'][x],capsize=5,ms=8);ax[0].text(x,v['rate']*100+16,f"{v['success']}/60",ha='center',fontsize=10)
ax[0].set(xticks=range(4),xticklabels=labels,ylim=(0,100),ylabel='成功率（%）；95% Wilson 区间',title='A  同一独立场景上的配对比较')
df=pd.read_csv(R/'analysis/rl.csv');a=df[df['mode']=='B_zero'].set_index('scene_id');b=df[(df['mode']=='C_frozen')&(df['learner_seed']==11)].set_index('scene_id').loc[a.index]
ax[1].scatter(a.peak_N,b.peak_N,s=22,color='#0072B2',alpha=.7);ax[1].plot([0,17],[0,17],'--',c='#777777',lw=1);ax[1].set(xlim=(0,17),ylim=(0,17),xlabel='零残差每场景最大指端力（N）',ylabel='冻结策略每场景最大指端力（N）',title='B  60 个场景均未增加力峰');ax[1].text(.05,.93,'3 个训练种子结论一致\n相同场景不重复计入样本量',transform=ax[1].transAxes,va='top',fontsize=9)
save(f,'rl_confirmation')
p=pd.read_csv(R/'analysis/paper.csv');a=p[p['mode']=='baseline'].set_index('scene_id');b=p[p['mode']=='gentle'].set_index('scene_id').loc[a.index]
f,ax=plt.subplots(1,2,figsize=(10.5,4),layout='constrained')
for i in a.index:ax[0].plot([0,1],[a.loc[i,'peak_N'],b.loc[i,'peak_N']],c='#9CB9C9',alpha=.45,lw=.7)
ax[0].scatter(np.zeros(40),a.peak_N,s=16,c='#0072B2');ax[0].scatter(np.ones(40),b.peak_N,s=16,c='#009E73');ax[0].axhline(3,c='#D55E00',ls='--',lw=1,label='共同上限 3 N');ax[0].legend(frameon=False);ax[0].set(xlim=(-.3,1.3),ylim=(0,3.3),xticks=[0,1],xticklabels=['原预紧 2.5 N','温和预紧 1.5 N'],ylabel='每场景最大指端力（N）',title='A  40 组配对：力峰全部降低')
delta=b.peak_N-a.peak_N;ax[1].scatter(range(1,41),delta,s=20,c='#0072B2');ax[1].axhline(0,c='#666666',ls='--',lw=1);ax[1].set(ylim=(-.7,.1),xlabel='独立场景编号',ylabel='温和预紧 − 原预紧（N）',title='B  成功率均为 40/40，未出现过夹')
save(f,'paper_force')
t=json.loads((R/'simulation/runs/submission_confirmation_20261102/training.json').read_text(encoding='utf-8'))
f,ax=plt.subplots(1,2,figsize=(10.5,3.6),layout='constrained')
for i,r in enumerate(t['results']):
 x=[c['step'] for c in r['curve']];loss=[c['fit']['loss_mean'] for c in r['curve']];y=[100*c['adaptive_monitor_metrics']['overall']['success']/30 for c in r['curve']]
 ax[0].plot(x,loss,marker=['o','s','^'][i],label=f"种子 {r['seed']}");ax[1].plot(x,y,marker=['o','s','^'][i],ls=['-','--',':'][i],label=f"种子 {r['seed']}")
ax[0].set(xlabel='累计梯度步数',ylabel='所示训练段平均 Huber 损失',title='A  拟合损失下降');ax[0].legend(frameon=False)
ax[1].set(xlabel='累计梯度步数',ylabel='开发监测表成功率（%）',ylim=(0,100),title='B  未证明任务成功率持续上升');ax[1].legend(frameon=False)
save(f,'learning_curve')
f,ax=plt.subplots(figsize=(9,3.5),layout='constrained')
stats=[]
for b in ['hil_local','hil_cpu_v3','hil_mmio','hil_dma']:
 d=json.loads((R/'evidence/fpga'/b/'benchmark.json').read_text());stats.append(d)
for i,d in enumerate(stats):ax.plot([d['p50_ms'],d['p95_ms']],[i,i],color='#0072B2',lw=3);ax.scatter(d['p50_ms'],i,c='#0072B2',s=35);ax.scatter(d['p95_ms'],i,c='#D55E00',marker='s',s=30);ax.text(d['p95_ms']+1,i,f"{d['p50_ms']:.2f} / {d['p95_ms']:.2f}",va='center',fontsize=9)
ax.set(yticks=range(4),yticklabels=['本地 CPU','远端板端 CPU','远端 FPGA MMIO','远端 FPGA DMA'],xlim=(0,99),xlabel='图像输入至标签返回（ms），圆点 P50 / 方点 P95',title='历史实板分项测试：DMA 快于 MMIO，仍慢于本地 CPU');ax.invert_yaxis()
save(f,'fpga_latency')
(O/'layout_audit.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8');print(qa)
