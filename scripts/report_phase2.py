"""Aggregate real-data results and draw publication-safe scalar previews."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path('.mplconfig').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE=Path('results/phase2')
COLORS={'analytic':'#0072B2','midpoint':'#009E73'}
LABEL={'analytic':'Analytic','midpoint':'10 midpoints'}


def read(path):return json.loads(Path(path).read_text())


def aggregate():
    runs=[]
    for folder in sorted((BASE/'runs').iterdir()):
        if (folder/'summary.json').exists():runs.append(dict(read(folder/'summary.json'),run=folder.name))
    assert len(runs)==28,f'Expected28 completed runs, got{len(runs)}'
    groups=[];pairs=[]
    for seq in ['freiburg1_xyz','freiburg1_desk']:
        for track in ['native','controlled']:
            for method in ['analytic','midpoint']:
                selected=[r for r in runs if r['device']=='mps' and r['sequence']==seq and r['track']==track and r['method']==method]
                assert sorted(r['seed'] for r in selected)==[0,1,2]
                row={'sequence':seq,'track':track,'method':method,'device':'mps','seeds':[0,1,2]}
                for key in ['reference_psnr','reference_ssim_mean','depth_rmse_m','alpha90_coverage','full_native_valid_center_psnr','observation_psnr']:
                    values=[r['final'][key] for r in selected]
                    row[key+'_mean']=float(np.mean(values));row[key+'_sd']=float(np.std(values,ddof=1))
                groups.append(row)
            deltas=[]
            for seed in [0,1,2]:
                chosen={r['method']:r for r in runs if r['device']=='mps' and r['sequence']==seq and r['track']==track and r['seed']==seed}
                deltas.append(chosen['midpoint']['final']['reference_psnr']-chosen['analytic']['final']['reference_psnr'])
            pairs.append({'sequence':seq,'track':track,'metric':'midpoint minus analytic reference PSNR(dB)',
                          'per_seed':deltas,'mean':float(np.mean(deltas)),'sample_sd':float(np.std(deltas,ddof=1)),
                          'midpoint_wins':sum(x>0 for x in deltas),'analytic_wins':sum(x<0 for x in deltas)})
    profiles=[read(p) for p in sorted((BASE/'profiles').glob('*.json'))]
    final_profiles=[read(p) for p in sorted((BASE/'final_profiles').glob('*.json'))]
    assert len(final_profiles)==12
    replay=read(BASE/'replay.json')
    payload={'reviewed':'2026-11-17','runs':runs,'mps_groups':groups,'paired_differences':pairs,'isolated_profiles':profiles,'final_model_profiles':final_profiles,'replay':replay,
             'scope':'Two15-frame real RGB-D subsets, each2400Gaussians at96x72; known mocap camera;150steps; no original paper table matched'}
    (BASE/'aggregate.json').write_text(json.dumps(payload,indent=2))
    return payload


def main():
    data=aggregate();runs=data['runs'];groups=data['mps_groups'];profiles=data['isolated_profiles'];final_profiles=data['final_model_profiles']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for track,ax in zip(['native','controlled'],axes[0]):
        for j,seq in enumerate(['freiburg1_xyz','freiburg1_desk']):
            for k,method in enumerate(['analytic','midpoint']):
                r=next(g for g in groups if g['sequence']==seq and g['track']==track and g['method']==method)
                x=j+(-.12 if k==0 else .12)
                ax.errorbar(x,r['reference_psnr_mean'],yerr=r['reference_psnr_sd'],fmt='o',color=COLORS[method],capsize=4,
                            label=LABEL[method] if j==0 else None)
                points=[r['final']['reference_psnr'] for r in runs if r['device']=='mps' and r['sequence']==seq and r['track']==track and r['method']==method]
                ax.scatter(x+np.linspace(-.04,.04,3),points,s=14,color=COLORS[method],alpha=.65)
        ax.set(xticks=[0,1],xticklabels=['fr1/xyz','fr1/desk'],ylabel=('Native observation PSNR(dB)' if track=='native' else 'Controlled-blur ROI center PSNR(dB)'))
        ax.legend()
    names=['dense','tiled','checkpoint'];labels=['Dense','Tiles, retained','Tiles, recomputed']
    for ax,key,ylabel in [(axes[1,0],'median_total_seconds','MPS forward + backward (s)'),
                          (axes[1,1],'memory','MPS sampled live peak (MiB)')]:
        for k,method in enumerate(['analytic','midpoint']):
            vals=[]
            for backend in names:
                r=next(p for p in profiles if p['device']=='mps' and p['backend']==backend and p['method']==method and p['tile']==32)
                vals.append(r[key] if key!='memory' else r['memory']['mps_sampled_peak_live_bytes']/2**20)
            ax.bar(np.arange(3)+(k-.5)*.28,vals,width=.28,color=COLORS[method],label=LABEL[method])
        ax.set(xticks=range(3),xticklabels=labels,ylabel=ylabel,yscale='log');ax.tick_params(axis='x',labelsize=9);ax.legend()
    for i,ax in enumerate(axes.flat):ax.text(-.12,1.03,f'({chr(97+i)})',transform=ax.transAxes,fontweight='bold')
    for ext in ['png','pdf']:fig.savefig(BASE/f'comparison.{ext}',dpi=180)
    plt.close(fig)
    # CPU and MPS memory are deliberately on separate axes: unified allocations
    # overlap and cannot be summed into a synthetic total.
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for col,device in enumerate(['cpu','mps']):
        for k,method in enumerate(['analytic','midpoint']):
            timings=[];memory=[]
            for backend in names:
                r=next(p for p in profiles if p['device']==device and p['backend']==backend and p['method']==method and p['tile']==32)
                timings.append(r['median_total_seconds']);memory.append(r['memory']['process_peak_rss_bytes' if device=='cpu' else 'mps_sampled_peak_live_bytes']/2**20)
            x=np.arange(3)+(k-.5)*.28
            axes[0,col].bar(x,timings,width=.28,color=COLORS[method],label=LABEL[method]);axes[1,col].bar(x,memory,width=.28,color=COLORS[method])
        axes[0,col].set(ylabel=device.upper()+' forward + backward(s)',yscale='log');axes[0,col].legend()
        axes[1,col].set(ylabel='CPU process peak RSS(MiB)' if device=='cpu' else 'MPS sampled live peak(MiB)',yscale='log')
        for ax in axes[:,col]:ax.set(xticks=range(3),xticklabels=labels);ax.tick_params(axis='x',labelsize=9)
    for ext in ['png','pdf']:fig.savefig(BASE/f'cpu_gpu_profile.{ext}',dpi=180)
    plt.close(fig)
    # Local qualitative figure: raw/derived TUM pixels stay out of public collection.
    fig,axes=plt.subplots(4,5,figsize=(13,9),layout='constrained')
    for row,(seq,track) in enumerate([(s,t) for s in ['freiburg1_xyz','freiburg1_desk'] for t in ['native','controlled']]):
        with np.load(f'data/prepared/{seq}.npz') as src:
            ref=src['rgb'][2];obs=src['rgb'][2] if track=='native' else src['blur'][2]
            mask=src['native_mask'][2] if track=='native' else src['controlled_mask'][2]
        images=[obs,ref]
        for method in ['analytic','midpoint']:
            stem=f'{seq}_{track}_{method}_mps_seed0'
            with np.load(BASE/'predictions'/f'{stem}_predictions.npz') as pred:
                images.append(pred['blur'][0] if track=='native' else pred['heldout'][0,...,:3])
        images.append(mask.astype(float))
        for col,image in enumerate(images):
            ax=axes[row,col];ax.imshow(np.clip(image,0,1),cmap='gray' if col==4 else None,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(['Observation','Original RGB reference','Analytic','10 midpoints','Fixed evaluation mask'][col],fontsize=10)
            if col==0:ax.set_ylabel(seq.replace('freiburg1_','fr1/')+'\n'+track)
    for ext in ['png','pdf']:fig.savefig(BASE/f'qualitative_real_rgbd.{ext}',dpi=180)
    plt.close(fig)
    audit=read(BASE/'data_renderer_audit.json')
    source={seq:read(BASE/f'{seq}_source.json') for seq in ['freiburg1_xyz','freiburg1_desk']}
    lines=['# RoMo-GS 第二阶段：分块渲染与真实 RGB-D 小规模重建','',
           '审阅日期：2026-11-17。独立实现；本阶段完成分块渲染、公开TUM RGB-D接入、解析积分/中点采样的局部重建与资源对照。未复现完整在线系统或论文结果表。','',
           '## 实验范围','',
           '- TUM `freiburg1_xyz`、`freiburg1_desk`各15帧：10训练、5留出；每序列2400高斯、96×72、固定mocap相机、150步。MPS两方法×两轨道×两序列×3种子=24次；CPU受控轨道种子0的4次配对验证，共28次。',
           '- 原生轨道：曝光1/60s是披露的假设，评估预测观测对原始RGB，不能声称真实传感器去模糊成功。受控轨道：在真实RGB-D上添加0.08s已知模糊，独立深度重投影生成标签；原始RGB是添加模糊前参考，仍可能含原生模糊。',
           '- 初始化仅用训练RGB-D。受控轨道的颜色只从模糊训练观测取得；测试RGB/深度不进入初始化或优化。位置±3cm、尺度变化范围固定；是已知位姿的离线小规模重建，不是未知位姿在线SLAM。',
           '- 受控轨道只在固定、全曝光都有效的重投影ROI训练/主要评估，同时报告全部有效深度像素上的中心渲染PSNR，保留ROI外的不足。没有通过预测质量筛像素。','',
           '## 图像与几何质量','',
           '主表为MPS三个种子的均值±样本标准差。原生与受控轨道的参考和mask不同，不能横向比较绝对PSNR来宣称去模糊收益。','',
           '| 序列 | 轨道 | 方法 | 主要PSNR(dB) | SSIM | 全有效深度区域中心PSNR(dB) | 深度RMSE(m) | Alpha>0.9覆盖率 |',
           '|---|---|---|---:|---:|---:|---:|---:|']
    for r in groups:
        fmt=lambda key: f"{r[key+'_mean']:.4f} ± {r[key+'_sd']:.4f}"
        lines.append(f"| {r['sequence']} | {r['track']} | {LABEL[r['method']]} | {fmt('reference_psnr')} | {fmt('reference_ssim_mean')} | {fmt('full_native_valid_center_psnr')} | {fmt('depth_rmse_m')} | {fmt('alpha90_coverage')} |")
    lines+=['','配对差（10中点减解析积分，同种子）：','', '| 序列 | 轨道 | PSNR差均值±样本SD(dB) | 中点胜/解析胜 |','|---|---|---:|---:|']
    for p in data['paired_differences']:lines.append(f"| {p['sequence']} | {p['track']} | {p['mean']:+.5f} ± {p['sample_sd']:.5f} | {p['midpoint_wins']}/{p['analytic_wins']} |")
    lines+=['','小差异只能解释为这两个短片段、当前容量与固定训练预算下的观察；3个种子只扰动颜色初始化，不是跨场景统计保证。完整逐帧、初始/最终值和CPU控制结果见 `aggregate.json`。','',
            '初始化到最终状态的变化（MPS三个种子的均值；更低的深度RMSE更好）：','',
            '| 序列 | 轨道 | 模型 | 初始→最终主要PSNR(dB) | 初始→最终深度RMSE(m) |',
            '|---|---|---|---:|---:|']
    for r in groups:
        selected=[x for x in runs if x['device']=='mps' and all(x[k]==r[k] for k in ['sequence','track','method'])]
        pi=np.mean([x['initial']['reference_psnr'] for x in selected]);di=np.mean([x['initial']['depth_rmse_m'] for x in selected])
        lines.append(f"| {r['sequence']} | {r['track']} | {LABEL[r['method']]} | {pi:.4f} → {r['reference_psnr_mean']:.4f} | {di:.4f} → {r['depth_rmse_m_mean']:.4f} |")
    lines+=['','颜色拟合改善不能自动等同于几何改善；表中保留深度误差没有改善或变差的情况。','',
            '![Comparison](../results/phase2/comparison.png)','',
            '图：上左为原生RGB观测PSNR，上右为受控模糊固定ROI中心渲染PSNR（均3种子，点为单次结果，误差条为样本SD）；下排为2400高斯、96×72、单个真实初始化视图的隔离MPS前后向耗时与采样到的GPU活动内存峰值。公开预览只包含本地标量统计，不含原始RGB-D图像。','',
            '![Local qualitative comparison](../results/phase2/qualitative_real_rgbd.png)','',
            '本地定性图：MPS训练的种子0地图、固定第一个留出帧view2，使用保留的源设备渲染输出；4行分别为xyz/native、xyz/controlled、desk/native、desk/controlled；白色mask为主要评估区域。展示完整视图以暴露mask外重建缺口。原生图像没有可用于清晰去模糊验证的独立真值。该图含公开TUM素材的派生图像，只在本地实现保留，不同步到总览。','',
            '## 隔离速度与内存','',
            '每个case使用新进程；CPU1线程；1预热+3次重复，表中为完整前后向中位耗时，包含投影/CPU分块规划，对位置、协方差、颜色、不透明度求梯度。训练队列的并发耗时不作为这里的速度结论。','',
            '| 设备 | 模型 | 后端 | 前后向(s) | 进程RSS峰值(MiB) | MPS活动峰值(MiB，2ms采样) | 前向后autograd保留存储(MiB) |',
            '|---|---|---|---:|---:|---:|---:|']
    for device in ['cpu','mps']:
        for method in ['analytic','midpoint']:
            for backend in ['dense','tiled','checkpoint']:
                r=next(p for p in profiles if p['device']==device and p['method']==method and p['backend']==backend and p['tile']==32)
                mem=r['memory'];gpu=f"{mem['mps_sampled_peak_live_bytes']/2**20:.2f}" if device=='mps' else '—'
                lines.append(f"| {device.upper()} | {LABEL[method]} | {backend} | {r['median_total_seconds']:.4f} | {mem['process_peak_rss_bytes']/2**20:.2f} | {gpu} | {r['retained_forward_unique_storage_bytes']/2**20:.3f} |")
    lines+=['','`tiled`保留各块计算图；`checkpoint`按块重算。分块裁剪明显减少矩阵规模，重算进一步减少保留的反向中间结果，但增加计算开销；GPU上解析整图版可能与分块版一样快甚至更快。RSS、MPS活动/驱动内存口径不同，在统一内存上相互重叠，不能相加；2ms采样可能漏过短暂峰值。不是机器人系统FPS或论文A6000内核的速度复现。','',
            '分块大小16/32/48的全部20组case保留在 `profiles/`。主配置统一32，未按每台设备单独挑选最优块大小。','',
            '![CPU and GPU resources](../results/phase2/cpu_gpu_profile.png)','',
            '## 最终模型的实际渲染开销','',
            '另对全部12份种子0最终模型进行隔离测量：各自新进程，在5个留出视图上共15次无梯度推理计时；首个训练视图上3次前后向计时。各方法已经优化成不同地图，以下是实际最终状态的开销，不能当作同一地图上纯内核的严格对照。前后向为渲染器的5通道平方均值损失，不含完整优化器与SSIM步骤。','',
            '| 序列 | 轨道 | 设备 | 方法 | 推理中位(s) | 前后向中位(s) | 推理MPS峰值(MiB) | 前后向MPS峰值(MiB) |',
            '|---|---|---|---|---:|---:|---:|---:|']
    for p in final_profiles:
        im=p['inference_memory']['mps_sampled_peak_live_bytes']/2**20
        bm=p['forward_backward_memory']['mps_sampled_peak_live_bytes']/2**20
        lines.append(f"| {p['sequence']} | {p['track']} | {p['device'].upper()} | {LABEL[p['method']]} | {p['inference_median_seconds']:.4f} | {p['forward_backward_median_seconds']:.4f} | {im:.2f} | {bm:.2f} |")
    lines+=['','CPU行的MPS数值为0，CPU进程RSS峰值、各次原始计时、GPU驱动缓存及基线内存详见 `final_profiles/`。MPS内存是2ms采样到的峰值，不能视为精确硬件峰值；这些数值不能与RSS相加。','',
            '## 数值与数据核验','',
            f"- 16项测试通过，包括精确分块值/梯度、独立逐时刻采样对照、空块、相机后方裁剪以及颜色初始化不泄漏清晰/测试RGB。两个真实序列各384高斯的全图对照中，不裁剪的最大输出差 {max(r['variants']['exact']['image_max_absolute_error'] for r in audit['renderer']):.3e}；裁剪后的最大输出差 {max(r['variants']['culled']['image_max_absolute_error'] for r in audit['renderer']):.3e}，最大梯度相对L2误差 {max(r['variants']['culled']['gradient_relative_l2'] for r in audit['renderer']):.3e}。",
            f"- 4个固定真实帧的标签32/64次积分核验，ROI最大像素差 {max(r['32_vs64_max_abs'] for r in audit['labels']):.3e}；零运动重投影及32次缓存重建核验通过。",
            '- 训练/测试分离、融合点来源、预处理缓存SHA-256均已机械核验；最后的28份参数在CPU重放，重算图像与指标见 `replay.json`。','',
            '| 序列 | 原生有效深度像素范围 | 受控ROI像素范围 |','|---|---:|---:|']
    for seq,src in source.items():
        nf=[f['native_valid_fraction'] for f in src['frames']];cf=[f['controlled_valid_fraction'] for f in src['frames']]
        lines.append(f"| {seq} | {min(nf):.1%}–{max(nf):.1%} | {min(cf):.1%}–{max(cf):.1%} |")
    exceptions=[r for r in data['replay'] if r.get('strict_cpu_image_gate_passed') is False]
    if exceptions:
        largest=max(r['max_prediction_absolute_error'] for r in exceptions)
        diag=read(BASE/'replay_discrepancy_audit.json')['views'][0]
        all_diag=read(BASE/'replay_discrepancy_audit.json')
        lines+=['','### CPU/GPU重放差异（保留原门限）','',
                f"{len(exceptions)}份参数的CPU重放超过原2e−4像素门限，最大差 {largest:.6g}；该门限未被放宽。所有PSNR/SSIM/深度指标仍通过独立CPU重算，像素例外额外在原MPS设备重放核验。",
                f"定位个案 `freiburg1_desk_native_midpoint_mps_seed1`、view2：高斯653/751在CPU上深度相等、MPS上相差2.384e−7m，稳定排序结果交换。固定MPS的排序再用CPU投影，模糊RGB最大差降为 {diag['cpu_projection_with_gpu_depth_order_vs_gpu_rgb_max']:.3e}；对同一MPS投影用CPU合成，最大差 {diag['cpu_raster_of_gpu_projection_vs_gpu_rgb_max']:.3e}；原GPU重放与保存图像完全相同。",
                f"四个例外均逐一完成同样控制：所有源设备重放完全一致；统一深度顺序后的CPU/GPU模糊RGB最大差 {all_diag['max_cpu_rgb_error_with_gpu_depth_order']:.3e}。全部28份CPU重算PSNR的最大差仅 {max(r['metric_absolute_errors']['reference_psnr'] for r in data['replay']):.3e}dB。",
                '这是已核验的近深度排序不连续性，不是通过修改模型来贴合结果。诊断、原始失败记录和后续源设备验证分别保存在 `replay_discrepancy_audit.json`、`replay_first_attempt.log`、`replay.json`。跨设备位级确定性尚未建立。']
    lines+=['','受控ROI较窄是本阶段的重要局限，尤其desk；这里的结果不能代表整个原始640×480视频的清晰重建能力。','',
            '## 实现变化与保留的探索','',
            '- 5步烟雾检查发现像素尺度初始化的覆盖不足；保留其日志与源快照，之后采用只依赖训练几何的间距尺度，并以60步试跑核查。正式28次实验全部使用相同新规则。',
            '- 保留逐时刻循环版本的20组初始profile，再加入5时刻张量批处理；新实现通过独立逐时刻值/梯度核验。当前主表使用重跑的批处理版本，历史profile不混入正式统计。',
            '- 已确认的范围限制：固定150步预算、2400高斯、对角世界协方差、已知位姿与狭窄受控ROI，且ROI外存在可见缺口。标定/配准误差、位姿精度、光照或原生曝光对剩余误差的贡献尚未独立定位，不作为已确认原因。没有选择最优测试集checkpoint，也没有隐藏不利种子或ROI外结果。','',
            '## 重建与证据索引','',
            '- `docs/phase2_protocol.md`：完整设置与统计口径；`execution_plan.json`：训练前固定的计划和源文件哈希。',
            '- `*_source.json`：官方数据URL、档案/逐文件/缓存哈希、时间戳、曝光构造与mask范围；原始与预处理数据只保存在本地忽略目录。',
            '- `data_renderer_audit.json`、`profiles/`、`runs/*/{config,summary,trace}.json`、`runs/*/parameters.npz`：原始证据。',
            '- `aggregate.json`：28次结果与20组资源测量；`replay.json`：全部参数重放；PNG/PDF结果图均保留。',
            '- 总览只公开已审阅标量快照、报告和聚合图；本地完整代码尚未单独公开。公开快照不等于公开可独立重跑的实现。','',
            '数据来源：[TUM RGB-D官方序列](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download)与[格式/标定说明](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats)。本地TUM派生图像保留原始数据权利与署名，不随总览发布。','']
    Path('docs/phase2_results.md').write_text('\n'.join(lines))
    print(json.dumps({'runs':len(runs),'groups':groups,'pairs':data['paired_differences']},indent=2))


if __name__=='__main__':main()
