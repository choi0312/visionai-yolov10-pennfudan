"""Create report figures exclusively from dataset records and experiment artifacts."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
BASE=Path(__file__).resolve().parent.parent
ASSETS=BASE/'figures';ASSETS.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
COLORS=['#7c8796','#287fa3','#be713c']

def eda():
    rows=json.loads((BASE/'data/pennfudan_yolo/manifest.json').read_text())
    fig,ax=plt.subplots(1,3,figsize=(10.7,2.35),layout='constrained')
    for a in ax:a.grid(axis='y',alpha=.16);a.set_axisbelow(True)
    splits=['train','val','test'];counts=[sum(r['split']==s for r in rows) for s in splits];objects=[sum(len(r['boxes']) for r in rows if r['split']==s) for s in splits]
    x=np.arange(3);ax[0].bar(x-.18,counts,.36,label='Images',color=COLORS[1]);ax[0].bar(x+.18,objects,.36,label='Boxes',color=COLORS[2]);ax[0].set_xticks(x,splits);ax[0].legend(fontsize=8);ax[0].set_title('Dataset split');ax[0].set_ylim(0,360)
    for i,(n,o) in enumerate(zip(counts,objects)):ax[0].text(i-.18,n+7,str(n),ha='center',fontsize=8);ax[0].text(i+.18,o+7,str(o),ha='center',fontsize=8)
    train=[r for r in rows if r['split']=='train'];obj=[len(r['boxes']) for r in train]
    ax[1].hist(obj,bins=np.arange(.5,9.6,1),color=COLORS[1],rwidth=.8);ax[1].set_title('Persons per training image');ax[1].set_xlabel('Number of boxes');ax[1].set_ylabel('Images')
    area=[(b[2]-b[0])*(b[3]-b[1])/(r['width']*r['height']) for r in train for b in r['boxes']]
    ax[2].hist(area,bins=12,color=COLORS[2],rwidth=.9);ax[2].set_title('Training box size');ax[2].set_xlabel('Box area / image area');ax[2].set_ylabel('Boxes')
    fig.savefig(ASSETS/'eda.png',dpi=240);plt.close(fig)
    stats={'train_median_persons_per_image':float(np.median(obj)),'train_median_relative_box_area':float(np.median(area)),'train_min_relative_box_area':min(area),'train_max_relative_box_area':max(area)}
    (ASSETS/'eda_stats.json').write_text(json.dumps(stats));print(stats)

def results_figures():
    import csv
    R=BASE/'results'
    rows=list(csv.DictReader((R/'train/results.csv').open()));rows=[{k.strip():float(v) for k,v in r.items()} for r in rows]
    fig,ax=plt.subplots(1,3,figsize=(10.7,2.55),layout='constrained')
    epochs=[r['epoch']+1 if rows[0]['epoch']==0 else r['epoch'] for r in rows]
    for a in ax:a.grid(alpha=.18);a.set_xlabel('Epoch')
    for k,l,c in [('train/box_loss','Train',COLORS[1]),('val/box_loss','Validation',COLORS[2])]:ax[0].plot(epochs,[r[k] for r in rows],label=l,color=c)
    ax[0].set_title('Box loss');ax[0].legend(fontsize=8)
    for k,l,c in [('train/cls_loss','Train',COLORS[1]),('val/cls_loss','Validation',COLORS[2])]:ax[1].plot(epochs,[r[k] for r in rows],label=l,color=c)
    ax[1].set_title('Classification loss');ax[1].legend(fontsize=8)
    for k,l,c in [('metrics/mAP50(B)','mAP50',COLORS[1]),('metrics/mAP50-95(B)','mAP50-95',COLORS[2])]:ax[2].plot(epochs,[r[k] for r in rows],label=l,color=c)
    ax[2].set_title('Validation detection performance');ax[2].set_ylim(0,1.05);ax[2].legend(fontsize=8)
    fig.savefig(ASSETS/'curves.png',dpi=240);plt.close(fig)
    before=json.loads((R/'before_predictions.json').read_text());after=json.loads((R/'after_predictions.json').read_text());bd={r['file']:r for r in before}
    # Show one improvement example and the hardest post-finetuning example; describe selection explicitly.
    gain=sorted(after,key=lambda r:((bd[r['file']]['fn']+bd[r['file']]['fp'])-(r['fn']+r['fp']),len(r['gt'])),reverse=True)
    hard=sorted(after,key=lambda r:(r['fn']+r['fp'],len(r['gt'])),reverse=True)
    selected=[gain[0],next(r for r in hard if r['file']!=gain[0]['file'])]
    fig,ax=plt.subplots(2,3,figsize=(10.7,5.25),layout='constrained')
    from matplotlib.patches import Rectangle
    for i,r in enumerate(selected):
        img=Image.open(BASE/'data/pennfudan_yolo/images/test'/r['file'])
        for j in range(3):
            ax[i,j].imshow(img);ax[i,j].set_xticks([]);ax[i,j].set_yticks([])
            current=[r['gt'],bd[r['file']],r][j]
            if j==0:
                boxes=current;scores=[None]*len(boxes);color='#24974c';ttl='Ground truth'
            else:
                boxes=current['pred_boxes'];scores=current['scores'];color='#077cab';ttl='Before finetuning' if j==1 else 'After finetuning'
            for b,s in zip(boxes,scores):
                if s is not None and s<.25:continue
                x1,y1,x2,y2=b;ax[i,j].add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=color,linewidth=1.6))
                if s is not None:ax[i,j].text(x1,y1,f'{s:.2f}',fontsize=7,color='white',bbox={'facecolor':color,'alpha':.8,'pad':1})
            if j>0:ttl+=f"\nTP {current['tp']}  FP {current['fp']}  FN {current['fn']}"
            else:ttl+='\n'+r['file']
            ax[i,j].set_title(ttl,fontsize=9)
    fig.savefig(ASSETS/'predictions.png',dpi=220);plt.close(fig)
    (ASSETS/'selected_examples.json').write_text(json.dumps(selected))
    fig,ax=plt.subplots(1,2,figsize=(10.7,2.45),layout='constrained')
    for tag,label,c in [('before','Before',COLORS[0]),('after','After',COLORS[1])]:
        z=np.load(R/(tag+'_pr.npz'));ax[0].plot(z['recall'],z['precision50'],label=label,color=c)
        pred=json.loads((R/(tag+'_predictions.json')).read_text());thresholds=np.arange(.1,.91,.1);pp=[];rr=[]
        # PR graph comes from COCOeval. Threshold sweep uses the same one-to-one matching rule as the experiment.
        def iou(a,b):
            x1,y1=np.maximum(a[:2],b[:2]);x2,y2=np.minimum(a[2:],b[2:]);inter=max(0,x2-x1)*max(0,y2-y1)
            return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-12)
        for t in thresholds:
            tp=fp=fn=0
            for r in pred:
                matched=set()
                for j in np.argsort(-np.array(r['scores'])):
                    if r['scores'][j]<t:continue
                    ov=[iou(np.array(r['pred_boxes'][j]),np.array(b)) if k not in matched else -1 for k,b in enumerate(r['gt'])]
                    k=int(np.argmax(ov))
                    if ov[k]>=.5:tp+=1;matched.add(k)
                    else:fp+=1
                fn+=len(r['gt'])-len(matched)
            pp.append(tp/(tp+fp) if tp+fp else 0);rr.append(tp/(tp+fn))
        ax[1].plot(thresholds,rr,label=label+' recall',color=c);ax[1].plot(thresholds,pp,'--',label=label+' precision',color=c)
    for a in ax:a.grid(alpha=.18);a.set_ylim(0,1.05);a.legend(fontsize=8)
    ax[0].set_xlabel('Recall');ax[0].set_ylabel('Interpolated precision');ax[0].set_title('Test PR curve at IoU 0.50')
    ax[1].set_xlabel('Confidence threshold');ax[1].set_title('Test threshold sweep at IoU 0.50')
    fig.savefig(ASSETS/'pr.png',dpi=220);plt.close(fig)

if __name__=='__main__':
    eda()
    results_figures()
