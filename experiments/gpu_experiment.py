"""YOLOv10n HF weights, baseline, 30-epoch finetuning, held-out COCO evaluation."""
import contextlib, hashlib, io, json, os, platform, time, zipfile
from pathlib import Path
import numpy as np
import torch, ultralytics, yaml
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
ROOT=Path(os.environ.get('VISIONAI_ROOT', '/workspace/visionai'));ROOT.mkdir(exist_ok=True);os.chdir(ROOT)
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
REV='3b91f578c5b492080f191a26aa35711bb44f608c'

def load_hf():
    path=hf_hub_download('jameslahm/yolov10n','model.safetensors',revision=REV)
    cfg=json.loads(Path(hf_hub_download('jameslahm/yolov10n','config.json',revision=REV)).read_text())
    model=YOLO('yolov10n.yaml');state=load_file(path)
    expected=set(model.model.state_dict());keys=set(state)
    if keys!=expected:
        candidates=[{k.removeprefix('model.'):v for k,v in state.items()},{'model.'+k:v for k,v in state.items()}]
        matches=[s for s in candidates if set(s)==expected]
        if len(matches)!=1:raise ValueError(f'HF key mismatch: {list(keys)[:5]} vs {list(expected)[:5]}')
        state=matches[0]
    model.model.load_state_dict(state,strict=True);model.model.names={int(k):v for k,v in cfg['names'].items()}
    model.save('hf_pretrained.pt')
    return YOLO('hf_pretrained.pt'),hashlib.sha256(Path(path).read_bytes()).hexdigest()

def coco_gt(records):
    gt={'info':{'description':'PennFudan masks to half-open boxes'},'licenses':[],'images':[],'annotations':[], 'categories':[{'id':1,'name':'person'}]}
    aid=1
    for iid,r in enumerate(records,1):
        gt['images'].append({'id':iid,'file_name':r['file'],'width':r['width'],'height':r['height']})
        for x1,y1,x2,y2 in r['boxes']:
            gt['annotations'].append({'id':aid,'image_id':iid,'category_id':1,'bbox':[x1,y1,x2-x1,y2-y1],'area':(x2-x1)*(y2-y1),'iscrowd':0});aid+=1
    return gt

def iou(a,b):
    x1,y1=np.maximum(a[:2],b[:2]);x2,y2=np.minimum(a[2:],b[2:]);inter=max(0,x2-x1)*max(0,y2-y1)
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-12)

def evaluate(model,records,tag):
    sources=[str(ROOT/'dataset/images/test'/r['file']) for r in records]
    model.predict(sources[0],imgsz=640,device=0,verbose=False)
    started=time.perf_counter()
    predictions=[model.predict(source,imgsz=640,conf=0.001,classes=[0],max_det=100,device=0,batch=1,half=False,verbose=False)[0] for source in sources]
    elapsed=time.perf_counter()-started
    detections=[];detail=[];tp=fp=fn=0
    for iid,(r,p) in enumerate(zip(records,predictions),1):
        a=p.boxes.xyxy.cpu().numpy();scores=p.boxes.conf.cpu().numpy();matched=set();itp=ifp=0
        for box,score in zip(a,scores):
            detections.append({'image_id':iid,'category_id':1,'bbox':[float(box[0]),float(box[1]),float(box[2]-box[0]),float(box[3]-box[1])],'score':float(score)})
        for j in np.argsort(-scores):
            if scores[j]<0.25:continue
            overlaps=[iou(a[j],np.array(b)) if k not in matched else -1 for k,b in enumerate(r['boxes'])]
            k=int(np.argmax(overlaps)) if overlaps else -1
            if k>=0 and overlaps[k]>=0.5:itp+=1;matched.add(k)
            else:ifp+=1
        ifn=len(r['boxes'])-itp;tp+=itp;fp+=ifp;fn+=ifn
        detail.append({'file':r['file'],'gt':r['boxes'],'pred_boxes':a.tolist(),'scores':scores.tolist(),'tp':itp,'fp':ifp,'fn':ifn,'speed':p.speed})
    gt=coco_gt(records);(OUT/'test_ground_truth.json').write_text(json.dumps(gt))
    with contextlib.redirect_stdout(io.StringIO()) as stream:
        coco=COCO(str(OUT/'test_ground_truth.json'));dt=coco.loadRes(detections);ev=COCOeval(coco,dt,'bbox');ev.evaluate();ev.accumulate();ev.summarize()
    (OUT/(tag+'_coco_log.txt')).write_text(stream.getvalue())
    metrics={'mAP50_95':float(ev.stats[0]),'mAP50':float(ev.stats[1]),'mAP75':float(ev.stats[2]),'precision_at_025':tp/(tp+fp) if tp+fp else 0,'recall_at_025':tp/(tp+fn) if tp+fn else 0,
        'TP':tp,'FP':fp,'FN':fn,'test_images':len(records),'test_objects':sum(len(r['boxes']) for r in records),'predict_wall_s':elapsed,
        'gpu_inference_ms_image':float(np.mean([p.speed['inference'] for p in predictions])),'evaluation_conf_floor':0.001,'operating_conf':0.25,'operating_iou':0.5,
        'coco_max_dets':[1,10,100],'prediction_batch_size':1,'precision_recall_matching':'score-ordered one-to-one greedy matching, IoU >= .5'}
    (OUT/(tag+'_metrics.json')).write_text(json.dumps(metrics,indent=2));(OUT/(tag+'_predictions.json')).write_text(json.dumps(detail))
    pr=ev.eval['precision'][:, :, 0, 0, 2]
    np.savez(OUT/(tag+'_pr.npz'),recall=ev.params.recThrs,precision50=pr[0],precision_mean=np.mean(np.where(pr<0,np.nan,pr),axis=0))
    print(tag.upper(),json.dumps(metrics),flush=True)
    return metrics

def main():
    with zipfile.ZipFile('experiment_data.zip') as z:z.extractall('dataset')
    cfg={'path':str(ROOT/'dataset'),'train':'images/train','val':'images/val','test':'images/test','names':{0:'person'}}
    Path('dataset/data.yaml').write_text(yaml.safe_dump(cfg))
    manifest=json.loads(Path('dataset/manifest.json').read_text());records=[r for r in manifest if r['split']=='test']
    environment={'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'ultralytics':ultralytics.__version__,'python':platform.python_version(),'hf_repo':'jameslahm/yolov10n','hf_revision':REV,'seed':42,'imgsz':640,'epochs':30,'batch':16,'optimizer':'AdamW','lr0':0.001,'pretrained_classes':80,'finetuned_classes':1,'nms_free':True}
    (OUT/'environment.json').write_text(json.dumps(environment,indent=2));print('ENVIRONMENT',json.dumps(environment),flush=True)
    model,sha=load_hf();environment['weights_sha256']=sha;(OUT/'environment.json').write_text(json.dumps(environment,indent=2))
    evaluate(model,records,'before')
    del model;torch.cuda.empty_cache();model=YOLO('hf_pretrained.pt')
    started=time.perf_counter()
    model.train(data='dataset/data.yaml',epochs=30,imgsz=640,batch=16,device=0,seed=42,deterministic=True,workers=4,
        optimizer='AdamW',lr0=0.001,lrf=0.01,weight_decay=0.0005,patience=30,amp=False,pretrained=True,
        project=str(OUT),name='train',exist_ok=True,plots=True,save=True,verbose=False,close_mosaic=10)
    environment['training_wall_s']=time.perf_counter()-started
    best=YOLO(str(OUT/'train/weights/best.pt'));evaluate(best,records,'after')
    (OUT/'environment.json').write_text(json.dumps(environment,indent=2))
    print('EXPERIMENT_COMPLETE',json.dumps(environment),flush=True)

if __name__=='__main__':main()
