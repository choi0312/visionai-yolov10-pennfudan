"""Convert instance masks to YOLO boxes and retain a reproducible split manifest."""
import hashlib, json, random, shutil, zipfile, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image
BASE=Path(__file__).resolve().parent.parent

def prepare():
    archive=BASE/'data/PennFudanPed.zip'
    archive.parent.mkdir(exist_ok=True)
    if not archive.exists():
        with urllib.request.urlopen('https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip', timeout=120) as response:
            archive.write_bytes(response.read())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != '9095a9613c95586f1c7f2a327d454833d16e0f5e17e5f83d35027ffd315b48e2':
        raise ValueError('Dataset archive differs from the recorded experiment.')
    with zipfile.ZipFile(archive) as z:z.extractall(BASE/'data/raw')
    root=BASE/'data/raw/PennFudanPed';out=BASE/'data/pennfudan_yolo';out.mkdir(exist_ok=True)
    rng=random.Random(42);splits={k:[] for k in ('train','val','test')}
    for prefix,nt,nv in [('FudanPed',52,11),('PennPed',66,15)]:
        names=sorted(p.name for p in (root/'PNGImages').glob(prefix+'*.png'));rng.shuffle(names)
        splits['train']+=names[:nt];splits['val']+=names[nt:nt+nv];splits['test']+=names[nt+nv:]
    records=[];hashes={}
    for split,names in splits.items():
        for folder in ['images','labels']:(out/folder/split).mkdir(parents=True,exist_ok=True)
        for name in sorted(names):
            img_path=root/'PNGImages'/name;mask=np.array(Image.open(root/'PedMasks'/(Path(name).stem+'_mask.png')))
            w,h=Image.open(img_path).size;boxes=[];lines=[]
            for oid in np.unique(mask):
                if oid==0:continue
                ys,xs=np.where(mask==oid)
                x1,y1,x2,y2=map(int,[xs.min(),ys.min(),xs.max()+1,ys.max()+1])
                boxes.append([x1,y1,x2,y2]);xc,yc,bw,bh=(x1+x2)/(2*w),(y1+y2)/(2*h),(x2-x1)/w,(y2-y1)/h
                assert 0<=xc<=1 and 0<=yc<=1 and 0<bw<=1 and 0<bh<=1
                lines.append(f'0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}')
            digest=hashlib.sha256(img_path.read_bytes()).hexdigest()
            if digest in hashes:raise ValueError('Duplicate image: '+name)
            hashes[digest]=split
            shutil.copy2(img_path,out/'images'/split/name)
            (out/'labels'/split/(Path(name).stem+'.txt')).write_text('\n'.join(lines)+'\n')
            records.append({'file':name,'split':split,'width':w,'height':h,'boxes':boxes,'sha256':digest})
    annotation_counts=[]
    import re
    for path in (root/'Annotation').glob('*.txt'):
        annotation_counts.append(int(re.search(r'Objects with ground truth : (\d+)', path.read_text()).group(1)))
    assert len(records)==170 and sum(len(r['boxes']) for r in records)==sum(annotation_counts)==423
    (out/'data.yaml').write_text('path: '+str(out)+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: person\n')
    summary={'seed':42,'images':len(records),'objects':sum(len(r['boxes']) for r in records),'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'splits':{s:{'images':sum(r['split']==s for r in records),'objects':sum(len(r['boxes']) for r in records if r['split']==s)} for s in splits},'duplicate_images':0,
        'bbox_convention':'xyxy half-open pixel boxes converted to normalized center x y width height; max coordinate +1',
        'split_method':'Source-stratified image random split; no scene identity metadata; backgrounds may recur across splits.'}
    (out/'manifest.json').write_text(json.dumps(records,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
    with zipfile.ZipFile(BASE/'data/experiment_data.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(out))
    return records
if __name__=='__main__':prepare()
