"""Local Runpod REST client; credentials never enter logs or GPU environment."""
import json, os, time, urllib.request, urllib.error, argparse
from pathlib import Path
BASE = Path(__file__).resolve().parent.parent

def api(method, path, payload=None):
    key = os.environ.get('RUNPOD_API_KEY')
    if not key:
        raise RuntimeError('Set RUNPOD_API_KEY in the local process environment; do not commit a key.')
    req = urllib.request.Request('https://rest.runpod.io/v1/'+path,
        data=json.dumps(payload).encode() if payload is not None else None,
        method=method, headers={'Authorization':'Bearer '+key, 'Content-Type':'application/json', 'User-Agent':'VisionAI-Practice/1.0'})
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            data = response.read()
            return json.loads(data) if data else {}
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'Runpod HTTP {exc.code}: '+exc.read().decode()[:500].replace(key,'[REDACTED]')) from None

def safe_summary(p):
    return {k:p.get(k) for k in ('id','name','desiredStatus','costPerHr','publicIp','portMappings','gpu','machineId')}

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['create','status','delete','watchdog']);parser.add_argument('--budget-usd',type=float,default=5.0);parser.add_argument('--max-runtime-minutes',type=float,default=30.0);args=parser.parse_args()
    (BASE/'.private').mkdir(mode=0o700,exist_ok=True)
    if args.budget_usd<=0 or args.max_runtime_minutes<=0:raise ValueError('Budget and runtime must be positive.')
    state_path=BASE/'.private/experiment_pod.json'
    if args.action=='create':
        if state_path.exists():
            if not json.loads(state_path.read_text()).get('deleted_at'):
                raise RuntimeError('Existing live experiment state; inspect before creating another Pod.')
            state_path.replace(BASE/'.private/previous_experiment_pod.json')
        public_key=(BASE/'.private/visionai_runpod.pub').read_text().strip()
        startup='mkdir -p /root/.ssh /run/sshd; printf "%s\\n" "$PUBLIC_KEY" > /root/.ssh/authorized_keys; chmod 700 /root/.ssh; chmod 600 /root/.ssh/authorized_keys; ssh-keygen -A; /usr/sbin/sshd; sleep 5400'
        p=api('POST','pods',{'name':'visionai-yolov10-pennfudan-20261010','cloudType':'SECURE','computeType':'GPU','gpuCount':1,
            'gpuTypeIds':['NVIDIA GeForce RTX 4090'],'gpuTypePriority':'availability',
            'imageName':'runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404','containerDiskInGb':30,'volumeInGb':0,
            'ports':['22/tcp'],'supportPublicIp':True,'env':{'PUBLIC_KEY':public_key},'dockerStartCmd':['bash','-lc',startup]})
        state={'pod':p,'created_at':time.time(),'max_dollars':args.budget_usd,'deadline':time.time()+min(args.max_runtime_minutes*60,args.budget_usd/max(float(p.get('costPerHr',1.5)),0.01)*3600-60)}
        state_path.write_text(json.dumps(state));state_path.chmod(0o600)
        print(json.dumps(safe_summary(p)))
        if not p.get('costPerHr') or float(p['costPerHr'])>1.5 or state['deadline']<=time.time():
            api('DELETE','pods/'+p['id']);raise RuntimeError('Hourly rate above experiment ceiling; new Pod deleted.')
    else:
        state=json.loads(state_path.read_text());pod_id=state['pod']['id']
        if args.action=='status':
            p=api('GET','pods/'+pod_id);state['pod']=p;state_path.write_text(json.dumps(state));print(json.dumps(safe_summary(p)))
        elif args.action=='delete':
            api('DELETE','pods/'+pod_id);state['deleted_at']=time.time();state_path.write_text(json.dumps(state));print('Experiment Pod deleted.')
        elif args.action=='watchdog':
            while time.time()<state['deadline']:
                time.sleep(min(30,state['deadline']-time.time()))
                if json.loads(state_path.read_text()).get('deleted_at'):raise SystemExit(0)
            api('DELETE','pods/'+pod_id);print('Budget watchdog deleted experiment Pod.')
