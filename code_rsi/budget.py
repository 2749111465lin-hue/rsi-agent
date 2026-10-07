"""External append-only resource ledger with reservations and scope limits."""
import dataclasses,hashlib,json,math,pathlib,threading,time,uuid,os

def stable(x):return json.dumps(x,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(stable(x).encode()).hexdigest()
def save(p,x):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(stable(x),encoding='utf-8');tmp.replace(p)
class LimitExceeded(RuntimeError):pass

class Ledger:
 def __init__(self,path,limits):
  self.path=pathlib.Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
  self.limits=limits;self.lock=threading.RLock();self.used={};self.pending={};self.events=[];self.stopped=False
  for scope,ls in limits.items():
   if any(not isinstance(v,(float,int)) or isinstance(v,bool) or not math.isfinite(v) or v<0 for v in ls.values()):raise ValueError('invalid limit')
  if self.path.exists():
   for line in self.path.read_text(encoding='utf-8').splitlines():self._apply(json.loads(line))
   if self.pending or self.stopped:raise LimitExceeded('unresolved reservations or overrun; reconcile before resume')
 def _add(self,scope,amount):
  u=self.used.setdefault(scope,{})
  for k,v in amount.items():u[k]=u.get(k,0)+v
 def _apply(self,e):
  self.events.append(e)
  if e['event']=='reserve':self.pending[e['id']]=e
  elif e['event']=='settle':
   p=self.pending.pop(e['id'])
   for s in p['scopes']:self._add(s,e['actual'])
   if e.get('reservation_exceeded'):self.stopped=True
 def _write(self,e):
  with self.path.open('a',encoding='utf-8') as f:f.write(stable(e)+'\n');f.flush();os.fsync(f.fileno())
  self._apply(e)
 def reserve(self,scopes,amount,metadata=None):
  with self.lock:
   if self.stopped:raise LimitExceeded('ledger stopped after overrun')
   if len(scopes)!=len(set(scopes)) or not scopes:raise ValueError('distinct scopes required')
   if any(not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or v<0 for v in amount.values()):raise ValueError('invalid amount')
   for s in scopes:
    if s not in self.limits:raise ValueError('unregistered scope: '+s)
    for k,v in amount.items():
     held=sum(e['amount'].get(k,0) for e in self.pending.values() if s in e['scopes'])
     if self.used.get(s,{}).get(k,0)+held+v>self.limits[s].get(k,math.inf)+1e-10:raise LimitExceeded(s+':'+k)
   rid=uuid.uuid4().hex;self._write({'event':'reserve','id':rid,'time':time.time(),'scopes':scopes,'amount':amount,'metadata':metadata or {}});return rid
 def settle(self,rid,actual=None):
  with self.lock:
   p=self.pending[rid];a=p['amount'] if actual is None else actual
   if any(not isinstance(v,(float,int)) or isinstance(v,bool) or not math.isfinite(v) or v<0 for v in a.values()):raise ValueError('invalid actual')
   over=any(v>p['amount'].get(k,0)+1e-9 for k,v in a.items())
   self._write({'event':'settle','id':rid,'time':time.time(),'actual':a,'reservation_exceeded':over})
   if over:raise LimitExceeded('provider exceeded conservative reservation; stop')
 def summary(self):return {'used':self.used,'pending':len(self.pending),'alerts':[s for s,u in self.used.items() if 'cny' in u and self.limits[s].get('cny',0)>0 and u['cny']>=.8*self.limits[s]['cny']]}

HIGH={'diagnose':{'input':16000,'output':4000},'write':{'input':32000,'output':12000},'repair1':{'input':16000,'output':8000},'repair2':{'input':16000,'output':8000}}
LOW={k:{m:v//4 for m,v in d.items()} for k,d in HIGH.items()}
def session_limits(session,high=True):
 stages=HIGH if high else LOW
 out={session:{'input':sum(v['input'] for v in stages.values()),'output':sum(v['output'] for v in stages.values()),'calls':12}}
 out.update({session+'/'+k:{**v,'calls':12} for k,v in stages.items()});return out

def question_limits(scope):return {scope:{'input':80000,'output':8000,'calls':6,'model_invocations':6,'retrieval':8,'seconds':180,'index_bytes':67108864}}
