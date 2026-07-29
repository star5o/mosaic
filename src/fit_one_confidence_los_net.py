#!/usr/bin/env python3
"""Train the fixed third-party LOS-Net reproduction on development only."""
from __future__ import annotations
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from evaluate_one_confidence_five_model import choose_threshold

class LOSNet(nn.Module):
 def __init__(self,topk,vocab,maxlen,hidden=128,heads=8,layers=1,dropout=0.0):
  super().__init__();half=hidden//2;self.vocab=vocab;self.proj=nn.Linear(topk,half);self.p_atp=nn.Parameter(torch.randn(1,1,half));self.p_rank=nn.Parameter(torch.randn(1,1,half));self.cls=nn.Parameter(torch.randn(1,1,hidden));self.pos=nn.Embedding(maxlen+1,hidden);enc=nn.TransformerEncoderLayer(hidden,heads,hidden,dropout,batch_first=True);self.enc=nn.TransformerEncoder(enc,layers);self.head=nn.Linear(hidden,1)
 def forward(self,tds,atp,rank):
  a=atp.unsqueeze(-1);rank_scale=2*(.5-rank.float().unsqueeze(-1)/self.vocab);x=torch.cat((self.proj(tds.float()),a*self.p_atp+a*rank_scale*self.p_rank),-1);b,n,_=x.shape;x=torch.cat((self.cls.expand(b,-1,-1),x),1);x=x+self.pos(torch.arange(n+1,device=x.device))[None];return self.head(self.enc(x)[:,0]).squeeze(-1)

def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
def fit(model,data,indices,epochs,device):
 ds=TensorDataset(*[x[indices] for x in data]);loader=DataLoader(ds,batch_size=64,shuffle=True);opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0.0);loss=nn.BCEWithLogitsLoss()
 for _ in range(epochs):
  model.train()
  for t,a,r,y in loader:
   t,a,r,y=t.to(device),a.to(device),r.to(device),y.to(device);opt.zero_grad();l=loss(model(t,a,r),y);l.backward();opt.step()
 return model
def predict(model,data,indices,device):
 model.eval();out=[]
 with torch.inference_mode():
  for off in range(0,len(indices),64):
   ix=indices[off:off+64];out.append(torch.sigmoid(model(data[0][ix].to(device),data[1][ix].to(device),data[2][ix].to(device))).cpu())
 return torch.cat(out).numpy()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--features',type=Path,required=True);ap.add_argument('--output-json',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--seed',type=int,default=2026072905);args=ap.parse_args();seed_all(args.seed);x=torch.load(args.features,map_location='cpu',weights_only=False);y=torch.tensor([1. if z=='HCE' else 0. for z in x['labels']]);dev=np.asarray([s=='development' for s in x['splits']]);val=~dev;dev_idx=np.flatnonzero(dev);val_idx=np.flatnonzero(val);train_idx,hold_idx=train_test_split(dev_idx,test_size=.2,stratify=y[dev_idx].numpy(),random_state=args.seed);device='cuda' if torch.cuda.is_available() else 'cpu';data=(x['sorted_tds_normalized'],x['normalized_atp'],x['atp_r'],y);kwargs=dict(topk=x['topk'],vocab=x['vocab_size'],maxlen=x['sorted_tds_normalized'].shape[1]);model=LOSNet(**kwargs).to(device);best=(float('inf'),0,None);patience=0;lossfn=nn.BCEWithLogitsLoss()
 for epoch in range(1,101):
  fit(model,data,train_idx,1,device);scores=predict(model,data,hold_idx,device);loss=float(lossfn(torch.tensor(np.log(scores/(1-scores+1e-8)+1e-8)),y[hold_idx]).item())
  if loss<best[0]-1e-5:best=(loss,epoch,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()});patience=0
  else:patience+=1
  if patience>=10:break
 best_epoch=best[1];seed_all(args.seed);model=LOSNet(**kwargs).to(device);fit(model,data,dev_idx,best_epoch,device);scores=np.full(len(y),np.nan);scores[dev_idx]=predict(model,data,dev_idx,device);scores[val_idx]=predict(model,data,val_idx,device);op=choose_threshold(y[val_idx].numpy().astype(int),scores[val_idx]);records=[{'sample_id':sid,'split':split,'label':label,'score':float(score)} for sid,split,label,score in zip(x['row_ids'],x['splits'],x['labels'],scores)];payload={'schema_version':'one-confidence-los-net-third-party-reproduction-v1','cell':x['cell'],'implementation_status':'approximate_third_party_reproduction','architecture':kwargs|{'hidden':128,'heads':8,'layers':1,'dropout':0.0,'pool':'cls'},'training':{'best_epoch_dev_internal':best_epoch,'lr':1e-4,'batch_size':64,'seed':args.seed},'validation':op,'records':records};args.output_json.write_text(json.dumps(payload,indent=2)+'\n');torch.save({'metadata':payload|{'records':None},'state_dict':{k:v.cpu() for k,v in model.state_dict().items()}},args.checkpoint)
if __name__=='__main__':main()
