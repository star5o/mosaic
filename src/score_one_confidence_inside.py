#!/usr/bin/env python3
"""Paper-formula INSIDE EigenScore without feature clipping (K=4 budget)."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
SYSTEM="Return only the requested factual answer entity or short value. Do not write a sentence, preamble, explanation, or reasoning."
def prompt(tok,q):
 msg=[{'role':'system','content':SYSTEM},{'role':'user','content':q}]
 try:return list(tok.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,enable_thinking=False))
 except TypeError:return list(tok.apply_chat_template(msg,tokenize=True,add_generation_prompt=True))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--samples',type=Path,required=True);ap.add_argument('--benchmark',type=Path,required=True);ap.add_argument('--cell',required=True);ap.add_argument('--model-path',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--locked',action='store_true');args=ap.parse_args()
 b=json.load(open(args.benchmark));split_names=('locked',) if args.locked else ('development','validation');rows={r['sample_id']:r for s in split_names for r in b['cells'][args.cell][s]['records']}; samples=[json.loads(x) for x in args.samples.open() if x.strip()]
 tok=AutoTokenizer.from_pretrained(args.model_path,local_files_only=True,trust_remote_code=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left';model=AutoModelForCausalLM.from_pretrained(args.model_path,local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,device_map='cuda').eval();layer=model.config.num_hidden_layers//2;records=[]
 for i,item in enumerate(samples,1):
  row=rows[item['sample_id']];p=prompt(tok,row['question']);seq=[p+tok.encode(a,add_special_tokens=False) for a in item['samples']];mx=max(map(len,seq));x=torch.full((len(seq),mx),tok.pad_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(x)
  for j,s in enumerate(seq):x[j,mx-len(s):]=torch.tensor(s,device='cuda');mask[j,mx-len(s):]=1
  with torch.inference_mode():h=model(x,attention_mask=mask,output_hidden_states=True,use_cache=False).hidden_states[layer]
  z=h[:,-1].float().T;zc=z-z.mean(dim=0,keepdim=True);sigma=zc.T@zc+0.001*torch.eye(len(seq),device='cuda');eig=torch.linalg.eigvalsh(sigma).clamp_min(1e-12);score=float(eig.log().mean())
  records.append({'sample_id':item['sample_id'],'split':item['split'],'label':item['label'],'risk_score':score,'eigenvalues':eig.cpu().tolist()})
  if i%50==0 or i==len(samples):print(f'{i}/{len(samples)}',flush=True)
 args.output.write_text(json.dumps({'schema_version':'one-confidence-inside-eigenscore-v1','cell':args.cell,'implementation_status':'approximate','paper_formula':'middle-layer last-token KxK covariance logdet, alpha=0.001','differences':'K=4 equal budget instead of K=10; feature clipping disabled','layer_index':layer,'records':records},indent=2)+'\n')
if __name__=='__main__':main()
