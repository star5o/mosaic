#!/usr/bin/env python3
"""Extract the LOS-Net reproduction inputs (top-k output signatures)."""
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
 ap=argparse.ArgumentParser();ap.add_argument('--benchmark',type=Path,required=True);ap.add_argument('--cell',required=True);ap.add_argument('--model-path',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--topk',type=int,default=1000);ap.add_argument('--batch-size',type=int,default=4);ap.add_argument('--limit',type=int);ap.add_argument('--locked',action='store_true');args=ap.parse_args()
 b=json.load(open(args.benchmark));split_names=('locked',) if args.locked else ('development','validation');rows=[(s,r) for s in split_names for r in b['cells'][args.cell][s]['records']];rows=rows[:args.limit] if args.limit else rows;tok=AutoTokenizer.from_pretrained(args.model_path,local_files_only=True,trust_remote_code=True);tok.pad_token=tok.pad_token or tok.eos_token;tok.padding_side='left';model=AutoModelForCausalLM.from_pretrained(args.model_path,local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,device_map='cuda').eval();seqs=[];actual_seqs=[];rank_seqs=[];labels=[];splits=[];ids=[]
 for off in range(0,len(rows),args.batch_size):
  batch=rows[off:off+args.batch_size]; sequences=[(prompt(tok,r['question'])+tok.encode(r['generated_answer'],add_special_tokens=False),s,r) for s,r in batch];mx=max(len(z[0]) for z in sequences);x=torch.full((len(sequences),mx),tok.pad_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(x);starts=[]
  for i,(z,_,_) in enumerate(sequences):starts.append(mx-len(z));x[i,starts[-1]:]=torch.tensor(z,device='cuda');mask[i,starts[-1]:]=1
  with torch.inference_mode():logits=model(x,attention_mask=mask,use_cache=False).logits.float()
  for i,(z,split,row) in enumerate(sequences):
   p_len=len(prompt(tok,row['question']));aids=tok.encode(row['generated_answer'],add_special_tokens=False);pos=starts[i]+p_len-1; values=[]; actual=[]; ranks=[]
   for j,tokid in enumerate(aids):
    l=logits[i,pos+j]; probs=l.softmax(-1); top_probs=probs.topk(args.topk).values
    mu=probs.mean();std=probs.std().clamp_min(1e-8); values.append(((top_probs-mu)/std).half().cpu()); chosen=probs[int(tokid)];actual.append(float((chosen-mu)/std));ranks.append(int((l>l[int(tokid)]).sum()))
   seqs.append(torch.stack(values));actual_seqs.append(torch.tensor(actual));rank_seqs.append(torch.tensor(ranks));labels.append(row['class']);splits.append(split);ids.append(row['sample_id'])
  print(f'{min(off+len(batch),len(rows))}/{len(rows)}',flush=True)
 maxlen=max(x.shape[0] for x in seqs);vocab_size=len(tok);tds=torch.zeros((len(seqs),maxlen,args.topk),dtype=torch.float16);atp=torch.zeros((len(seqs),maxlen));rank=torch.full((len(seqs),maxlen),vocab_size,dtype=torch.int32)
 for i,z in enumerate(seqs):
  tds[i,:z.shape[0]]=z;atp[i,:z.shape[0]]=actual_seqs[i];rank[i,:z.shape[0]]=rank_seqs[i]
 # Label metadata stays separate from model-readable LOS tensors.
 torch.save({'schema_version':'one-confidence-los-input-v1','cell':args.cell,'topk':args.topk,'vocab_size':vocab_size,'row_ids':ids,'splits':splits,'labels':labels,'sorted_tds_normalized':tds,'normalized_atp':atp,'atp_r':rank,'implementation_status':'third_party_reproduction_input','normalization_note':'full-vocabulary probabilities standardized by full-vocabulary mean/std, then top-k retained'},args.output)
if __name__=='__main__':main()
