#!/usr/bin/env python3
"""Gold-free local P(True) approximation for public dev/validation rows."""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM="Judge whether the proposed answer correctly answers the question. Respond with exactly True or False."

def base_ids(tok,q,a):
    msg=[{'role':'system','content':SYSTEM},{'role':'user','content':f'Question: {q}\nProposed answer: {a}\nIs the proposed answer correct?'}]
    try:return list(tok.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,enable_thinking=False))
    except TypeError:return list(tok.apply_chat_template(msg,tokenize=True,add_generation_prompt=True))

def seq_logp(model,base,candidate):
    ids=torch.tensor([base+candidate],device='cuda');
    with torch.inference_mode(): logits=model(ids,use_cache=False).logits[0]
    start=len(base)-1; lp=logits[start:start+len(candidate)].float().log_softmax(-1)
    return float(sum(lp[i,t] for i,t in enumerate(candidate)))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--benchmark',type=Path,required=True);ap.add_argument('--cell',required=True);ap.add_argument('--model-path',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--locked',action='store_true');args=ap.parse_args()
    b=json.load(open(args.benchmark));split_names=('locked',) if args.locked else ('development','validation');rows=[(s,r) for s in split_names for r in b['cells'][args.cell][s]['records']]
    tok=AutoTokenizer.from_pretrained(args.model_path,local_files_only=True,trust_remote_code=True);model=AutoModelForCausalLM.from_pretrained(args.model_path,local_files_only=True,trust_remote_code=True,torch_dtype=torch.bfloat16,device_map='cuda').eval()
    true=tok.encode('True',add_special_tokens=False);false=tok.encode('False',add_special_tokens=False);records=[]
    for i,(split,row) in enumerate(rows,1):
        base=base_ids(tok,row['question'],row['generated_answer']); lt=seq_logp(model,base,true);lf=seq_logp(model,base,false);m=max(lt,lf);pt=math.exp(lt-m)/(math.exp(lt-m)+math.exp(lf-m))
        records.append({'sample_id':row['sample_id'],'split':split,'label':row['class'],'p_true':pt,'risk_score':1-pt,'true_logp':lt,'false_logp':lf})
        if i%50==0 or i==len(rows):print(f'{i}/{len(rows)}',flush=True)
    args.output.write_text(json.dumps({'schema_version':'one-confidence-ptrue-local-v1','cell':args.cell,'implementation_status':'approximate','gold_visible':False,'prompt':SYSTEM,'records':records},indent=2)+'\n')
if __name__=='__main__':main()
