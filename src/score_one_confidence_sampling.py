#!/usr/bin/env python3
"""Score K-sample self-consistency and NLI Semantic Entropy."""
from __future__ import annotations
import argparse, json, math, re, string
from collections import Counter
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def norm(x):
    x=x.lower(); x=''.join(ch if ch not in string.punctuation else ' ' for ch in x)
    return ' '.join(x.split())

class NLI:
    def __init__(self,path):
        self.tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
        # This legacy DeBERTa checkpoint mixes FP32 relative-position tensors
        # into attention and fails under BF16 on current Transformers.
        self.model=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True,torch_dtype=torch.float32).cuda().eval()
        self.entail=next(int(i) for i,n in self.model.config.id2label.items() if 'entail' in str(n).lower())
        self.cache={}
    def entails(self,q,a,b):
        key=(q,a,b)
        if key not in self.cache:
            batch=self.tok(f'Question: {q} Answer: {a}',b,return_tensors='pt',truncation=True).to('cuda')
            with torch.inference_mode(): label=int(self.model(**batch).logits.argmax(-1))
            self.cache[key]=label==self.entail
        return self.cache[key]
    def equivalent(self,q,a,b):
        if norm(a)==norm(b): return True
        return self.entails(q,a,b) and self.entails(q,b,a)

def clusters(nli,q,answers):
    reps=[]; assignment=[]
    for answer in answers:
        found=None
        for i,rep in enumerate(reps):
            if nli.equivalent(q,answer,rep): found=i; break
        if found is None: found=len(reps); reps.append(answer)
        assignment.append(found)
    return assignment,reps

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--samples',type=Path,required=True); ap.add_argument('--benchmark',type=Path,required=True); ap.add_argument('--cell',required=True); ap.add_argument('--nli-model',required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--locked',action='store_true'); args=ap.parse_args()
    b=json.load(open(args.benchmark)); split_names=('locked',) if args.locked else ('development','validation'); rows={r['sample_id']:r for s in split_names for r in b['cells'][args.cell][s]['records']}
    sampled=[json.loads(x) for x in args.samples.open() if x.strip()]; nli=NLI(args.nli_model); records=[]
    for i,item in enumerate(sampled,1):
        row=rows[item['sample_id']]; answers=item['samples']; ids,reps=clusters(nli,row['question'],answers); counts=Counter(ids); k=len(answers)
        sem_entropy=-sum((n/k)*math.log(n/k) for n in counts.values())
        original_support=sum(nli.equivalent(row['question'],row['generated_answer'],a) for a in answers)/k
        records.append({'sample_id':item['sample_id'],'split':item['split'],'label':item['label'],'self_consistency_risk':1-original_support,'semantic_entropy_risk':sem_entropy,'cluster_ids':ids,'cluster_representatives':reps,'original_support':original_support})
        if i%50==0 or i==len(sampled): print(f'{i}/{len(sampled)}',flush=True)
    args.output.write_text(json.dumps({'schema_version':'one-confidence-nli-sampling-scores-v1','cell':args.cell,'K':len(sampled[0]['samples']),'nli_model':'microsoft/deberta-large-mnli','clustering':'greedy bidirectional entailment','records':records},ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__': main()
