#!/usr/bin/env python3
"""Pair-preserving bootstrap intervals for public validation scores."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from evaluate_one_confidence_five_model import metrics
CELLS=('mistral7|TriviaQA','qwen8|TriviaQA','llama31|TriviaQA','qwen14|PopQA','nemo12|PopQA')
def score_files(root,cell):
 slug=cell.replace('|','-').replace('TriviaQA','triviaqa').replace('PopQA','popqa'); out={}
 def add(path,method,field='score'):
  p=root/path
  if not p.exists():return
  x=json.load(open(p));
  if method=='sequence_confidence': rec=x['methods'][method]['cells'][cell]['scores']
  elif method=='los_net': rec=x['records']
  else: rec=x['methods'][method]['scores']
  out[method]={z['sample_id']:float(z['score']) for z in rec}
 add('sequence_confidence_public.json','sequence_confidence')
 for m in ('predictive_token_entropy','prompt_end_hidden_linear_probe','first_answer_hidden_linear_probe','mean_answer_hidden_linear_probe'):add(f'{slug}-public-eval.json',m)
 for f,m in [(f'{slug}-ptrue-eval.json','p_true'),(f'{slug}-sampling-eval-k4.json','self_consistency'),(f'{slug}-sampling-eval-k4.json','semantic_entropy'),(f'{slug}-paraphrase-eval.json','paraphrase_consistency'),(f'{slug}-inside-eval-k4.json','inside_eigenscore'),(f'{slug}-los-eval.json','los_net')]:add(f,m)
 mosaic=json.load(open(root/'minimal_mosaic_public.json'))['cells'][cell]
 for m,v in mosaic.items():out[m]={z['sample_id']:float(z['score']) for z in v['scores']}
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--benchmark',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--draws',type=int,default=2000);args=ap.parse_args();b=json.load(open(args.benchmark));rng=np.random.default_rng(2026072907);result={}
 for cell in CELLS:
  pairs=b['cells'][cell]['validation']['pairs'];scores=score_files(args.root,cell);result[cell]={}
  for method,smap in scores.items():
   # Threshold is the one selected on the complete validation set and is held fixed.
   threshold=next(z['threshold'] for z in json.load(open(args.root/'validation_matrix.json'))['cells'] if False) if False else None
   # Read threshold from the corresponding source file.
   slug=cell.replace('|','-').replace('TriviaQA','triviaqa').replace('PopQA','popqa')
   if method in ('behavioral_component','persistence_component','internal_component','behavioral_plus_internal_average','flat_feature_concatenation','minimal_gated_mosaic'): src=json.load(open(args.root/'minimal_mosaic_public.json'))['cells'][cell][method]
   elif method=='sequence_confidence':src=json.load(open(args.root/'sequence_confidence_public.json'))['methods'][method]['cells'][cell]
   else:
    files={'predictive_token_entropy':f'{slug}-public-eval.json','prompt_end_hidden_linear_probe':f'{slug}-public-eval.json','first_answer_hidden_linear_probe':f'{slug}-public-eval.json','mean_answer_hidden_linear_probe':f'{slug}-public-eval.json','p_true':f'{slug}-ptrue-eval.json','self_consistency':f'{slug}-sampling-eval-k4.json','semantic_entropy':f'{slug}-sampling-eval-k4.json','paraphrase_consistency':f'{slug}-paraphrase-eval.json','inside_eigenscore':f'{slug}-inside-eval-k4.json','los_net':f'{slug}-los-eval.json'};src=json.load(open(args.root/files[method]))['methods'][method] if method in ('predictive_token_entropy','prompt_end_hidden_linear_probe','first_answer_hidden_linear_probe','mean_answer_hidden_linear_probe','p_true','self_consistency','semantic_entropy','paraphrase_consistency','inside_eigenscore') else json.load(open(args.root/files[method]))
   threshold=src['validation']['threshold'];pair_values=[(smap[p['hce_id']],smap[p['hcc_id']]) for p in pairs]
   vals=[]
   for _ in range(args.draws):
    ix=rng.integers(0,len(pair_values),len(pair_values));arr=np.asarray([v for i in ix for v in (pair_values[i][0],pair_values[i][1])]);lab=np.asarray([1,0]*len(ix));q=metrics(lab,arr,threshold);vals.append([q['accuracy'],q['precision'],q['recall'],q['fpr'],q['f1']])
   a=np.asarray(vals);result[cell][method]={'threshold':threshold,'bootstrap_draws':args.draws,'accuracy_ci95':np.quantile(a[:,0],[.025,.975]).tolist(),'precision_ci95':np.quantile(a[:,1],[.025,.975]).tolist(),'recall_ci95':np.quantile(a[:,2],[.025,.975]).tolist(),'fpr_ci95':np.quantile(a[:,3],[.025,.975]).tolist(),'f1_ci95':np.quantile(a[:,4],[.025,.975]).tolist()}
 args.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
