#!/usr/bin/env python3
"""Evaluate the executable, phenotype-free minimum MOSAIC gate."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from evaluate_one_confidence_five_model import choose_threshold

CELLS=("mistral7|TriviaQA","qwen8|TriviaQA","llama31|TriviaQA","qwen14|PopQA","nemo12|PopQA")
def load_records(path,field):
 x=json.load(open(path));return {r['sample_id']:float(r[field]) for r in x['records']}
def est():return make_pipeline(StandardScaler(),LogisticRegression(C=.01,class_weight='balanced',max_iter=3000,random_state=2026072906))
def oof(x,y):
 out=np.empty(len(y));cv=StratifiedKFold(5,shuffle=True,random_state=2026072906)
 for tr,ho in cv.split(x,y):out[ho]=est().fit(x[tr],y[tr]).predict_proba(x[ho])[:,1]
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args();root=args.root;out={'schema_version':'one-confidence-minimal-mosaic-v1','implementation_status':'minimal_gate_no_phenotype','locked_accessed':False,'cells':{}}
 for cell in CELLS:
  slug=cell.replace('|','-').replace('TriviaQA','triviaqa').replace('PopQA','popqa');hid=json.load(open(root/f'{slug}-public-eval.json'));panel=json.load(open(root/f'{slug}-paraphrase-scores.json'));sample=json.load(open(root/f'{slug}-sampling-scores-k4.json'))
  ids=[r['sample_id'] for r in hid['methods']['prompt_end_hidden_linear_probe']['scores']]; lookup={m:{z['sample_id']:z['score'] for z in r['scores']} for m,r in hid['methods'].items() if 'hidden_linear_probe' in m}; rows={z['sample_id']:z for z in hid['methods']['prompt_end_hidden_linear_probe']['scores']};beh=load_records(root/f'{slug}-paraphrase-scores.json','risk_score');pers=load_records(root/f'{slug}-sampling-scores-k4.json','semantic_entropy_risk');
  # All sources must have exactly the same public IDs.
  if set(ids)!=set(beh) or set(ids)!=set(pers):raise ValueError(f'alignment failure {cell}')
  y=np.asarray([1 if rows[i]['label']=='HCE' else 0 for i in ids]);dev=np.asarray([rows[i]['split']=='development' for i in ids]);val=~dev
  internal=np.column_stack([[lookup[k][sample_id] for sample_id in ids] for k in ('prompt_end_hidden_linear_probe','first_answer_hidden_linear_probe','mean_answer_hidden_linear_probe')]);behavior=np.asarray([beh[sample_id] for sample_id in ids]);persistence=np.asarray([pers[sample_id] for sample_id in ids]);
  # Individual pathways: behavior/persistence are fixed risk signals; internal is learned on dev.
  internal_oof=oof(internal[dev],y[dev]);internal_model=est().fit(internal[dev],y[dev]);internal_scores=np.full(len(ids),np.nan);internal_scores[dev]=internal_oof;internal_scores[val]=internal_model.predict_proba(internal[val])[:,1]
  def op(scores):return choose_threshold(y[val],scores[val])
  methods={'behavioral_component':behavior,'persistence_component':persistence,'internal_component':internal_scores}
  # Equal average uses development z-normalization and no fitted classifier.
  z=[]
  for a in (behavior,persistence,internal_scores):z.append((a-np.nanmean(a[dev]))/(np.nanstd(a[dev])+1e-8))
  methods['behavioral_plus_internal_average']=(z[0]+z[2])/2
  flat=np.column_stack([behavior,persistence,internal]);flat_model=est().fit(flat[dev],y[dev]);flat_scores=np.full(len(ids),np.nan);flat_scores[dev]=oof(flat[dev],y[dev]);flat_scores[val]=flat_model.predict_proba(flat[val])[:,1];methods['flat_feature_concatenation']=flat_scores
  # Gate receives only OOF expert outputs on dev; no in-sample pathway scores.
  b_exp=est().fit(behavior[dev,None],y[dev]);p_exp=est().fit(persistence[dev,None],y[dev]);i_exp=est().fit(internal[dev],y[dev]);expert_dev=np.column_stack([oof(behavior[dev,None],y[dev]),oof(persistence[dev,None],y[dev]),internal_oof]);expert_val=np.column_stack([b_exp.predict_proba(behavior[val,None])[:,1],p_exp.predict_proba(persistence[val,None])[:,1],i_exp.predict_proba(internal[val])[:,1]]);gate=est().fit(expert_dev,y[dev]);gate_scores=np.full(len(ids),np.nan);gate_scores[dev]=gate.predict_proba(expert_dev)[:,1];gate_scores[val]=gate.predict_proba(expert_val)[:,1];methods['minimal_gated_mosaic']=gate_scores
  cell_out={}
  for name,scores in methods.items():cell_out[name]={'validation':op(scores),'scores':[{'sample_id':i,'split':rows[i]['split'],'label':rows[i]['label'],'score':float(scores[j])} for j,i in enumerate(ids)]}
  out['cells'][cell]=cell_out
 args.output.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
