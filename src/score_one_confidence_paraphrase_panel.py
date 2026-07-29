#!/usr/bin/env python3
"""NLI semantic disagreement for the frozen four-template panel."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from score_one_confidence_sampling import NLI
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--panel',type=Path,required=True);ap.add_argument('--benchmark',type=Path,required=True);ap.add_argument('--cell',required=True);ap.add_argument('--nli-model',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--locked',action='store_true');args=ap.parse_args();b=json.load(open(args.benchmark));split_names=('locked',) if args.locked else ('development','validation');rows={r['sample_id']:r for s in split_names for r in b['cells'][args.cell][s]['records']};items=[json.loads(x) for x in args.panel.open() if x.strip()];nli=NLI(args.nli_model);records=[]
 for i,item in enumerate(items,1):
  row=rows[item['sample_id']];answers=item['panel_answers'];support=sum(nli.equivalent(row['question'],row['generated_answer'],a) for a in answers)/len(answers);records.append({'sample_id':item['sample_id'],'split':item['split'],'label':item['label'],'risk_score':1-support,'original_support':support,'panel_answers':answers})
  if i%50==0 or i==len(items):print(f'{i}/{len(items)}',flush=True)
 args.output.write_text(json.dumps({'schema_version':'one-confidence-fixed-paraphrase-nli-v1','cell':args.cell,'templates':items[0]['templates'],'nli_model':'microsoft/deberta-large-mnli','records':records},ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
