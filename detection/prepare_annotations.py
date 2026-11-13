import argparse,json
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--dataset-root',required=True,type=Path);ap.add_argument('--rgb-out',required=True,type=Path);ap.add_argument('--panorama-out',required=True,type=Path);a=ap.parse_args(); ds=a.dataset_root.resolve(); a.rgb_out.mkdir(parents=True,exist_ok=True); a.panorama_out.mkdir(parents=True,exist_ok=True); summary={}
 for split in ('train','val','test'):
  source=json.loads((ds/'annotations'/f'instances_{split}.json').read_text()); summary[split]={}
  for modality,out,suffix in (('rgb',a.rgb_out,'rgb_single'),('panorama',a.panorama_out,'panorama_single')):
   imgs=[]; remap={}
   for item in source['images']:
    if item.get('modality')!=modality: continue
    q=dict(item); q['id']=len(imgs)+1; q['file_name']=str(ds/q['file_name']); remap[item['id']]=q['id']; imgs.append(q)
   anns=[]
   for item in source['annotations']:
    if item['image_id'] not in remap: continue
    q=dict(item); q['id']=len(anns)+1; q['image_id']=remap[item['image_id']]; q['category_id']=1; anns.append(q)
   payload={k:v for k,v in source.items() if k not in ('images','annotations','categories')}; payload.update({'images':imgs,'annotations':anns,'categories':[{'id':1,'name':'uav','supercategory':'uav'}]}); p=out/f'instances_{split}_{suffix}.json'; p.write_text(json.dumps(payload,separators=(',',':'),ensure_ascii=False)); summary[split][modality]={'images':len(imgs),'annotations':len(anns),'path':str(p)}
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
