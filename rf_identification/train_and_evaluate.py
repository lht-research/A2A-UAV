from __future__ import annotations
import argparse, csv, hashlib, json, math, os, random, time
from collections import defaultdict
from datetime import timedelta
from itertools import count
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import models
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

METHODS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
    "efficientnet_v2_s": (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.DEFAULT),
    "convnext_tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.DEFAULT),
    "swin_t": (models.swin_t, models.Swin_T_Weights.DEFAULT),
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT),
}

def dump(p, x):
    p = Path(p); tmp = p.with_suffix(p.suffix + ".tmp"); tmp.write_text(json.dumps(x, indent=2, ensure_ascii=False)); tmp.replace(p)

def make_model(name, n):
    builder, weights = METHODS[name]; m = builder(weights=weights)
    if hasattr(m, "fc"): m.fc = nn.Linear(m.fc.in_features, n)
    elif name == "swin_t": m.head = nn.Linear(m.head.in_features, n)
    else:
        ix = next(i for i in range(len(m.classifier)-1, -1, -1) if isinstance(m.classifier[i], nn.Linear))
        m.classifier[ix] = nn.Linear(m.classifier[ix].in_features, n)
    return m

class OriginalPNG(Dataset):
    def __init__(self, rows, classes, cache):
        self.rows, self.classes, self.cache = rows, classes, Path(cache)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]; p = self.cache / r["split"] / (r["sample_id"] + ".png")
        with Image.open(p) as im:
            if im.size != (1333, 800) or im.mode != "RGB": raise RuntimeError(f"bad cache {p}")
            x = torch.from_numpy(np.array(im, copy=True)).permute(2, 0, 1).contiguous()
        return x, self.classes[r["label"]], i

class RecordingPKSampler(Sampler):
    def __init__(self, rows, batch, world, rank, seed, probs):
        if batch != 6: raise ValueError("recording PK requires local batch 6")
        self.rows, self.batch, self.world, self.rank, self.seed = rows, batch, world, rank, seed
        self.epoch = 0; self.groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for i, r in enumerate(rows): self.groups[r["label"]][r["canonical_recording_key"]][r["iq_key"]].append(i)
        self.labels = sorted(self.groups); self.recordings = {c: sorted(self.groups[c]) for c in self.labels}
        if any(len(v)<2 for v in self.recordings.values()):raise ValueError("Every class must have two training recordings")
        self.probs = np.asarray([float(probs.get(c, 1.0)) for c in self.labels]); self.probs /= self.probs.sum()
        self.steps = max(1, math.ceil(len(rows) / (world * batch)))
    def __len__(self): return self.steps * self.batch
    def set_epoch(self, e): self.epoch = e
    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch * 1009); plans = [[] for _ in range(self.world)]
        queues,cursors={},{}
        def take(key,items):
            if key not in queues or cursors[key]>=len(queues[key]):
                queues[key]=rng.permutation(len(items));cursors[key]=0
            value=items[int(queues[key][cursors[key]])];cursors[key]+=1
            return value
        for _ in range(self.steps):
            for rank in range(self.world):
                batch = []
                for c in rng.choice(self.labels, size=3, replace=False, p=self.probs):
                    chosen = rng.choice(self.recordings[c], size=2, replace=False)
                    for rec in chosen:
                        windows=sorted(self.groups[c][rec]);window=take(("window",c,rec),windows)
                        batch.append(take(("frame",c,rec,window),self.groups[c][rec][window]))
                plans[rank].extend(batch)
        return iter(plans[self.rank])

def load_index(index, cohort):
    data = json.loads(Path(index).read_text())
    rows = data["rows"]
    counts = {split: len(records) for split, records in rows.items()}
    if cohort == "paper":
        if counts != {"train": 62076, "val": 16472, "test": 27318}:
            raise ValueError("The paper cohort requires the archived experimental index")
    elif data["audit"].get("cohort") != "downloaded" or counts != data.get("counts"):
        raise ValueError("Run prepare_index.py to build the downloaded-cohort index")
    if len(data["class_to_idx"]) != 15 or data["audit"]["cap_per_recording"] != 0:
        raise ValueError("Expected 15 classes and complete recording groups")
    return data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=("paper", "downloaded"), default="paper")
    ap.add_argument("--index", required=True); ap.add_argument("--cache-root", required=True); ap.add_argument("--output-dir", required=True)
    ap.add_argument("--method", choices=METHODS, required=True); ap.add_argument("--epochs", type=int, default=0, help="0 trains until the validation stopping criterion; positive values are explicit limits")
    ap.add_argument("--freeze-epochs", type=int, default=1); ap.add_argument("--batch-size", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=6); ap.add_argument("--min-epochs", type=int, default=10)
    ap.add_argument("--min-delta",type=float,default=.001);ap.add_argument("--cosine-epochs",type=int,default=12)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--micro-batch-size", type=int, default=2)
    ap.add_argument("--hard-class-factor", type=float, default=2.0)
    ap.add_argument("--hard-classes", default="air2s,air3,air3s,flip,mini3pro,mini4pro_csq,mini4pro_xm,mini4pro_z,mini5pro,neo1,neo2")
    a = ap.parse_args()
    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1)); local = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local); torch.manual_seed(a.seed + rank); np.random.seed(a.seed + rank); random.seed(a.seed + rank); torch.set_num_threads(2)
    if world > 1: dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    data = load_index(a.index, a.cohort); rows = data["rows"]; classes = data["class_to_idx"]; names = list(classes); n = len(names)
    policy=json.loads((Path(a.cache_root)/"cache_policy.json").read_text())
    assert (policy["width"],policy["height"],policy["mode"]) == (1333,800,"RGB")
    recs={s:{r["canonical_recording_key"] for r in v} for s,v in rows.items()}
    assert not recs["train"]&recs["test"] and not recs["train"]&recs["val"] and not recs["val"]&recs["test"]
    hard = {x for x in a.hard_classes.split(",") if x}; probs = {c: a.hard_class_factor if c in hard else 1.0 for c in names}; ce_weights = np.ones(n, dtype=np.float32)
    ds = {s: OriginalPNG(v, classes, a.cache_root) for s, v in rows.items()}; sampler = RecordingPKSampler(rows["train"], a.batch_size, world, rank, a.seed, probs)
    loaders = {s: DataLoader(d, a.batch_size, sampler=sampler if s == "train" else list(range(rank, len(d), world)), num_workers=a.workers, pin_memory=True, persistent_workers=a.workers > 0, prefetch_factor=2) for s, d in ds.items()}
    dev = torch.device("cuda"); model = make_model(a.method, n).to(dev); ddp = DDP(model, device_ids=[local], broadcast_buffers=False, find_unused_parameters=True) if world > 1 else model
    if hasattr(model, "fc"): head = list(model.fc.parameters())
    elif hasattr(model, "head"): head = list(model.head.parameters())
    else: head = list(model.classifier.parameters())
    hid = {id(p) for p in head}; backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in hid]
    opt = torch.optim.AdamW([{"params": backbone, "lr": 1e-5}, {"params": head, "lr": 3e-4}], weight_decay=.05)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(ce_weights, device=dev), label_smoothing=.05)
    mean = torch.tensor([.485, .456, .406], device=dev).view(1,3,1,1); std = torch.tensor([.229, .224, .225], device=dev).view(1,3,1,1)
    def prep(x): return x.to(dev, non_blocking=True).float().div_(255).sub_(mean).div_(std)
    def evaluate(split, save=False):
        ddp.eval(); items=[]; loss_sum=seen=0
        with torch.inference_mode():
            for x,y,idx in loaders[split]:
                x=prep(x); yc=y.to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16): z=ddp(x)
                loss_sum += nn.functional.cross_entropy(z.float(),yc,reduction="sum").item(); seen += len(y)
                items.extend((int(i),int(yy),int(pp)) for i,yy,pp in zip(idx,y,z.argmax(1).cpu()))
        gathered=[None]*world if world>1 else None
        if world>1: dist.all_gather_object(gathered,items)
        else: gathered=[items]
        totals=torch.tensor([loss_sum,seen],dtype=torch.float64,device=dev)
        if world>1: dist.all_reduce(totals)
        result=None
        if rank==0:
            merged=sorted([x for q in gathered for x in q]); ys=[x[1] for x in merged]; ps=[x[2] for x in merged]
            assert [x[0] for x in merged]==list(range(len(rows[split]))),"Evaluation must cover each test sample exactly once"
            result={"accuracy":float(accuracy_score(ys,ps)),"balanced_accuracy":float(balanced_accuracy_score(ys,ps)),"macro_f1":float(f1_score(ys,ps,labels=list(range(n)),average="macro",zero_division=0)),"weighted_f1":float(f1_score(ys,ps,labels=list(range(n)),average="weighted",zero_division=0)),"loss":float(totals[0].item()/totals[1].item()),"n":len(ys)}
            if save:
                dump(Path(a.output_dir)/f"{split}_classification_report.json",classification_report(ys,ps,labels=list(range(n)),target_names=names,output_dict=True,zero_division=0));dump(Path(a.output_dir)/f"{split}_confusion_matrix.json",{"classes":names,"matrix":confusion_matrix(ys,ps,labels=list(range(n))).tolist()})
                with (Path(a.output_dir)/f"{split}_predictions.csv").open("w",newline="") as f:
                    writer=csv.writer(f);writer.writerow(["sample_id","recording","true_label","pred_label"])
                    for i,yy,pp in merged:writer.writerow([rows[split][i]["sample_id"],rows[split][i]["canonical_recording_key"],names[yy],names[pp]])
        payload=[result]
        if world>1: dist.broadcast_object_list(payload,src=0)
        return payload[0]
    out=Path(a.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    config={**vars(a),"method":a.method,"input":"1333x800 pixel-verified lossless cache","shape":[3,800,1333],"class_sampling_weights":probs,"class_loss_weights":dict(zip(names,ce_weights.tolist())),"protocol":"Recording-PK P=3 K=2; uniform recording then shuffled IQ-window/frame cycles; hard-class sampler factor 2; uniform CE with smoothing 0.05","counts":{s:len(v) for s,v in rows.items()},"data_audit":data["audit"],"world_size":world,"effective_batch":world*a.batch_size,"index_sha256":hashlib.sha256(Path(a.index).read_bytes()).hexdigest(),"script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"selection":"validation Macro-F1 only; no test evaluation until stopping","convergence_rule":{"minimum_epochs":a.min_epochs,"cosine_horizon":a.cosine_epochs,"patience":a.patience,"minimum_improvement":a.min_delta,"minimum_plateau_lr_reductions":2,"max_epochs":a.epochs or None},"torch":torch.__version__,"torchvision":__import__("torchvision").__version__}
    if rank==0:
        if not a.eval_only:
            if not a.smoke and (out/"metrics.jsonl").exists():raise RuntimeError("Refusing to overwrite existing training")
            dump(out/"run_config.json",config)
        else:dump(out/"eval_recovery_config.json",config)
    if world>1:dist.barrier()
    def train_batch(x,y):
        total_weight=criterion.weight[y].sum()
        total_loss=0.
        for start in range(0,len(y),a.micro_batch_size):
            end=min(start+a.micro_batch_size,len(y))
            for view_id in range(2):
                view=x[start:end].clone()
                if random.random()<.5:
                    span=max(1,int(view.shape[-1]*random.uniform(.005,.02)));st=random.randrange(max(1,view.shape[-1]-span));view[:,:,:,st:st+span]=0
                if random.random()<.5:
                    span=max(1,int(view.shape[-2]*random.uniform(.005,.02)));st=random.randrange(max(1,view.shape[-2]-span));view[:,:,st:st+span,:]=0
                sync=end==len(y) and view_id==1
                ctx=ddp.no_sync() if world>1 and not sync else __import__("contextlib").nullcontext()
                with ctx:
                    with torch.autocast("cuda",dtype=torch.bfloat16):z=ddp(view)
                    loss=.5*nn.functional.cross_entropy(z.float(),y[start:end],weight=criterion.weight,label_smoothing=.05,reduction="sum")/total_weight
                    loss.backward();total_loss+=float(loss.detach())
        nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
        return total_loss
    if a.smoke:
        ddp.train()
        for mod in model.modules():
            if isinstance(mod,nn.modules.batchnorm._BatchNorm):mod.eval()
        x,y,idx=next(iter(loaders["train"]))
        assert tuple(x.shape[1:])==(3,800,1333)
        for c in set(y.tolist()):
            assigned=[rows["train"][i]["canonical_recording_key"] for i in idx[y==c].tolist()]
            assert len(assigned)==2 and len(set(assigned))==2
        opt.zero_grad(set_to_none=True);loss=train_batch(prep(x),y.to(dev))
        if rank==0:
            temp=out/"checkpoint.tmp";torch.save({"model":model.state_dict(),"epoch":0},temp);temp.replace(out/"checkpoint.pt")
        if world>1:dist.barrier()
        ck=torch.load(out/"checkpoint.pt",map_location="cpu")
        assert ck["epoch"]==0 and math.isfinite(loss)
        if rank==0:dump(out/"smoke_metrics.json",{"smoke_passed":True,"shape":[6,3,800,1333],"loss":loss,"checkpoint_load":True,"world_size":world})
        if world>1:dist.barrier();dist.destroy_process_group()
        return
    best=-1; plateau_best=-1; bad=0; plateau_scale=1.;last_decay=0;decays=0;converged=False;epoch=0;best_epoch=0
    iterations=[] if a.eval_only else count(1) if a.epochs==0 else range(1,a.epochs+1)
    for epoch in iterations:
        sampler.set_epoch(epoch); ddp.train(); frozen=epoch<=a.freeze_epochs
        for p in backbone:p.requires_grad_(not frozen)
        if frozen:model.eval()
        for mod in model.modules():
            if isinstance(mod,nn.modules.batchnorm._BatchNorm):mod.eval()
        fraction=min(1.,max(0,epoch-a.freeze_epochs-1)/max(1,a.cosine_epochs-a.freeze_epochs-1))
        factor=.1+.9*.5*(1+math.cos(math.pi*fraction))
        opt.param_groups[0]['lr']=0. if frozen else 1e-5*factor*plateau_scale
        opt.param_groups[1]['lr']=3e-4*factor*plateau_scale
        started=time.time();loss_sum=0.;train_steps=0
        for step,(x,y,_) in enumerate(loaders["train"],1):
            x=prep(x);y=y.to(dev);opt.zero_grad(set_to_none=True);loss_sum+=train_batch(x,y);train_steps+=1
            if rank==0 and step%200==0:print(json.dumps({"phase":"train","epoch":epoch,"step":step,"steps":len(loaders["train"]),"elapsed":time.time()-started,"learning_rates":[g["lr"] for g in opt.param_groups]}),flush=True)
        val=evaluate("val")
        if not math.isfinite(val["macro_f1"]) or not math.isfinite(val["loss"]):raise RuntimeError("Non-finite validation metrics")
        rec={"epoch":epoch,"seconds":time.time()-started,"train_loss":loss_sum/train_steps,"learning_rates":[g["lr"] for g in opt.param_groups],"val":val}
        if rank==0:
            if val["macro_f1"]>best:
                best=val["macro_f1"];best_epoch=epoch;torch.save({"model":model.state_dict(),"epoch":epoch},out/"best.pt.tmp");(out/"best.pt.tmp").replace(out/"best.pt")
            if val["macro_f1"]>plateau_best+a.min_delta:
                plateau_best=val["macro_f1"];bad=0
            else:bad+=1
            if epoch>=max(a.min_epochs,a.cosine_epochs) and bad>=3 and epoch-last_decay>=3 and decays<2:
                plateau_scale*=.5;last_decay=epoch;decays+=1
            converged=epoch>=max(a.min_epochs,a.cosine_epochs) and decays>=2 and bad>=a.patience and epoch-last_decay>=3
            rec.update(plateau_epochs=bad,plateau_lr_reductions=decays,converged=converged,best_epoch=best_epoch,best_val_macro_f1=best)
            with (out/"metrics.jsonl").open("a") as f:f.write(json.dumps(rec)+"\n")
            dump(out/"convergence_state.json",{"epoch":epoch,"converged":converged,"best_epoch":best_epoch,"best_val_macro_f1":best,"plateau_epochs":bad,"lr_scale":plateau_scale,"lr_reductions":decays})
            print(json.dumps(rec),flush=True)
        control=[{"stop":converged,"scale":plateau_scale}]
        if world>1:dist.broadcast_object_list(control,src=0)
        plateau_scale=control[0]["scale"]
        if control[0]["stop"]:break
    if not a.eval_only and not control[0]["stop"]:
        if rank==0:dump(out/"termination.json",{"status":"not_converged","epoch":epoch,"reason":"explicit epoch limit","needs_extension":True})
        if world>1:dist.barrier();dist.destroy_process_group()
        return
    if world>1:dist.barrier()
    ck=torch.load(out/"best.pt",map_location="cpu");model.load_state_dict(ck["model"]);selection=evaluate("val",True);final=evaluate("test",True)
    if rank==0:
        dump(out/"selection_metrics.json",{"val":selection,"best_epoch":ck["epoch"]})
        dump(out/"final_metrics.json",{"test":final,"best_epoch":ck["epoch"],"epochs_ran":epoch,"converged":converged,"finished":time.time()});print(json.dumps(final),flush=True)
    if world>1:dist.barrier();dist.destroy_process_group()
if __name__=="__main__":main()
