# train_fttransformer.py
import argparse, json, os, math
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import torch, torch.nn as nn, torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not available. Install with: pip install wandb")

def set_seed(seed=42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"]=str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_xy(data_root: Path, label_col: str | None):
    X = pd.read_parquet(data_root / "X_features.parquet")
    ydf = pd.read_csv(data_root / "y_label.csv")
    if label_col is None:
        label_col = "label" if "label" in ydf.columns else ydf.columns[0]
    y = ydf[label_col].astype(int).values
    if len(y) != len(X): raise ValueError("Row mismatch")
    return X.values.astype(np.float32), y, label_col, X.columns.tolist()

def load_splits_optional(data_root: Path):
    p = data_root / "splits.csv"
    if not p.exists(): return None
    sp = pd.read_csv(p)
    return np.where(sp["is_train"].values==1)[0], np.where(sp["is_val"].values==1)[0], np.where(sp["is_test"].values==1)[0]

def metrics(y, p, tag=""):
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    auc = roc_auc_score(y, p); ap = average_precision_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)
    ks = float(np.max(tpr - fpr)); gini = 2*auc - 1
    brier = np.mean((p - y)**2); ll = log_loss(y, p)
    return {"tag": tag, "auc": float(auc), "ap": float(ap), "ks": ks, "gini": gini, "brier": brier, "logloss": float(ll)}

# ----- FT-Transformer -----
class NumericalFeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.xavier_uniform_(self.weight)
    def forward(self, x): # (B,F)
        return x.unsqueeze(-1) * self.weight + self.bias # (B,F,D)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, attn_drop, ff_drop, prenorm=True):
        super().__init__()
        self.prenorm = prenorm
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=attn_drop, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(ff_drop),
                                nn.Linear(d_ff, d_model), nn.Dropout(ff_drop))
    def forward(self, x):
        if self.prenorm:
            z = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0]
            z = z + self.ff(self.ln2(z))
        else:
            z = self.ln1(x + self.attn(x, x, x, need_weights=False)[0])
            z = self.ln2(z + self.ff(z))
        return z

class FTTransformer(nn.Module):
    def __init__(self, n_features, d_model=96, n_layers=4, n_heads=8, ff_mult=2.0,
                 attn_drop=0.1, ff_drop=0.1, cls_token=True, feature_dropout=0.05):
        super().__init__()
        self.tokenizer = NumericalFeatureTokenizer(n_features, d_model)
        self.cls_token = cls_token
        self.feature_dropout = feature_dropout
        if cls_token:
            self.cls = nn.Parameter(torch.zeros(1,1,d_model))
            nn.init.trunc_normal_(self.cls, std=0.02)
        d_ff = int(d_model * ff_mult)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, attn_drop, ff_drop, prenorm=True)
                                     for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
    def forward(self, x): # (B,F)
        tok = self.tokenizer(x) # (B,F,D)
        if self.feature_dropout > 0 and self.training:
            mask = (torch.rand(tok.shape[:2], device=tok.device) < self.feature_dropout).unsqueeze(-1)
            tok = tok.masked_fill(mask, 0.0)
        z = torch.cat([self.cls.expand(tok.size(0),-1,-1), tok], dim=1) if self.cls_token else tok
        for blk in self.blocks: z = blk(z)
        out = self.ln(z[:,0,:]) if self.cls_token else self.ln(z.mean(dim=1))
        return self.head(out).squeeze(1)

def cosine_warmup(optimizer, step, total_steps, warmup_ratio=0.06, base_lr=1e-3, min_lr=1e-6):
    w = int(total_steps * warmup_ratio)
    if step < w: lr = base_lr * (step / max(1,w))
    else:
        t = (step - w) / max(1, total_steps - w)
        lr = min_lr + 0.5*(base_lr - min_lr)*(1 + math.cos(math.pi * t))
    for pg in optimizer.param_groups: pg["lr"] = lr
    return lr

def eval_preds(model, X_t, y, device):
    model.eval()
    with torch.no_grad(), autocast(enabled=(device.type=="cuda")):
        p = torch.sigmoid(model(X_t)).float().cpu().numpy()
    return p

def train_one(model, Xtr, ytr, Xva, yva, device, args, run=None, fold_tag="tvt"):
    scaler = StandardScaler().fit(Xtr)
    Xtr = np.nan_to_num(scaler.transform(Xtr).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    Xva = np.nan_to_num(scaler.transform(Xva).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.tensor(yva, dtype=torch.float32, device=device)

    pos = float(ytr_t.sum().item()); neg = float(len(ytr_t)-ytr_t.sum().item())
    pos_weight = torch.tensor([(neg / max(pos, 1.0))], device=device, dtype=torch.float32)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler_amp = GradScaler(enabled=(device.type=="cuda"))

    steps_per_epoch = int(math.ceil(len(Xtr_t) / args.batch))
    total_steps = steps_per_epoch * args.epochs
    best_auc, best_state, best_epoch, no_improve = -1, None, 0, 0

    for epoch in range(1, args.epochs+1):
        model.train(); perm = torch.randperm(len(Xtr_t), device=device)
        running_loss = 0.0
        for s in range(0, len(Xtr_t), args.batch):
            idx = perm[s:s+args.batch]; xb, yb = Xtr_t[idx], ytr_t[idx]
            opt.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type=="cuda")):
                logits = model(xb); loss = crit(logits, yb)
            scaler_amp.scale(loss).backward()
            if args.grad_clip and args.grad_clip > 0:
                scaler_amp.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler_amp.step(opt); scaler_amp.update()
            step = (epoch-1)*steps_per_epoch + (s//args.batch) + 1
            cosine_warmup(opt, step, total_steps, warmup_ratio=args.warmup_ratio, base_lr=args.lr, min_lr=args.min_lr)
            running_loss += loss.item() * len(idx)

        p_tr = eval_preds(model, Xtr_t, ytr, device)
        p_va = eval_preds(model, Xva_t, yva, device)
        mtr = metrics(ytr, p_tr, f"train_{fold_tag}")
        mva = metrics(yva, p_va, f"val_{fold_tag}")
        print(f"[{fold_tag}][{epoch:03d}] train_auc={mtr['auc']:.4f} val_auc={mva['auc']:.4f} "
              f"train_ap={mtr['ap']:.4f} val_ap={mva['ap']:.4f} lr={opt.param_groups[0]['lr']:.2e}")

        if WANDB_AVAILABLE and run is not None:
            wandb.log({
                f"{fold_tag}/epoch": epoch, f"{fold_tag}/lr": opt.param_groups[0]['lr'],
                f"{fold_tag}/train_loss": running_loss / len(Xtr_t),
                f"{fold_tag}/train_auc": mtr["auc"], f"{fold_tag}/train_ap": mtr["ap"], f"{fold_tag}/train_ks": mtr["ks"], f"{fold_tag}/train_logloss": mtr["logloss"],
                f"{fold_tag}/val_auc": mva["auc"], f"{fold_tag}/val_ap": mva["ap"], f"{fold_tag}/val_ks": mva["ks"], f"{fold_tag}/val_logloss": mva["logloss"],
            })

        if mva["auc"] > best_auc + 1e-5:
            best_auc, best_epoch = mva["auc"], epoch
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= args.patience: break

    if best_state is not None: model.load_state_dict(best_state)
    return model, scaler, best_auc, best_epoch

def run_tvt(X, y, tr, va, te, device, args, run=None, out_dir: Path = Path("model_outputs/fttr_tvt")):
    model = FTTransformer(n_features=X.shape[1], d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
                          ff_mult=args.ff_mult, attn_drop=args.attn_dropout, ff_drop=args.dropout,
                          cls_token=True, feature_dropout=args.feature_dropout).to(device)
    model, scaler, best_auc, best_epoch = train_one(model, X[tr], y[tr], X[va], y[va], device, args, run, "tvt")
    Xva_t = torch.tensor(np.nan_to_num(scaler.transform(X[va]).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0), device=device)
    Xte_t = torch.tensor(np.nan_to_num(scaler.transform(X[te]).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0), device=device)
    with torch.no_grad():
        pva = torch.sigmoid(model(Xva_t)).float().cpu().numpy()
        pte = torch.sigmoid(model(Xte_t)).float().cpu().numpy()
    sm_val = metrics(y[va], pva, "val_raw"); sm_tst = metrics(y[te], pte, "test_raw")
    lr_cal = LogisticRegression(max_iter=1000).fit(pva.reshape(-1,1), y[va])
    pte_cal = lr_cal.predict_proba(pte.reshape(-1,1))[:,1]
    sm_tst_cal = metrics(y[te], pte_cal, "test_cal")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir/"summary.json").write_text(json.dumps({"val": sm_val, "test_raw": sm_tst, "test_cal": sm_tst_cal}, indent=2))
    pd.DataFrame({"p_val": pva, "y_val": y[va]}).to_csv(out_dir/"val_preds.csv", index=False)
    pd.DataFrame({"p_test_raw": pte, "p_test_cal": pte_cal, "y_test": y[te]}).to_csv(out_dir/"test_preds.csv", index=False)
    torch.save(model.state_dict(), out_dir/"fttr_state.pt")
    (out_dir/"scaler.json").write_text(json.dumps({"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}))
    if WANDB_AVAILABLE and run is not None:
        wandb.log({
            "val/auc": sm_val["auc"], "val/ap": sm_val["ap"], "val/ks": sm_val["ks"], "val/logloss": sm_val["logloss"],
            "test_raw/auc": sm_tst["auc"], "test_raw/ap": sm_tst["ap"], "test_raw/ks": sm_tst["ks"], "test_raw/logloss": sm_tst["logloss"],
            "test_cal/auc": sm_tst_cal["auc"], "test_cal/ap": sm_tst_cal["ap"], "test_cal/ks": sm_tst_cal["ks"], "test_cal/logloss": sm_tst_cal["logloss"],
            "best_epoch": best_epoch, "best_val_auc": best_auc
        })
    return {"val": sm_val, "test_raw": sm_tst, "test_cal": sm_tst_cal}

def run_kfold(X, y, test_idx, device, args, run=None, out_dir: Path = Path("model_outputs/fttr_kfold")):
    n = len(y); oof = np.zeros(n, dtype=np.float32)
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    train_pool = np.setdiff1d(np.arange(n), test_idx) if test_idx is not None and len(test_idx)>0 else np.arange(n)
    fold = 0; records = []
    for tr_rel, va_rel in skf.split(train_pool, y[train_pool]):
        fold += 1
        tr_idx = train_pool[tr_rel]; va_idx = train_pool[va_rel]
        model = FTTransformer(n_features=X.shape[1], d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
                              ff_mult=args.ff_mult, attn_drop=args.attn_dropout, ff_drop=args.dropout,
                              cls_token=True, feature_dropout=args.feature_dropout).to(device)
        model, scaler, best_auc, best_epoch = train_one(model, X[tr_idx], y[tr_idx], X[va_idx], y[va_idx], device, args, run, f"fold{fold}")
        Xva_t = torch.tensor(np.nan_to_num(scaler.transform(X[va_idx]).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0), device=device)
        with torch.no_grad():
            pva = torch.sigmoid(model(Xva_t)).float().cpu().numpy()
        oof[va_idx] = pva
        rec = metrics(y[va_idx], pva, f"fold{fold}_val")
        rec["fold"] = fold; rec["best_epoch"] = best_epoch; rec["best_val_auc"] = best_auc
        records.append(rec)
        if WANDB_AVAILABLE and run is not None:
            wandb.log({f"kfold/fold{fold}_val_auc": rec["auc"], f"kfold/fold{fold}_val_ap": rec["ap"], "kfold/fold": fold})

    out_dir.mkdir(parents=True, exist_ok=True)
    sm_oof = metrics(y[train_pool], oof[train_pool], "oof")
    pd.DataFrame({"oof": oof, "y": y}).to_csv(out_dir/"oof_preds.csv", index=False)
    (out_dir/"oof_summary.json").write_text(json.dumps(sm_oof, indent=2))
    pd.DataFrame(records).to_csv(out_dir/"per_fold_summary.csv", index=False)
    if WANDB_AVAILABLE and run is not None:
        wandb.log({"oof/auc": sm_oof["auc"], "oof/ap": sm_oof["ap"], "oof/ks": sm_oof["ks"], "oof/logloss": sm_oof["logloss"]})

    if test_idx is not None and len(test_idx)>0:
        model = FTTransformer(n_features=X.shape[1], d_model=args.d_model, n_layers=args.layers, n_heads=args.heads,
                              ff_mult=args.ff_mult, attn_drop=args.attn_dropout, ff_drop=args.dropout,
                              cls_token=True, feature_dropout=args.feature_dropout).to(device)
        scaler = StandardScaler().fit(X[train_pool])
        Xtr = np.nan_to_num(scaler.transform(X[train_pool]).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        Xte = np.nan_to_num(scaler.transform(X[test_idx]).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device); ytr_t = torch.tensor(y[train_pool], dtype=torch.float32, device=device)
        Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
        pos = float(ytr_t.sum().item()); neg = float(len(ytr_t)-ytr_t.sum().item())
        pos_weight = torch.tensor([(neg / max(pos, 1.0))], device=device, dtype=torch.float32)
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler_amp = GradScaler(enabled=(device.type=="cuda"))
        steps_per_epoch = int(math.ceil(len(Xtr_t) / args.batch))
        total_steps = steps_per_epoch * max(5, args.epochs//3)
        for epoch in range(1, max(5, args.epochs//3)+1):
            model.train(); perm = torch.randperm(len(Xtr_t), device=device)
            for s in range(0, len(Xtr_t), args.batch):
                idx = perm[s:s+args.batch]; xb, yb = Xtr_t[idx], ytr_t[idx]
                opt.zero_grad(set_to_none=True)
                with autocast(enabled=(device.type=="cuda")):
                    logits = model(xb); loss = crit(logits, yb)
                scaler_amp.scale(loss).backward(); scaler_amp.step(opt); scaler_amp.update()
                step = (epoch-1)*steps_per_epoch + (s//args.batch) + 1
                cosine_warmup(opt, step, total_steps, warmup_ratio=args.warmup_ratio, base_lr=args.lr, min_lr=args.min_lr)
        with torch.no_grad():
            pte = torch.sigmoid(model(Xte_t)).float().cpu().numpy()
        sm_tst = metrics(y[test_idx], pte, "test_raw")
        (out_dir/"test_summary.json").write_text(json.dumps(sm_tst, indent=2))
        if WANDB_AVAILABLE and run is not None:
            wandb.log({"test/auc": sm_tst["auc"], "test/ap": sm_tst["ap"], "test/ks": sm_tst["ks"], "test/logloss": sm_tst["logloss"]})
        return {"oof": sm_oof, "test_raw": sm_tst}
    return {"oof": sm_oof}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--label-column", default="label_clustered")
    ap.add_argument("--out-dir", default="model_outputs/fttr")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--mode", choices=["tvt","kfold"], default="tvt")
    ap.add_argument("--n-splits", type=int, default=5)

    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d-model", type=int, default=96)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ff-mult", type=float, default=2.0)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--attn-dropout", type=float, default=0.10)
    ap.add_argument("--feature-dropout", type=float, default=0.05)

    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y, label, feats = load_xy(Path(args.data_root), args.label_column)
    splits = load_splits_optional(Path(args.data_root))

    run = None
    if WANDB_AVAILABLE:
        run = wandb.init(
            entity="ved100-carnegie-mellon-university",
            project="Risk_Score",
            name=f"fttr-{args.mode}-{args.label_column}",
            config={**vars(args), "n_features": X.shape[1], "n_samples": len(X), "prevalence": float(y.mean())},
            tags=["fttransformer", args.mode, args.label_column]
        )
        wandb.log({"dataset/n_samples": len(X), "dataset/n_features": X.shape[1],
                   "dataset/n_positives": int(y.sum()), "dataset/prevalence": float(y.mean())})

    if args.mode == "tvt":
        if splits is None:
            raise FileNotFoundError(f"{Path(args.data_root)/'splits.csv'} not found. Run make_splits.py first.")
        tr, va, te = splits
        print(f"[FTTR TVT] X={X.shape} prev={y.mean():.2%} | train={len(tr)} val={len(va)} test={len(te)}")
        res = run_tvt(X, y, tr, va, te, device, args, run, out_dir)
        print(json.dumps(res, indent=2))
    else:
        test_idx = None
        if splits is not None:
            test_idx = np.where(pd.read_csv(Path(args.data_root)/"splits.csv")["is_test"].values==1)[0]
            print(f"[FTTR KFold] Holding out test={len(test_idx)} rows. CV over the rest.")
        else:
            print("[FTTR KFold] No splits.csv found. Running CV over all rows.")
        res = run_kfold(X, y, test_idx, device, args, run, out_dir)
        print(json.dumps(res, indent=2))

    if WANDB_AVAILABLE and run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()