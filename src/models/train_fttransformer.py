import os, math, json, time, numpy as np, pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not available. Install with: pip install wandb")

from ..utils.io_utils import load_xy, dump_json
from ..utils.metrics import summarize_all, decile_table

def set_seed(seed: int = 42):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ---------- Feature Tokenizer (numeric) ----------
class NumericalFeatureTokenizer(nn.Module):
    """
    Turns (B, F) numeric matrix into (B, F, D) token embeddings:
    out[:, i, :] = x[:, i:i+1] * W[i] + b[i]
    where W: (F, D), b: (F, D) are learnable.
    """
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias   = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)
        # expand to (B, F, 1) * (F, D) -> (B, F, D)
        return x.unsqueeze(-1) * self.weight + self.bias

# ---------- Transformer Block ----------
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, attn_drop: float, ff_drop: float, prenorm=True):
        super().__init__()
        self.prenorm = prenorm
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=attn_drop, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(ff_drop),
            nn.Linear(d_ff, d_model),
            nn.Dropout(ff_drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.prenorm:
            z = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0]
            z = z + self.ff(self.ln2(z))
        else:
            z = self.ln1(x + self.attn(x, x, x, need_weights=False)[0])
            z = self.ln2(z + self.ff(z))
        return z

# ---------- FT-Transformer Model ----------
class FTTransformer(nn.Module):
    def __init__(self, n_features: int, d_model: int = 96, n_layers: int = 4, n_heads: int = 8,
                 d_ff_mult: float = 2.0, attn_drop: float = 0.1, ff_drop: float = 0.1,
                 cls_token: bool = True, feature_dropout: float = 0.0):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.tokenizer = NumericalFeatureTokenizer(n_features, d_model)
        self.cls_token = cls_token
        self.feature_dropout = feature_dropout

        if cls_token:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls, std=0.02)

        d_ff = int(d_model * d_ff_mult)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, attn_drop, ff_drop, prenorm=True)
        for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, F) numeric features (already standardized)
        returns logits: (B,)
        """
        tok = self.tokenizer(x)  # (B, F, D)

        if self.feature_dropout > 0.0 and self.training:
            # stochastic feature dropout (column dropout)
            drop_mask = (torch.rand(tok.shape[:2], device=tok.device) < self.feature_dropout).unsqueeze(-1)
            tok = tok.masked_fill(drop_mask, 0.0)

        if self.cls_token:
            B = tok.size(0)
            cls = self.cls.expand(B, -1, -1)  # (B,1,D)
            z = torch.cat([cls, tok], dim=1)  # (B, 1+F, D)
        else:
            z = tok

        for blk in self.blocks:
            z = blk(z)

        if self.cls_token:
            out = self.ln(z[:, 0, :])      # CLS
        else:
            out = self.ln(z.mean(dim=1))   # mean pool

        logits = self.head(out).squeeze(1)
        return logits

    def count_parameters(self):
        """Count total and trainable parameters in the model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total_params, trainable_params

# ---------- Training utilities ----------
def cosine_warmup_lr(optimizer, step, total_steps, warmup_ratio=0.06, base_lr=1e-3, min_lr=1e-6):
    warmup_steps = int(total_steps * warmup_ratio)
    if step < warmup_steps:
        lr = base_lr * (step / max(1, warmup_steps))
    else:
        t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        lr = min_lr + 0.5*(base_lr - min_lr)*(1 + math.cos(math.pi * t))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr

def train_fold(Xtr, ytr, Xva, yva, args, device, fold_num=1, use_wandb=False):
    # Standardize
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Xtr).astype(np.float32)
    Xva = scaler.transform(Xva).astype(np.float32)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.tensor(yva, dtype=torch.float32, device=device)

    model = FTTransformer(
        n_features=Xtr.shape[1],
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        d_ff_mult=args.ff_mult,
        attn_drop=args.attn_dropout,
        ff_drop=args.dropout,
        cls_token=True,
        feature_dropout=args.feature_dropout
    ).to(device)
    
    # Count parameters
    total_params, trainable_params = model.count_parameters()
    print(f"  Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Log model parameters to WandB (only for first fold to avoid duplicates)
    if use_wandb and fold_num == 1:
        wandb.log({
            "model/total_parameters": total_params,
            "model/trainable_parameters": trainable_params,
            "model/parameter_efficiency": trainable_params / Xtr.shape[1] if Xtr.shape[1] > 0 else 0
        })

    # Imbalance handling
    pos = float(ytr_t.sum().item()); neg = float(len(ytr_t) - ytr_t.sum().item())
    pos_weight = torch.tensor([(neg / max(pos, 1.0))], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler_amp = GradScaler(enabled=(device.type == "cuda"))

    # Steps and ES
    steps_per_epoch = math.ceil(len(Xtr_t) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    best_auc, best_state, best_epoch = -1, None, 0
    no_improve = 0

    for epoch in range(1, args.epochs+1):
        model.train()
        perm = torch.randperm(len(Xtr_t), device=device)
        for i in range(0, len(Xtr_t), args.batch_size):
            idx = perm[i:i+args.batch_size]
            xb = Xtr_t[idx]; yb = ytr_t[idx]
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler_amp.scale(loss).backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            # schedule
            global_step = (epoch-1)*steps_per_epoch + (i//args.batch_size) + 1
            cosine_warmup_lr(optimizer, global_step, total_steps,
                             warmup_ratio=args.warmup_ratio, base_lr=args.lr, min_lr=args.min_lr)

        # validation
        model.eval()
        with torch.no_grad(), autocast(enabled=(device.type == "cuda")):
            pv = torch.sigmoid(model(Xva_t)).float().cpu().numpy()
        auc = roc_auc_score(yva, pv); ap = average_precision_score(yva, pv)

        if auc > best_auc + 1e-5:
            best_auc, best_epoch, no_improve = auc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1

        print(f"  [epoch {epoch:03d}] loss={float(loss):.4f}  AUC={auc:.4f}  AP={ap:.4f}  "
              f"(best AUC={best_auc:.4f} @ {best_epoch})")
        
        # Log to WandB
        if use_wandb:
            wandb.log({
                f"fold_{fold_num}/train_loss": float(loss),
                f"fold_{fold_num}/val_auc": auc,
                f"fold_{fold_num}/val_ap": ap,
                f"fold_{fold_num}/best_auc": best_auc,
                f"fold_{fold_num}/epoch": epoch,
                f"fold_{fold_num}/learning_rate": optimizer.param_groups[0]["lr"],
                f"fold_{fold_num}/no_improve": no_improve
            })
        
        if no_improve >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # final validation predictions
    model.eval()
    with torch.no_grad():
        pva = torch.sigmoid(model(Xva_t)).float().cpu().numpy()

    return model, scaler, pva, {"best_auc": best_auc, "best_epoch": best_epoch}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/processed")
    ap.add_argument("--out-dir",   default="model_outputs/fttr")
    ap.add_argument("--n-splits",  type=int, default=5)
    ap.add_argument("--seed",      type=int, default=42)

    # Architecture
    ap.add_argument("--layers",    type=int, default=4)
    ap.add_argument("--d-model",   type=int, default=96)
    ap.add_argument("--heads",     type=int, default=8)
    ap.add_argument("--ff-mult",   type=float, default=2.0)
    ap.add_argument("--dropout",   type=float, default=0.15)
    ap.add_argument("--attn-dropout", type=float, default=0.10)
    ap.add_argument("--feature-dropout", type=float, default=0.05)

    # Optimization
    ap.add_argument("--epochs",    type=int, default=160)
    ap.add_argument("--batch-size",type=int, default=256)
    ap.add_argument("--lr",        type=float, default=1e-3)
    ap.add_argument("--min-lr",    type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--patience",  type=int, default=20)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--target",    default=None, help="Target column name in y_label.csv")
    
    # WandB options
    ap.add_argument("--wandb-project", default="ft-transformer-tabular", help="WandB project name")
    ap.add_argument("--wandb-run-name", default=None, help="WandB run name")
    ap.add_argument("--disable-wandb", action="store_true", help="Disable WandB logging")

    args = ap.parse_args()
    set_seed(args.seed)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[FTTR] device={device}")

    # Initialize WandB
    use_wandb = WANDB_AVAILABLE and not args.disable_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            tags=["ft-transformer", "tabular", "credit-risk"]
        )
        print(f"[FTTR] WandB logging enabled: {wandb.run.name}")
    else:
        print(f"[FTTR] WandB logging disabled")

    # Load data (already FE-ed and leakage-safe)
    X, y, tgt = load_xy(args.data_root, target_name=args.target)
    X = X.astype(np.float32).values
    y = y.astype(np.int64)
    prev = float(y.mean())
    print(f"[FTTR] X={X.shape}, positives={y.sum()}/{len(y)} (prev={prev:.2%}), target={tgt}")
    
    # Log dataset info to WandB
    if use_wandb:
        wandb.log({
            "dataset/n_samples": len(y),
            "dataset/n_features": X.shape[1],
            "dataset/positive_rate": prev,
            "dataset/n_positives": int(y.sum()),
            "dataset/target_name": tgt
        })

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(y), dtype=np.float32)
    folds = []

    last_model, last_scaler = None, None
    for k, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"\n== Fold {k}/{args.n_splits} ==")
        model, scaler, pva, info = train_fold(X[tr], y[tr], X[va], y[va], args, device, 
                                            fold_num=k, use_wandb=use_wandb)
        oof[va] = pva; folds.append({"fold": k, **info})
        last_model, last_scaler = model, scaler
        print(f"[fold {k}] best AUC={info['best_auc']:.4f} @ epoch {info['best_epoch']}")
        
        # Log fold summary to WandB
        if use_wandb:
            wandb.log({
                f"cv/fold_{k}_final_auc": info['best_auc'],
                f"cv/fold_{k}_best_epoch": info['best_epoch']
            })

    # Summaries
    smry, dec = summarize_all(y, oof, label="fttr")
    pd.DataFrame(folds).to_csv(Path(args.out_dir)/"folds_summary.csv", index=False)
    pd.DataFrame({"oof_fttr": oof, "y": y}).to_csv(Path(args.out_dir)/"oof_fttr.csv", index=False)
    dec.to_csv(Path(args.out_dir)/"deciles_fttr.csv", index=False)
    dump_json(smry, Path(args.out_dir)/"summary.json")

    # Save last trained fold model + scaler stats
    torch.save(last_model.state_dict(), Path(args.out_dir)/"fttr_state.pt")
    dump_json({"mean": last_scaler.mean_.tolist(), "scale": last_scaler.scale_.tolist()},
              Path(args.out_dir)/"scaler.json")

    # Log final results to WandB
    if use_wandb:
        # Log overall CV metrics
        wandb.log({
            "final/cv_auc_mean": smry.get("auc", 0),
            "final/cv_auc_std": np.std([f["best_auc"] for f in folds]),
            "final/cv_gini": smry.get("gini", 0),
            "final/cv_ks": smry.get("ks", 0),
            "final/cv_ap": smry.get("ap", 0),
            "final/cv_logloss": smry.get("logloss", 0),
            "final/cv_brier": smry.get("brier", 0)
        })
        
        # Log deciles table as wandb table
        wandb.log({"final/deciles_table": wandb.Table(dataframe=dec)})
        
        # Log artifacts
        try:
            wandb.save(str(Path(args.out_dir) / "*.csv"))
            wandb.save(str(Path(args.out_dir) / "*.json"))
            wandb.save(str(Path(args.out_dir) / "*.pt"))
        except:
            pass  # Don't fail if artifacts can't be saved
        
        wandb.finish()

    print("\n[FTTR] Summary")
    print(pd.DataFrame([{"model":"fttr", **smry}]).to_string(index=False))
    print(f"Artifacts → {args.out_dir}")

if __name__ == "__main__":
    main()