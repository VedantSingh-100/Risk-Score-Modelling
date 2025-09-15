# train_tabmlp.py
import os, json, math, time, numpy as np, pandas as pd
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim

from ..utils.io_utils import load_xy, dump_json
from ..utils.metrics import summarize_all, decile_table

SEED = 42
torch.backends.cudnn.benchmark = True

def set_seed(seed=SEED):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def maybe_wandb(args):
    if not args.wandb:
        return None
    try:
        import wandb
        # Use specific entity and project for your CMU workspace
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project or "Risk_Score", 
            config=vars(args),
            name=f"tabmlp_baseline_seed{args.seed}"
        )
        return wandb
    except Exception as e:
        print(f"[wandb] disabled ({e})")
        return None

class TabMLP(nn.Module):
    def __init__(self, in_dim, hidden=[512,256,128], dropout=0.2):
        super().__init__()
        layers = []
        dim_prev = in_dim
        for h in hidden:
            layers += [
                nn.Linear(dim_prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout)
            ]
            dim_prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dim_prev, 1)

    def forward(self, x):
        z = self.backbone(x)
        logits = self.head(z)
        return logits.squeeze(1)

def train_one_fold(Xtr, ytr, Xva, yva, device, max_epochs=120, lr=1e-3, wd=1e-4, patience=15, wandb=None, fold_idx=None):
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xva_s = scaler.transform(Xva)

    Xtr_t = torch.tensor(Xtr_s, dtype=torch.float32).to(device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).to(device)
    Xva_t = torch.tensor(Xva_s, dtype=torch.float32).to(device)
    yva_t = torch.tensor(yva, dtype=torch.float32).to(device)

    model = TabMLP(in_dim=Xtr.shape[1]).to(device)

    # class imbalance
    pos = ytr.sum(); neg = len(ytr) - pos
    pos_weight = torch.tensor([(neg / max(pos, 1))], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best_auc, best_state, best_epoch = -1, None, 0
    no_improve = 0

    for epoch in range(1, max_epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(Xtr_t)
        loss = criterion(logits, ytr_t)
        loss.backward()
        optimizer.step()

        # val
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Xva_t)).cpu().numpy()
        auc = roc_auc_score(yva, pv)
        ap = average_precision_score(yva, pv)
        
        # Log to wandb
        if wandb is not None and fold_idx is not None:
            wandb.log({
                f"fold_{fold_idx}/train_loss": float(loss.item()),
                f"fold_{fold_idx}/val_auc": auc,
                f"fold_{fold_idx}/val_ap": ap,
                f"fold_{fold_idx}/epoch": epoch,
                f"fold_{fold_idx}/no_improve": no_improve
            })
        
        if auc > best_auc + 1e-5:
            best_auc = auc
            best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # predictions
    model.eval()
    with torch.no_grad():
        pva = torch.sigmoid(model(Xva_t)).cpu().numpy()

    return model, scaler, pva, {"best_auc": best_auc, "best_epoch": best_epoch}

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data/processed")
    ap.add_argument("--out-dir", default="model_outputs/tabmlp")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=140)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=18)
    ap.add_argument("--target", default=None, help="Target column in y_label.csv; default uses io_utils fallback.")
    ap.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    ap.add_argument("--wandb-project", default=None, help="W&B project name")
    ap.add_argument("--wandb-entity", default="ved100-carnegie-mellon-university", 
                    help="W&B entity (organization/username)")
    args = ap.parse_args()

    set_seed(args.seed)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    X, y, target = load_xy(args.data_root, target_name=args.target)
    X = X.astype(np.float32).values  # ensure dense float32
    y = y.astype(np.int64)
    print(f"Loaded X={X.shape}, positives={int(y.sum())}/{len(y)}  (target={target})")

    # Initialize wandb
    wandb = maybe_wandb(args)
    if wandb is not None:
        wandb.config.update({
            "n_features": X.shape[1],
            "n_samples": len(y),
            "n_positives": int(y.sum()),
            "prevalence": float(y.mean()),
            "target_name": target
        })

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(y), dtype=np.float32)
    fold_rows = []
    last_model = None; last_scaler = None

    for k, (tr, va) in enumerate(skf.split(X, y), 1):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]
        model, scaler, pva, info = train_one_fold(
            Xtr, ytr, Xva, yva, device,
            max_epochs=args.epochs, lr=args.lr, wd=args.wd, patience=args.patience,
            wandb=wandb, fold_idx=k
        )
        oof[va] = pva
        last_model, last_scaler = model, scaler
        ap = average_precision_score(yva, pva); auc = roc_auc_score(yva, pva)
        fold_rows.append({"fold": k, "val_auc": auc, "val_ap": ap, **info})
        
        # Log fold summary to wandb
        if wandb is not None:
            wandb.log({
                f"fold_{k}/final_val_auc": auc,
                f"fold_{k}/final_val_ap": ap,
                f"fold_{k}/best_val_auc": info['best_auc'],
                f"fold_{k}/best_epoch": info['best_epoch'],
                "fold": k
            })
        
        print(f"[fold {k}] AUC={auc:.4f} AP={ap:.4f} (best_auc={info['best_auc']:.4f} @epoch {info['best_epoch']})")

    smry, dec = summarize_all(y, oof, label="deep")
    pd.DataFrame(fold_rows).to_csv(Path(args.out_dir)/"folds_summary.csv", index=False)
    dec.to_csv(Path(args.out_dir)/"deciles_deep.csv", index=False)
    pd.DataFrame({"oof_deep": oof, "y": y}).to_csv(Path(args.out_dir)/"oof_deep.csv", index=False)
    dump_json(smry, Path(args.out_dir)/"summary.json")

    # Log final summary metrics to wandb
    if wandb is not None:
        # Overall CV metrics
        wandb.log({
            "cv_auc_mean": smry['auc'],
            "cv_ap_mean": smry['ap'],
            "cv_auc_std": np.std([row['val_auc'] for row in fold_rows]),
            "cv_ap_std": np.std([row['val_ap'] for row in fold_rows]),
            "mean_best_epoch": np.mean([row['best_epoch'] for row in fold_rows]),
            **{f"summary_{k}": v for k, v in smry.items()}
        })
        
        # Log fold-wise metrics as a table
        fold_metrics = []
        for row in fold_rows:
            fold_metrics.append([row['fold'], row['val_auc'], row['val_ap'], row['best_auc'], row['best_epoch']])
        
        fold_table = wandb.Table(
            columns=['Fold', 'Val_AUC', 'Val_AP', 'Best_AUC', 'Best_Epoch'],
            data=fold_metrics
        )
        wandb.log({"fold_summary_table": fold_table})
        
        wandb.finish()

    # Save last fold model + scaler (for inference template)
    torch.save(last_model.state_dict(), Path(args.out_dir)/"tabmlp_state.pt")
    # Save scaler params
    dump_json({"mean": last_scaler.mean_.tolist(), "scale": last_scaler.scale_.tolist()},
              Path(args.out_dir)/"scaler.json")

    print("\n[DEEP] Summary")
    print(pd.DataFrame([{"model":"tabmlp", **smry}]).to_string(index=False))
    print(f"Artifacts → {args.out_dir}")

if __name__ == "__main__":
    main()
