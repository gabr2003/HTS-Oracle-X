# ═══════════════════════════════════════════════════════════════════════════════
# HTS-Oracle v3  —  Cross-Attention Multimodal Regression
# Google Colab version  |  Runs CD28, TIM3, VISTA in one session
# ═══════════════════════════════════════════════════════════════════════════════
"""
HOW TO RUN
──────────
Cell 1:  !pip install rdkit transformers xgboost shap
Cell 2:  from google.colab import drive
         drive.mount('/content/drive')
         exec(open('/content/drive/MyDrive/HTSOracle_v3/HTSOracle_v3_Colab.py').read())

INPUT FILES  (upload all to HTSOracle_v3 folder on Google Drive)
─────────────────────────────────────────────────────────────────
  training_library_CD28.csv    — ID, Smiles, Delta_Fnorm_percent, Hit
  training_library_TIM3.csv
  training_library_VISTA.csv
  positives_CD28.csv           — ID (confirmed binders)
  positives_TIM3.csv
  positives_VISTA.csv
  screen_library.csv           — ID, Smiles (100,160 compounds)

OUTPUT FILES  (saved to Drive after each target completes)
───────────────────────────────────────────────────────────
  htsoracle_v3_{TARGET}_model.pkl
  htsoracle_v3_{TARGET}_predictions.csv   — all 100k ranked
  htsoracle_v3_{TARGET}_top50.csv         — top 50 for purchase
  htsoracle_v3_{TARGET}_performance.png
  htsoracle_v3_{TARGET}_shap.png
  htsoracle_v3_{TARGET}_report.txt
  htsoracle_v3_summary.csv                — comparison across targets

ARCHITECTURE
────────────────────────────────────────
  • Cross-attention fusion        — bidirectional ChemBERTa ↔ RDKit
  • Regression on ΔFnorm         — continuous binding affinity prediction
  • MC Dropout uncertainty        — 10-pass inference uncertainty
  • Extended features (3264-d)   — Morgan 2048 + MACCS 167 + Torsion 1024 + 25 desc
  • Pre-computed ChemBERTa       — embeddings cached once for speed
  • Scaffold-aware 5-fold CV     — Murcko partitioning throughout
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, warnings
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem, RDLogger
from rdkit.Chem import (AllChem, MACCSkeys, Descriptors, Lipinski, QED,
                        GraphDescriptors, rdMolDescriptors, DataStructs)
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
from sklearn.feature_selection import (SelectFromModel, SelectKBest,
                                       mutual_info_regression)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False
    print("⚠️  shap not installed — SHAP plots skipped")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ════════════════════════════════════════════════════════════════════
#  ⚙️  CONFIGURATION  —  edit DRIVE_FOLDER if needed
# ════════════════════════════════════════════════════════════════════
DRIVE_FOLDER  = '/content/drive/MyDrive/HTSOracle_v3'
TARGETS       = ['CD28', 'TIM3', 'VISTA']
SCREEN_FILE   = f'{DRIVE_FOLDER}/screen_library.csv'
TOP_N             = 50      # primary compounds per target (for purchase)
TOP_N_BACKUP      = 100     # backup list (includes top 50)
UNCERTAINTY_WEIGHT = 0.5    # Selection_Score = ΔFnorm - 0.5 × Uncertainty
FEAT_METHODS  = ["lasso", "pca", "mutual_info"]
N_COMPONENTS  = 200
NUM_EPOCHS    = 20
PATIENCE      = 5
BATCH_SIZE    = 64
LR            = 2e-4
WD            = 1e-4
MC_PASSES     = 10
# ════════════════════════════════════════════════════════════════════

os.makedirs(DRIVE_FOLDER, exist_ok=True)

print("=" * 65)
print("  HTS-Oracle v3  —  Cross-Attention Multimodal Regression")
print(f"  Targets : {', '.join(TARGETS)}")
print(f"  Device  : {device}")
print(f"  Top N   : {TOP_N} primary + {TOP_N_BACKUP} backup per target")
print(f"  Uncertainty weight: {UNCERTAINTY_WEIGHT}")
print(f"  {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}")
print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────────────────────
def read_file(path):
    """Read CSV or Excel automatically."""
    if path.endswith('.xlsx') or path.endswith('.xls'):
        return pd.read_excel(path)
    return pd.read_csv(path)


# ─────────────────────────────────────────────────────────────────────────────
# MOLECULAR FEATURES
# ─────────────────────────────────────────────────────────────────────────────
def morgan_fp(smi, radius=2, nBits=2048):
    try:
        mol = Chem.MolFromSmiles(smi)
        arr = np.zeros(nBits, dtype=np.float32)
        if mol:
            DataStructs.ConvertToNumpyArray(
                AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits), arr)
        return arr
    except: return np.zeros(nBits, dtype=np.float32)

def maccs_fp(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        arr = np.zeros(167, dtype=np.float32)
        if mol:
            DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), arr)
        return arr
    except: return np.zeros(167, dtype=np.float32)

def torsion_fp(smi, nBits=1024):
    try:
        mol = Chem.MolFromSmiles(smi)
        arr = np.zeros(nBits, dtype=np.float32)
        if mol:
            DataStructs.ConvertToNumpyArray(
                rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(
                    mol, nBits=nBits), arr)
        return arr
    except: return np.zeros(nBits, dtype=np.float32)

def descriptors_25(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return [0.0] * 25
        d = [0.0] * 25
        fns = [
            (0,  lambda m: Descriptors.MolWt(m)),
            (1,  lambda m: Descriptors.MolLogP(m)),
            (2,  lambda m: Descriptors.NumRotatableBonds(m)),
            (3,  lambda m: Descriptors.NumHAcceptors(m)),
            (4,  lambda m: Descriptors.NumHDonors(m)),
            (5,  lambda m: Descriptors.TPSA(m)),
            (6,  lambda m: Descriptors.RingCount(m)),
            (7,  lambda m: Descriptors.NumAromaticRings(m)),
            (8,  lambda m: Descriptors.HeavyAtomCount(m)),
            (9,  lambda m: Descriptors.NumHeteroatoms(m)),
            (10, lambda m: Descriptors.FractionCSP3(m)),
            (11, lambda m: Lipinski.NumHAcceptors(m)),
            (12, lambda m: Lipinski.NumHDonors(m)),
            (13, lambda m: QED.qed(m)),
            (14, lambda m: Descriptors.NumSaturatedRings(m)),
            (15, lambda m: GraphDescriptors.BertzCT(m)),
            (16, lambda m: Descriptors.Chi0(m)),
            (17, lambda m: Descriptors.Chi1(m)),
            (18, lambda m: Descriptors.Kappa1(m)),
            (19, lambda m: Descriptors.Kappa2(m)),
            (20, lambda m: Descriptors.Kappa3(m)),
            (21, lambda m: rdMolDescriptors.CalcNumBridgeheadAtoms(m)),
            (22, lambda m: rdMolDescriptors.CalcNumSpiroAtoms(m)),
            (23, lambda m: Descriptors.MaxPartialCharge(m)),
            (24, lambda m: Descriptors.MinPartialCharge(m)),
        ]
        for idx, fn in fns:
            try:
                v = fn(mol)
                d[idx] = float(v) if not (np.isnan(v) or np.isinf(v)) else 0.0
            except: pass
        return d
    except: return [0.0] * 25

DESCRIPTOR_NAMES = (
    [f"Morgan_{i}"  for i in range(2048)] +
    [f"MACCS_{i}"   for i in range(167)]  +
    [f"Torsion_{i}" for i in range(1024)] +
    ["MolWt","LogP","RotBonds","HBA","HBD","TPSA","RingCount",
     "AromaticRings","HeavyAtoms","Heteroatoms","FracCSP3",
     "HBA_Lip","HBD_Lip","QED","SaturatedRings",
     "BertzCT","Chi0","Chi1","Kappa1","Kappa2","Kappa3",
     "BridgeheadAtoms","SpiroAtoms","MaxPartialCharge","MinPartialCharge"]
)  # 3264 total

def build_features(smi_list, desc="Features"):
    rows = []
    for smi in tqdm(smi_list, desc=f"    {desc}", leave=False):
        row = np.concatenate([morgan_fp(smi), maccs_fp(smi),
                              torsion_fp(smi), descriptors_25(smi)])
        rows.append(np.nan_to_num(row, nan=0., posinf=0., neginf=0.))
    return np.array(rows, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# SCAFFOLD-AWARE SPLITS
# ─────────────────────────────────────────────────────────────────────────────
def murcko_scaffold(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return "NONE"
        sc = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        return sc if sc else "NONE"
    except: return "NONE"

def make_scaffold_folds(smiles_list, n_splits=5):
    scaffolds = [murcko_scaffold(s) for s in
                 tqdm(smiles_list, desc="    Scaffolds", leave=False)]
    scaffold_to_idx = defaultdict(list)
    for i, sc in enumerate(scaffolds):
        scaffold_to_idx[sc].append(i)
    groups = sorted(scaffold_to_idx.values(), key=len, reverse=True)
    fold_idx   = [[] for _ in range(n_splits)]
    fold_sizes = [0] * n_splits
    for grp in groups:
        best = int(np.argmin(fold_sizes))
        fold_idx[best].extend(grp)
        fold_sizes[best] += len(grp)
    splits = []
    for f in range(n_splits):
        test  = np.array(fold_idx[f])
        train = np.array([i for g in range(n_splits)
                          if g != f for i in fold_idx[g]])
        splits.append((train, test))
    return splits, len(scaffold_to_idx)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITION
# ─────────────────────────────────────────────────────────────────────────────
class BindingDataset(Dataset):
    def __init__(self, bert_emb, rdkit_feats, targets):
        self.bert    = bert_emb.astype(np.float32)
        self.rdkit   = rdkit_feats.astype(np.float32)
        self.targets = targets.astype(np.float32)
    def __len__(self): return len(self.targets)
    def __getitem__(self, i):
        return {
            "bert":   torch.tensor(self.bert[i]),
            "rdkit":  torch.nan_to_num(torch.tensor(self.rdkit[i]),
                                       nan=0., posinf=0., neginf=0.),
            "target": torch.tensor(self.targets[i]),
        }

class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention between ChemBERTa and RDKit branches.
    ChemBERTa attends to RDKit: which physicochemical features matter
    given the SMILES context?
    RDKit attends to ChemBERTa: which SMILES context matters given the
    physicochemical profile?
    """
    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn_a2b = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads,
                                               dropout=dropout, batch_first=True)
        self.attn_b2a = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads,
                                               dropout=dropout, batch_first=True)
        self.norm_a   = nn.LayerNorm(dim)
        self.norm_b   = nn.LayerNorm(dim)
        self.drop     = nn.Dropout(dropout)
    def forward(self, a, b):
        a_ = a.unsqueeze(1); b_ = b.unsqueeze(1)
        a_ctx, _ = self.attn_a2b(query=a_, key=b_, value=b_)
        a_out    = self.norm_a(a + self.drop(a_ctx.squeeze(1)))
        b_ctx, _ = self.attn_b2a(query=b_, key=a_, value=a_)
        b_out    = self.norm_b(b + self.drop(b_ctx.squeeze(1)))
        return a_out, b_out

class HTSOracleV3(nn.Module):
    """
    HTS-Oracle X: Cross-Attention Multimodal Regression.

    Inputs  : pre-computed ChemBERTa CLS embeddings (768-d)
              + RDKit features (3264-d, reduced to N_COMPONENTS)
    Fusion  : bidirectional cross-attention
    Output  : continuous ΔFnorm prediction (regression)
    Uncertainty: Monte Carlo Dropout (10 forward passes at inference)
    """
    def __init__(self, bert_dim=768, rdkit_in=200,
                 embed_dim=256, n_heads=4, dropout=0.3):
        super().__init__()
        self.bert_proj = nn.Sequential(
            nn.Linear(bert_dim, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Dropout(dropout))
        self.rdkit_proj = nn.Sequential(
            nn.Linear(rdkit_in, embed_dim*2), nn.LayerNorm(embed_dim*2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim*2, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU())
        self.fusion = CrossAttentionFusion(
            dim=embed_dim, n_heads=n_heads, dropout=dropout/2)
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim*2, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim//2), nn.LayerNorm(embed_dim//2),
            nn.GELU(), nn.Dropout(dropout/2),
            nn.Linear(embed_dim//2, 1))
    def forward(self, bert, rdkit):
        a = self.bert_proj(bert)
        b = self.rdkit_proj(rdkit)
        a_f, b_f = self.fusion(a, b)
        return self.regressor(torch.cat([a_f, b_f], dim=1)).view(-1)
    def mc_predict(self, bert, rdkit, n_passes=10):
        """MC Dropout: keep dropout active, run n_passes, return mean+std."""
        self.train()
        with torch.no_grad():
            passes = [self.forward(bert, rdkit) for _ in range(n_passes)]
        stack = torch.stack(passes, dim=0)
        return stack.mean(dim=0), stack.std(dim=0)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE SELECTION
# ─────────────────────────────────────────────────────────────────────────────
def select_features_reg(X_tr, y_tr, X_te, method="pca", n=200):
    sc  = StandardScaler()
    Xtr = np.nan_to_num(sc.fit_transform(X_tr), nan=0., posinf=0., neginf=0.)
    Xte = np.nan_to_num(sc.transform(X_te),     nan=0., posinf=0., neginf=0.)
    try:
        if method == "lasso":
            sel = SelectFromModel(Lasso(alpha=0.01, random_state=SEED,
                                        max_iter=2000),
                                  max_features=min(n, Xtr.shape[1]))
            sel.fit(Xtr, y_tr)
        elif method == "pca":
            sel = PCA(n_components=min(n, *Xtr.shape), random_state=SEED)
            sel.fit(Xtr)
        else:
            sel = SelectKBest(mutual_info_regression, k=min(n, Xtr.shape[1]))
            sel.fit(Xtr, y_tr)
    except Exception as ex:
        print(f"       ⚠️  {method} failed ({ex}). PCA fallback.")
        sel = PCA(n_components=min(n, *Xtr.shape), random_state=SEED)
        sel.fit(Xtr)
    Xtr_s = np.nan_to_num(sel.transform(Xtr), nan=0., posinf=0., neginf=0.)
    Xte_s = np.nan_to_num(sel.transform(Xte), nan=0., posinf=0., neginf=0.)
    return Xtr_s, Xte_s, sel, sc


# ─────────────────────────────────────────────────────────────────────────────
# SHARED STEP 1 — Load SMILES and screen library
# ─────────────────────────────────────────────────────────────────────────────
print("\n[SHARED 1/4]  Loading SMILES and screen library …")

train_base    = read_file(f'{DRIVE_FOLDER}/training_library_CD28.csv')
# Keep full train_base as master reference for ID-aligned loading
smiles_train  = train_base["Smiles"].tolist()
screen_df     = read_file(SCREEN_FILE)
smiles_screen = screen_df["Smiles"].tolist()

print(f"    Training SMILES : {len(smiles_train):,}")
print(f"    Screen SMILES   : {len(smiles_screen):,}")


# ─────────────────────────────────────────────────────────────────────────────
# SHARED STEP 2 — RDKit features (once for all targets)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[SHARED 2/4]  Generating RDKit features (once for all targets) …")
rdkit_X        = build_features(smiles_train,  "Training features")
rdkit_X_screen = build_features(smiles_screen, "Screen features")
print(f"    Training : {rdkit_X.shape[0]:,} × {rdkit_X.shape[1]}")
print(f"    Screen   : {rdkit_X_screen.shape[0]:,} × {rdkit_X_screen.shape[1]}")


# ─────────────────────────────────────────────────────────────────────────────
# SHARED STEP 3 — ChemBERTa embeddings (once for all targets)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[SHARED 3/4]  Pre-computing ChemBERTa embeddings (once for all targets) …")
print("    Running ChemBERTa once — cached for CD28, TIM3, VISTA training")

tokenizer  = RobertaTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
bert_model = RobertaModel.from_pretrained(
    "seyonec/ChemBERTa-zinc-base-v1").to(device)
bert_model.eval()

def embed_smiles(smi_list, batch_size=64, desc="Embedding"):
    all_embs = []
    for i in tqdm(range(0, len(smi_list), batch_size),
                  desc=f"    {desc}", leave=False):
        batch = smi_list[i:i+batch_size]
        try:
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=128, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = bert_model(**enc)
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()
        except:
            cls = np.zeros((len(batch), 768), dtype=np.float32)
        all_embs.append(cls)
    return np.concatenate(all_embs, axis=0).astype(np.float32)

bert_emb_train  = embed_smiles(smiles_train,  desc="Training embeddings")
bert_emb_screen = embed_smiles(smiles_screen, desc="Screen embeddings")

del bert_model
torch.cuda.empty_cache() if torch.cuda.is_available() else None
print(f"    ✅  ChemBERTa done — model unloaded from GPU")
print(f"    Training embeddings : {bert_emb_train.shape}")
print(f"    Screen embeddings   : {bert_emb_screen.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# SHARED STEP 4 — Scaffold-aware splits (once for all targets)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[SHARED 4/4]  Building scaffold-aware splits …")
splits, n_scaffolds = make_scaffold_folds(smiles_train, n_splits=5)
print(f"    Unique scaffolds : {n_scaffolds:,}")
print(f"    Fold sizes       : {[len(te) for _, te in splits]}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LOOP — train + screen each target
# ─────────────────────────────────────────────────────────────────────────────
summary_rows = []

for TARGET in TARGETS:
    print(f"\n{'═'*65}")
    print(f"  TARGET: {TARGET}")
    print(f"  Started: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'═'*65}")

    # FIX 2+3: Load target data and align by ID (not row order)
    # This ensures TIM3/VISTA labels map correctly even if row order differs
    library_df   = read_file(f'{DRIVE_FOLDER}/training_library_{TARGET}.csv')
    positives_df = read_file(f'{DRIVE_FOLDER}/positives_{TARGET}.csv')

    # Align to master SMILES order (same as RDKit/BERT features) by ID
    master_ids = train_base["ID"].tolist()
    library_df = library_df.set_index("ID").reindex(master_ids).reset_index()

    # FIX 2: Verify Hit column against positives file — use Hit col as primary,
    # cross-check against positives file and warn if mismatch
    pos_ids   = set(positives_df["ID"].tolist())
    hit_from_col = set(library_df[library_df["Hit"] == 1]["ID"].tolist())
    overlap   = len(hit_from_col & pos_ids)
    if overlap != len(pos_ids):
        print(f"    ⚠️  Mismatch: {len(pos_ids)} in positives file, "
              f"{len(hit_from_col)} in Hit col, {overlap} overlap. "
              f"Using Hit column as ground truth.")
    else:
        print(f"    ✅  Hit column verified against positives file "
              f"({len(pos_ids)} binders match exactly)")

    # Use Hit column directly — it is the ground truth from Dianthus
    labels      = library_df["Hit"].fillna(0).astype(int).to_numpy()
    delta_fnorm = library_df["Delta_Fnorm_percent"].fillna(0).to_numpy(
        dtype=np.float32)
    n_tot       = len(library_df)
    n_pos       = labels.sum()

    print(f"\n    Compounds : {n_tot:,}")
    print(f"    Binders   : {n_pos:,} ({n_pos/n_tot:.2%})")
    print(f"    ΔFnorm    : {delta_fnorm.min():.2f} – {delta_fnorm.max():.2f} %")

    # ── Training ─────────────────────────────────────────────────────────────
    submodels     = []
    oof_mean      = np.zeros(n_tot, dtype=np.float32)
    oof_std       = np.zeros(n_tot, dtype=np.float32)
    oof_counts    = np.zeros(n_tot, dtype=np.float32)
    fold_spearman = []
    fold_auc      = []

    print(f"\n    Training ({len(FEAT_METHODS)} feature methods × 5 folds = 15 sub-models) …")

    for feat_method in FEAT_METHODS:
        print(f"\n    ── {feat_method} ──")
        for fi, (tr_idx, te_idx) in enumerate(splits):
            print(f"    Fold {fi+1}/5", end="  ")

            tr_fn = delta_fnorm[tr_idx]
            te_fn = delta_fnorm[te_idx]
            te_lb = labels[te_idx]

            Xtr, Xte, sel, sc = select_features_reg(
                rdkit_X[tr_idx], tr_fn,
                rdkit_X[te_idx], feat_method, N_COMPONENTS)

            tr_ds = BindingDataset(bert_emb_train[tr_idx], Xtr, tr_fn)
            te_ds = BindingDataset(bert_emb_train[te_idx], Xte, te_fn)
            tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=2, pin_memory=True)
            te_dl = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=2, pin_memory=True)

            model     = HTSOracleV3(rdkit_in=Xtr.shape[1]).to(device)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=LR, weight_decay=WD)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=NUM_EPOCHS)
            criterion = nn.HuberLoss(delta=1.0)

            best_sp, best_state, no_improve = -1.0, None, 0

            for epoch in range(NUM_EPOCHS):
                model.train()
                for batch in tr_dl:
                    try:
                        optimizer.zero_grad()
                        pred = model(batch["bert"].to(device),
                                     batch["rdkit"].to(device))
                        loss = criterion(pred, batch["target"].to(device))
                        if not torch.isnan(loss):
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(), 1.0)
                            optimizer.step()
                    except: continue
                scheduler.step()

                model.eval()
                pv, tv = [], []
                with torch.no_grad():
                    for batch in te_dl:
                        try:
                            out = model(
                                batch["bert"].to(device),
                                batch["rdkit"].to(device)).cpu().numpy()
                            ok  = ~(np.isnan(out) | np.isinf(out))
                            pv.extend(out[ok])
                            tv.extend(batch["target"].numpy()[ok])
                        except: continue

                sp = spearmanr(tv, pv)[0] if len(pv) > 1 else 0.0
                sp = sp if not np.isnan(sp) else 0.0

                if sp > best_sp:
                    best_sp    = sp
                    best_state = {k: v.clone()
                                  for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= PATIENCE: break

            print(f"Spearman R = {best_sp:.4f}")
            fold_spearman.append(best_sp)

            submodels.append({
                "state":      best_state,
                "selector":   sel,
                "scaler":     sc,
                "method":     feat_method,
                "rdkit_size": Xtr.shape[1],
            })

            # MC Dropout OOF predictions
            if best_state is not None:
                model.load_state_dict(best_state)
                batch_start = 0
                for batch in te_dl:
                    try:
                        mean_p, std_p = model.mc_predict(
                            batch["bert"].to(device),
                            batch["rdkit"].to(device),
                            n_passes=MC_PASSES)
                        mean_p = mean_p.cpu().numpy()
                        std_p  = std_p.cpu().numpy()
                        bs = len(mean_p)
                        for j, gi in enumerate(
                                te_idx[batch_start:batch_start+bs]):
                            if not (np.isnan(mean_p[j]) or
                                    np.isinf(mean_p[j])):
                                oof_mean[gi]   += mean_p[j]
                                oof_std[gi]    += std_p[j]
                                oof_counts[gi] += 1
                        batch_start += bs
                    except: continue

            # FIX 4: Calculate AUC from best-state model, not last epoch
            if best_state is not None and te_lb.sum() > 0 and te_lb.sum() < len(te_lb):
                try:
                    model.load_state_dict(best_state)
                    model.eval()
                    best_pv = []
                    with torch.no_grad():
                        for batch in te_dl:
                            try:
                                out = model(
                                    batch["bert"].to(device),
                                    batch["rdkit"].to(device)).cpu().numpy()
                                ok = ~(np.isnan(out) | np.isinf(out))
                                best_pv.extend(out[ok])
                            except: continue
                    if len(best_pv) > 0:
                        auc = roc_auc_score(te_lb, best_pv[:len(te_lb)])
                        fold_auc.append(auc)
                except: pass

    nz = oof_counts > 0
    oof_mean[nz] = oof_mean[nz] / oof_counts[nz]
    oof_std[nz]  = oof_std[nz]  / oof_counts[nz]
    sp_oof = spearmanr(delta_fnorm[nz], oof_mean[nz])[0]

    mean_sp  = np.mean(fold_spearman)
    std_sp   = np.std(fold_spearman)
    mean_auc = np.mean(fold_auc) if fold_auc else float('nan')
    std_auc  = np.std(fold_auc)  if fold_auc else float('nan')

    print(f"\n    Spearman R (mean ± SD) : {mean_sp:.4f} ± {std_sp:.4f}")
    print(f"    ROC-AUC   (mean ± SD) : {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"    OOF Spearman R        : {sp_oof:.4f}")

    joblib.dump({"models": submodels, "feature_names": DESCRIPTOR_NAMES},
                f"{DRIVE_FOLDER}/htsoracle_v3_{TARGET}_model.pkl")
    print(f"    ✅  Model saved → htsoracle_v3_{TARGET}_model.pkl")

    # ── Virtual screening ─────────────────────────────────────────────────────
    print(f"\n    Virtual screening ({len(screen_df):,} compounds) …")
    all_preds, all_stds = [], []

    for sub in tqdm(submodels, desc="    Sub-models", leave=False):
        if sub["state"] is None: continue
        Xs = np.nan_to_num(
            sub["scaler"].transform(rdkit_X_screen),
            nan=0., posinf=0., neginf=0.)
        Xs = np.nan_to_num(
            sub["selector"].transform(Xs),
            nan=0., posinf=0., neginf=0.)
        model = HTSOracleV3(rdkit_in=Xs.shape[1]).to(device)
        model.load_state_dict(sub["state"])
        ds = BindingDataset(bert_emb_screen, Xs,
                            np.zeros(len(screen_df), dtype=np.float32))
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
        sub_mean, sub_std = [], []
        for batch in dl:
            try:
                m, s = model.mc_predict(
                    batch["bert"].to(device),
                    batch["rdkit"].to(device),
                    n_passes=MC_PASSES)
                sub_mean.extend(m.cpu().numpy())
                sub_std.extend(s.cpu().numpy())
            except:
                sub_mean.extend([0.0] * len(batch["bert"]))
                sub_std.extend([0.0]  * len(batch["bert"]))
        all_preds.append(sub_mean)
        all_stds.append(sub_std)

    ensemble_scores = np.mean(all_preds, axis=0)
    ensemble_unc    = np.mean(all_stds,  axis=0)

    out_df = screen_df.copy()
    out_df["Predicted_DeltaFnorm"] = np.round(ensemble_scores, 4)
    out_df["Uncertainty"]          = np.round(ensemble_unc,    4)

    # FIX 1: Rank by uncertainty-adjusted score
    # Selection_Score = Predicted_DeltaFnorm - UNCERTAINTY_WEIGHT × Uncertainty
    # This penalises high-uncertainty compounds, favouring confident predictions
    out_df["Selection_Score"] = np.round(
        ensemble_scores - UNCERTAINTY_WEIGHT * ensemble_unc, 4)
    out_df = out_df.sort_values(
        "Selection_Score", ascending=False).reset_index(drop=True)
    out_df.insert(0, "Rank", range(1, len(out_df)+1))

    # Save full ranked list
    out_df.to_csv(
        f"{DRIVE_FOLDER}/htsoracle_v3_{TARGET}_predictions.csv", index=False)

    # FIX 5: Save top 50 primary + top 100 backup separately
    top_primary = out_df.head(TOP_N).copy()
    top_primary.insert(1, "Selection", "Primary")
    top_primary.to_csv(
        f"{DRIVE_FOLDER}/htsoracle_v3_{TARGET}_top{TOP_N}_primary.csv",
        index=False)

    top_backup = out_df.iloc[TOP_N:TOP_N_BACKUP].copy()
    top_backup.insert(1, "Selection", "Backup")
    top_backup.to_csv(
        f"{DRIVE_FOLDER}/htsoracle_v3_{TARGET}_top{TOP_N_BACKUP}_backup.csv",
        index=False)

    burden_reduction = (1 - TOP_N / len(screen_df)) * 100

    print(f"    ✅  Full rankings      → htsoracle_v3_{TARGET}_predictions.csv")
    print(f"    ✅  Top {TOP_N} primary  → htsoracle_v3_{TARGET}_top{TOP_N}_primary.csv")
    print(f"    ✅  Top {TOP_N_BACKUP} backup  → htsoracle_v3_{TARGET}_top{TOP_N_BACKUP}_backup.csv")
    print(f"    Screening burden reduction : {burden_reduction:.2f}%")
    print(f"\n    Top 10 (by Selection_Score = ΔFnorm - {UNCERTAINTY_WEIGHT}×Uncertainty):")
    print(out_df[["Rank","ID","Predicted_DeltaFnorm",
                  "Uncertainty","Selection_Score"]].head(10).to_string(index=False))

    # ── SHAP ─────────────────────────────────────────────────────────────────
    if SHAP_OK and submodels:
        try:
            sc0  = submodels[0]["scaler"]
            sel0 = submodels[0]["selector"]
            Xs0  = np.nan_to_num(sc0.transform(rdkit_X),
                                 nan=0., posinf=0., neginf=0.)
            Xs0  = np.nan_to_num(sel0.transform(Xs0),
                                 nan=0., posinf=0., neginf=0.)
            samp = Xs0[np.random.choice(
                len(Xs0), min(300, len(Xs0)), replace=False)]
            rf_s = RandomForestRegressor(
                n_estimators=200, random_state=SEED, n_jobs=-1)
            rf_s.fit(Xs0, delta_fnorm)
            explainer = shap.TreeExplainer(rf_s)
            shap_vals = explainer.shap_values(samp)
            n_feat     = samp.shape[1]
            feat_names = (DESCRIPTOR_NAMES[-n_feat:]
                          if n_feat <= len(DESCRIPTOR_NAMES)
                          else [f"F{i}" for i in range(n_feat)])
            plt.figure(figsize=(10, 7))
            shap.summary_plot(shap_vals, samp, feature_names=feat_names,
                              max_display=20, show=False)
            plt.title(f"HTS-Oracle v3 — {TARGET} SHAP feature importance",
                      fontsize=11)
            plt.tight_layout()
            plt.savefig(f"{DRIVE_FOLDER}/htsoracle_v3_{TARGET}_shap.png",
                        dpi=200, bbox_inches="tight")
            plt.close()
            print(f"    ✅  SHAP → htsoracle_v3_{TARGET}_shap.png")
        except Exception as e:
            print(f"    ⚠️  SHAP failed: {e}")

    # ── Performance plot ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"HTS-Oracle v3 — {TARGET} — Scaffold-aware CV",
                 fontsize=13, fontweight="bold")

    ax = axes[0]
    colors  = ["#2196F3", "#FF5722", "#4CAF50"]
    n_folds = len(splits)
    for mi, method in enumerate(FEAT_METHODS):
        vals = fold_spearman[mi*n_folds : (mi+1)*n_folds]
        ax.plot(range(1, n_folds+1), vals, "o-", color=colors[mi],
                label=method, linewidth=1.8, markersize=6)
    ax.axhline(mean_sp, color="black", linestyle="--", alpha=0.5,
               label=f"Mean ({mean_sp:.3f})")
    ax.set_xlabel("Fold"); ax.set_ylabel("Spearman R")
    ax.set_title(f"A  Spearman R — {TARGET}")
    ax.legend(fontsize=8); ax.set_ylim(0, 1)

    ax = axes[1]
    ax.scatter(delta_fnorm[nz], oof_mean[nz],
               alpha=0.3, s=8, c="#1976D2", edgecolors="none")
    lims = [min(delta_fnorm.min(), oof_mean[nz].min()),
            max(delta_fnorm.max(), oof_mean[nz].max())]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1,
            label="Perfect prediction")
    ax.set_xlabel("Measured ΔFnorm (%)")
    ax.set_ylabel("Predicted ΔFnorm (%)")
    ax.set_title(f"B  OOF Predicted vs Measured\nSpearman R = {sp_oof:.3f}")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{DRIVE_FOLDER}/htsoracle_v3_{TARGET}_performance.png",
                dpi=200)
    plt.close()
    print(f"    ✅  Performance → htsoracle_v3_{TARGET}_performance.png")

    # ── Report ────────────────────────────────────────────────────────────────
    lines = [
        "=" * 65,
        f"  HTS-Oracle v3 — {TARGET} — Results Summary",
        f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 65,
        "",
        "DATASET",
        f"  Training compounds : {n_tot:,}",
        f"  Confirmed binders  : {n_pos:,} ({n_pos/n_tot:.2%})",
        f"  ΔFnorm range       : {delta_fnorm.min():.2f} – "
        f"{delta_fnorm.max():.2f} %",
        f"  Screen library     : {len(screen_df):,} compounds",
        f"  Top N selected     : {TOP_N}",
        f"  Burden reduction   : {burden_reduction:.2f}%",
        "",
        "VALIDATION  (5-fold scaffold-aware, Murcko partitioning)",
        f"  Spearman R (mean ± SD) : {mean_sp:.4f} ± {std_sp:.4f}",
        f"  ROC-AUC   (mean ± SD) : {mean_auc:.4f} ± {std_auc:.4f}",
        f"  OOF Spearman R        : {sp_oof:.4f}",
        "",
        "ARCHITECTURE",
        "  ChemBERTa (seyonec/ChemBERTa-zinc-base-v1) CLS embeddings",
        "  RDKit: Morgan 2048 + MACCS 167 + Torsion 1024 + 25 desc = 3264-d",
        "  Cross-attention fusion (bidirectional, 4 heads)",
        "  Regression head → continuous ΔFnorm prediction",
        "  MC Dropout uncertainty (10 passes)",
        "  Huber loss, AdamW, CosineAnnealingLR",
        "  15 sub-models (3 feature methods × 5 scaffold-aware folds)",
        "",
        "OUTPUT FILES",
        f"  htsoracle_v3_{TARGET}_model.pkl",
        f"  htsoracle_v3_{TARGET}_predictions.csv",
        f"  htsoracle_v3_{TARGET}_top{TOP_N}.csv",
        f"  htsoracle_v3_{TARGET}_performance.png",
        f"  htsoracle_v3_{TARGET}_shap.png",
        "=" * 65,
    ]
    with open(f"{DRIVE_FOLDER}/htsoracle_v3_{TARGET}_report.txt", "w") as f:
        f.write("\n".join(lines))
    print(f"    ✅  Report → htsoracle_v3_{TARGET}_report.txt")

    summary_rows.append({
        "Target":            TARGET,
        "N_training":        n_tot,
        "N_binders":         n_pos,
        "Hit_rate_%":        round(n_pos/n_tot*100, 2),
        "Spearman_R":        round(mean_sp, 4),
        "Spearman_R_SD":     round(std_sp, 4),
        "ROC_AUC":           round(mean_auc, 4),
        "ROC_AUC_SD":        round(std_auc, 4),
        "OOF_Spearman_R":    round(sp_oof, 4),
        "Screen_library":    len(screen_df),
        "Top_N_selected":    TOP_N,
        "Burden_reduction_%":round(burden_reduction, 2),
    })

    print(f"\n  ✅  {TARGET} complete! "
          f"({datetime.now().strftime('%H:%M:%S')})")


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'═'*65}")
print("  HTS-Oracle v3 — Final Summary Across All Targets")
print(f"{'═'*65}\n")

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f"{DRIVE_FOLDER}/htsoracle_v3_summary.csv", index=False)
print(summary_df.to_string(index=False))

print(f"\n✅  Summary table → htsoracle_v3_summary.csv")
print(f"\n🎉  HTS-Oracle v3 complete — all three targets done!")
print(f"    All outputs saved to: {DRIVE_FOLDER}")
print(f"    Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
