# HTS-Oracle X

**Cross-Attention Multimodal Regression for AI-Guided Drug Discovery**

HTS-Oracle X is a deep learning platform for virtual screening of large compound libraries against immune checkpoint targets. It combines ChemBERTa language model embeddings with RDKit molecular descriptors via bidirectional cross-attention, and outputs continuous binding affinity predictions (ΔFnorm) with uncertainty estimates via Monte Carlo Dropout.

Targets in this release: **CD28 · TIM-3 · VISTA**

---

## Architecture

| Component | Details |
|---|---|
| ChemBERTa branch | `seyonec/ChemBERTa-zinc-base-v1` CLS embeddings (768-d) |
| RDKit branch | Morgan 2048 + MACCS 167 + Torsion 1024 + 25 descriptors = **3264-d** |
| Fusion | Bidirectional cross-attention (4 heads, 256-d) |
| Output | Continuous ΔFnorm regression (Huber loss) |
| Uncertainty | Monte Carlo Dropout (10 inference passes) |
| Validation | Scaffold-aware 5-fold CV (Murcko partitioning) |
| Ensemble | 15 sub-models (3 feature methods × 5 folds) |

---

## Requirements

```bash
pip install rdkit transformers torch shap scikit-learn scipy joblib tqdm
```

Runs on **Google Colab** (GPU recommended). ChemBERTa is downloaded automatically from HuggingFace on first run.

---

## Input Files

Upload all files to `HTSOracle_v3/` in your Google Drive:

| File | Columns | Description |
|---|---|---|
| `training_library_CD28.csv` | ID, Smiles, Delta_Fnorm_percent, Hit | Dianthus-screened training set |
| `training_library_TIM3.csv` | ID, Smiles, Delta_Fnorm_percent, Hit | |
| `training_library_VISTA.csv` | ID, Smiles, Delta_Fnorm_percent, Hit | |
| `positives_CD28.csv` | ID | Confirmed binders (used for cross-validation) |
| `positives_TIM3.csv` | ID | |
| `positives_VISTA.csv` | ID | |
| `screen_library.csv` | ID, Smiles | Virtual screening library (100,160 compounds) |

---

## How to Run

```python
# Cell 1 — Install dependencies
!pip install rdkit transformers xgboost shap

# Cell 2 — Mount Drive and run
from google.colab import drive
drive.mount('/content/drive')
exec(open('/content/drive/MyDrive/HTSOracle_v3/HTSOracleX.py').read())
```

The script runs all three targets sequentially in a single session. ChemBERTa embeddings and RDKit features are computed once and shared across targets.

---

## Output Files

For each target (`CD28`, `TIM3`, `VISTA`):

| File | Description |
|---|---|
| `htsoracle_v3_{TARGET}_predictions.csv` | All 100,160 compounds ranked by Selection Score |
| `htsoracle_v3_{TARGET}_top50_primary.csv` | Top 50 compounds for purchase |
| `htsoracle_v3_{TARGET}_top100_backup.csv` | Backup list (ranks 51–100) |
| `htsoracle_v3_{TARGET}_model.pkl` | Saved ensemble (15 sub-models) |
| `htsoracle_v3_{TARGET}_performance.png` | Spearman R per fold + OOF predicted vs measured |
| `htsoracle_v3_{TARGET}_shap.png` | SHAP feature importance (top 20 RDKit features) |
| `htsoracle_v3_{TARGET}_report.txt` | Full performance summary |
| `htsoracle_v3_summary.csv` | Cross-target comparison table |

**Selection Score** = Predicted ΔFnorm − 0.5 × Uncertainty  
Compounds are ranked by this score, which penalizes high-uncertainty predictions.

---

## Configuration

Key parameters at the top of the script:

```python
TOP_N              = 50     # primary compounds per target
TOP_N_BACKUP       = 100    # backup list size
UNCERTAINTY_WEIGHT = 0.5    # uncertainty penalty in selection score
NUM_EPOCHS         = 20
PATIENCE           = 5      # early stopping
MC_PASSES          = 10     # Monte Carlo Dropout passes
```

---

## Citation

If you use HTS-Oracle X in your research, please cite:

> Gabr Lab, Columbia University Irving Medical Center — Department of Radiology  
> *HTS-Oracle X: Cross-Attention Multimodal Regression for AI-Guided Immune Checkpoint Drug Discovery* (2025)

---

## Contact

**Moustafa Gabr, PhD**  
Director of Molecular Imaging & Therapeutics Research  
Department of Radiology, Columbia University Irving Medical Center  
