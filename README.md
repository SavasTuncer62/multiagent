# Self-Verifying Multi-Agent Decision Support for Pneumothorax Analysis on CT

Code for the manuscript:

> **A Self-Verifying Multi-Agent Decision Support System for Pneumothorax Analysis on Computed Tomography: Segmentation Backbone, Failure Awareness and Report Consistency**
> Savaş Tunçer, Mustafa Yıldırım, Özal Yıldırım, Yusuf Doğan. *Manuscript under review.*

The system combines four rule-coordinated agents on axial CT slices:

| Agent | Role |
|---|---|
| A1 Triage | Classifies a slice as negative, uncertain or positive from the segmentation model's probability map |
| A2 Segmentation | Produces the mask with a U-Net backbone. SAM2 is used either for fusion or as an independent second opinion |
| A3 Verification | Interpretable checks (model–SAM2 agreement, body mask, air grey level, area range, reliability coefficient), patient-level laterality check, retries and human-review flags |
| A4 Reporting | Structured patient-level record and free text from Qwen2-VL-2B-Instruct, followed by a rule-based consistency check with regeneration and template fallback |

The study evaluates each component separately: post-processing of the segmentation backbone, the auxiliary lung task, the role of SAM2 (fusion vs. second opinion), feedback-driven re-segmentation, failure flagging and report consistency.

## Repository contents

```
study1_multi_agent_pneumothorax_ct/
├── pnx_ortak.py                       shared helper module (data split, U-Net and SAM2 inference, metrics)
├── requirements.txt
└── notebooks/
    ├── 1_segmentation_backbone.ipynb  post-processing grid, multi-task and single-task ResNet34 U-Net training
    ├── 2_multi_agent_system.ipynb     agents A1–A4, fusion and second-opinion modes, ablations, reports, statistics
    ├── 3_failure_examples.ipynb       rule-based selection and visualisation of failed test slices
    └── additional_analyses/
        ├── A1_wider_grid_and_nnunet.ipynb           wider post-processing grid and a 2D nnU-Net reference
        ├── A2_validation_score_and_seeds.ipynb      failure score fitted on validation slices, three training seeds
        └── A3_body_centred_hemithorax_split.ipynb   hemithorax split at the body-mask centroid, SAM2 leakage example
```

The notebooks in `additional_analyses/` are sensitivity analyses. They reuse the weights and outputs of the main notebooks and do not retrain the main models (except the nnU-Net reference and the extra seeds).

## Running order

All notebooks are written for Google Colab with a T4 or L4 GPU and Google Drive as storage. The first cells install the packages, clone SAM2, download the SAM2.1 checkpoint and write `pnx_ortak.py` to the runtime disk, so the helper module does not need to be uploaded separately.

1. `1_segmentation_backbone.ipynb` selects the post-processing settings on the validation patients and trains the multi-task and single-task ResNet34 U-Net models (about 45–60 min per model). The weights are written to `pnx_taban/`.
2. `2_multi_agent_system.ipynb` runs the agent configurations. Set `TABAN` (`'tek_gorevli'` = single-task, `'cok_gorevli'` = multi-task, `'temel'` = baseline U-Net) and `A2_MODU` (`'birlesim'` = fusion, `'ikinci_gorus'` = second opinion) at the top of the notebook. The main configuration is `TABAN = 'tek_gorevli'` (single-task) with `A2_MODU = 'ikinci_gorus'`.
3. `3_failure_examples.ipynb` reads the slice-level results of step 2 and reproduces the masks of the selected failed slices.
4. Optional: the notebooks in `additional_analyses/`.

## Expected data layout

The in-house dataset is **not included** (see *Data availability*). The notebooks expect the following folders on Google Drive:

| Variable | Default path | Content |
|---|---|---|
| `RAW_DIR` (in `pnx_ortak.py`) | `/content/drive/MyDrive/CT_goruntuler_anon` | One folder per patient with `N.jpg` slices and `N.png` label maps (0 background, 1 pneumothorax, 2 lung) |
| `DRIVE_BACKUP_DIR` (in `pnx_ortak.py`) | `/content/drive/MyDrive/pnx_artifacts` | Weights of the baseline MobileNetV2 U-Net (224 × 224 patches, `pnomotoraks_unet.pth`) |
| `OUT_DIR` / `MODEL_DIR` | `/content/drive/MyDrive/pnx_taban` | Output of notebook 1: `guclu_Unet_resnet34.pth` (multi-task) and `guclu_Unet_resnet34_akcigersiz.pth` (single-task) |

Only lower-case `.jpg`/`.png` pairs are loaded. Other extensions are skipped and listed. The baseline U-Net is loaded, not trained, in these notebooks. Its training settings are given in the manuscript.

To use a different storage location, change the paths at the top of each notebook and in `pnx_ortak.py`.

## Software

- Python 3, PyTorch (Colab default)
- `segmentation_models_pytorch==0.5.0`
- `transformers==5.21.0`, `accelerate`, `qwen-vl-utils` (Qwen2-VL-2B-Instruct for the reporting agent)
- SAM2 from https://github.com/facebookresearch/sam2 with the `sam2.1_hiera_small.pt` checkpoint
- OpenCV, SciPy, scikit-learn, pandas, matplotlib, Pillow
- `nnunetv2==2.8.1` (only for `A1_wider_grid_and_nnunet.ipynb`)

See `requirements.txt`. The installation cells in the notebooks pin the same versions.

## Reproducibility

- Patients are split at patient level with a fixed seed (42): 111 training, 17 validation and 43 test patients.
- All model, threshold and post-processing choices are made on the validation patients only.
- Python, NumPy and PyTorch seeds are fixed at 42. Report texts are sampled with a fixed seed.
- The notebooks write the slice-level and patient-level results as CSV files to their output folders. Some summary statistics and the final figures of the manuscript were computed from these CSV files.

## Notes

- Comments and variable names in the notebooks are partly in Turkish (ASCII characters only).
- Reports are generated in Turkish, and the consistency rules of the reporting agent operate on Turkish keywords.
- The system is a research prototype and is not intended for clinical use.

## Data availability

The CT dataset was collected retrospectively at a university hospital with ethics committee approval and is not publicly available because of patient privacy. It can be requested from the corresponding author, subject to ethics committee approval.

## Citation

The citation will be added once the article is published.

## Contact

Savaş Tunçer (corresponding author), savastuncer62@gmail.com
