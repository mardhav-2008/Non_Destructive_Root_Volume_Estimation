
# Non-Destructive Root Volume Estimation

**Project G14 — Crop Breeding Trials**

A deep-learning project for estimating crop root volume from underground scan images without physically disturbing the root system.

The project uses the CGIAR Root Volume Estimation dataset from Zindi and investigates image-based regression, metadata-assisted prediction, image-area baselines, and uncertainty quantification.

---

## 1. Objectives

The main objectives are:

- Develop a CNN regressor to estimate root volume from underground scan images.
- Compare CNN predictions against image-area heuristic and metadata-based baselines.
- Evaluate models using RMSE, MAE, and R².
- Estimate prediction intervals and evaluate their empirical coverage.
- Develop a simple interface that returns a root-volume estimate with an uncertainty range.

## 2. Dataset

**Source:** CGIAR Root Volume Estimation Challenge — Zindi.

The dataset contains:

- `Train.csv`: labelled observations.
- `Test.csv`: observations without root-volume labels.
- `Sample_Submission.csv`: submission format.
- `data/train/`: training scan images.
- `data/test/`: test scan images.

Training data contains 386 labelled observations across 98 folders.

Each observation includes metadata such as:

- FolderName
- PlantNumber
- Side
- Start
- End
- Genotype
- Stage
- RootVolume

The scan images are grayscale images stored in RGBA format. An observation consists of multiple image strips, which are stacked vertically in slice order.

## 3. Methodology

### Image preprocessing

1. Load the image strips belonging to an observation.
2. Convert them to grayscale.
3. Stack strips vertically.
4. Resize to a width of 512 pixels while preserving aspect ratio.
5. Pad to a height of 256 pixels.
6. Produce two channels:
   - Grayscale intensity.
   - Validity mask identifying non-padding pixels.

No independent per-image intensity normalization is applied.

### CNN architecture

The image encoder uses convolutional blocks with channels:

`2 → 16 → 32 → 64 → 128`

The extracted features are passed through adaptive average pooling and a regression head.

The project also implements:
- Image-only CNN.
- Metadata-only CNN.
- Multimodal CNN.

Metadata features include PlantNumber, Genotype, Stage, Side, Start, End, and strip count.

### Training

The initial training configuration uses:

- Optimizer: AdamW
- Learning rate: 0.0003
- Weight decay: 0.0001
- Loss: SmoothL1
- Batch size: 8
- Early stopping for initial experiments.

Grouped cross-validation uses five folds, with FolderName as the grouping variable.

## 4. Current Results

### Grouped cross-validation

Pooled out-of-fold results from the initial CNN experiments:

| Model | RMSE | MAE | R² |
|---|---:|---:|---:|
| Image CNN | 1.4981 | 1.0765 | 0.0342 |
| Metadata CNN | 1.4769 | 1.0371 | 0.0612 |
| Multimodal CNN (SmoothL1) | 1.5240 | 1.0745 | 0.0004 |
| Multimodal CNN (MSE) | 1.4985 | 1.0682 | 0.0336 |
| Multimodal CNN (log-MSE) | 1.5160 | 1.0572 | 0.0109 |

### Baseline comparison

Pooled out-of-fold results:

| Baseline | RMSE | MAE | R² |
|---|---:|---:|---:|
| Image-area Ridge | 1.5010 | 1.0650 | 0.0304 |
| Metadata Ridge | 1.4861 | 1.0791 | 0.0495 |
| Metadata + image-area Ridge | 1.4728 | 1.0650 | 0.0665 |

The metadata + image-area baseline currently has the best pooled RMSE among these experiments. Its advantage over the other models is small.

### Uncertainty estimation

Split conformal prediction was implemented using folder-disjoint training, calibration, and evaluation sets.

The image CNN achieved:

- Evaluation RMSE: 1.4848
- Evaluation MAE: 1.1963
- Evaluation R²: -0.0705
- Observation-level empirical coverage: 92.68%
- Observation-level mean interval width: 4.46

The folder-maximum calibration method achieved 85% whole-folder coverage on 20 evaluation folders, below the 90% target.

These results come from a separate holdout experiment and must not be directly compared with the five-fold cross-validation results.

## 5. Important Findings and Limitations

### Target imbalance

The training target distribution is strongly concentrated around lower root volumes.

| Root-volume range | Observations |
|---|---:|
| 0–1 | 112 |
| 1–2 | 93 |
| 2–3 | 112 |
| 3–4 | 34 |
| 4–5 | 21 |
| Above 5 | 14 |

Only 3.63% of training observations have root volumes above 5.

Both image and metadata CNNs tend to underestimate high root volumes and overestimate very low root volumes.

### Other limitations

- The dataset contains relatively few labelled observations.
- Identical image sequences can occur with different PlantNumber labels and different root-volume targets.
- Model performance varies across held-out folders.
- Current prediction intervals are relatively wide.
- Empirical coverage on a single split does not establish guaranteed future coverage.
- The current GUI uses an experimental checkpoint and is not a validated deployment system.

## 6. Project Structure

```text
Non_Destructive_Root_Volume_Estimation/
│
├── data/
│   ├── train/
│   └── test/
│
├── Models/
│
├── outputs/
│   ├── models/
│   ├── splits/
│   ├── cross_validation/
│   ├── baseline_comparison/
│   └── uncertainty/
│
├── scripts/
│   ├── audit_observation_images.py
│   ├── prepare_data.py
│   ├── train_models.py
│   ├── cross_validate_models.py
│   ├── compare_baselines.py
│   ├── evaluate_simple_area.py
│   ├── evaluate_spatial_area.py
│   └── calibrate_predictions.py
│
├── src/
│   ├── preprocessing.py
│   ├── dataset.py
│   └── models/
│       ├── __init__.py
│       └── cnn_regressor.py
│
├── Train.csv
├── Test.csv
├── Sample_Submission.csv
├── requirements.txt
└── README.md
```

Additional scripts and output folders may be added as the project develops.

## 7. Installation

Python 3.13 is the development environment.

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd Non_Destructive_Root_Volume_Estimation
```

Create a virtual environment:

```bash
uv venv
source .venv/bin/activate
```

Install dependencies:

```bash
uv pip install -r requirements.txt
```

Alternatively, install using pip:

```bash
pip install -r requirements.txt
```

For the desktop GUI on Ubuntu, install Tkinter if it is not already available:

```bash
sudo apt install python3-tk
```

If using a custom Python installation, ensure that its Tkinter support is available to that interpreter.

## 8. Running the Project

Prepare the data:

```bash
uv run python scripts/prepare_data.py
```

Train the initial models:

```bash
uv run python scripts/train_models.py
```

Run grouped cross-validation:

```bash
uv run python scripts/cross_validate_models.py
```

Compare baselines:

```bash
uv run python scripts/compare_baselines.py
```

Run uncertainty calibration:

```bash
uv run python scripts/calibrate_predictions.py
```

Launch the desktop application:

```bash
uv run python scripts/gui.py
```

## 9. Future Roadmap

### Modelling improvements
- [ ] Weighted SmoothL1 regression for underrepresented target ranges.
- [ ] Weighted MSE experiment.
- [ ] Re-evaluate log-target regression.
- [ ] Improve predictions at very low and high root volumes.
- [ ] Investigate folder-specific and genotype-related effects.
- [ ] Reassess image representations and CNN architecture.

### Validation and uncertainty
- [ ] Repeat grouped experiments across multiple random seeds.
- [ ] Evaluate observation-level and whole-folder coverage.
- [ ] Investigate subgroup calibration.
- [ ] Improve prediction interval efficiency.
- [ ] Evaluate calibration stability across folds.

### Final application
- [ ] Train a final model using the selected configuration.
- [ ] Calibrate the final model with a separate calibration set.
- [ ] Add prediction history and CSV export.
- [ ] Improve GUI visualisation and usability.
- [ ] Document inference requirements and model limitations.

## 10. Project Status

The data audit, preprocessing pipeline, CNN architectures, baseline comparisons, grouped cross-validation, and initial uncertainty experiments are implemented.

The project is currently in the model-improvement and application-development stage.

The final objective is a reproducible, non-destructive root-volume estimation tool that reports both a point estimate and a meaningful uncertainty interval.
