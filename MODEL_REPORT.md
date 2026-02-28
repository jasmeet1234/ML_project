# Land-cover modeling decision (single-model approach)

## What I implemented
- **One model only** (as requested): a **multi-output XGBoost regressor** that predicts 2021 land-cover composition for each grid cell from 2020 satellite-derived features + 2020 composition.
- Targets predicted per cell:
  - `comp_built_up_2021`
  - `comp_water_2021`
  - `comp_vegetation_2021`
  - `comp_other_2021`
- Land-cover change is then derived as:
  - `delta_class = class_2021_pred - class_2020`

## Why this model is suitable
- It handles nonlinear relationships between spectral indices and land-cover proportions.
- It works well on tabular engineered features (fits assignment constraints).
- With median imputation + tree boosting, it is robust to the missing-value pattern in your data.
- Multi-output wrapper allows training one unified modeling pipeline for all four classes.

## Data reality checks from the teammate dataset
- Rows: 42,864 for each year (2020 and 2021).
- About **4.99%** of rows have all engineered spectral features missing.
- Label vectors are valid proportions and sum to 1 for every row.

## Evaluation strategy
- Used a deterministic **spatial-like holdout** based on hashed `cell_id` (80% train / 20% test).
- Evaluated both:
  1. **Composition quality** (MAE, RMSE, R²)
  2. **Change quality** on derived deltas (MAE, RMSE, R²)
- Added a change-specific metric:
  - **False change rate** with threshold `|delta| >= 0.05` on any class.

## How to run
```bash
pip install -r requirements.txt
python train_model.py
```

Artifacts written to `artifacts/`:
- `landcover_model.joblib` (trained pipeline)
- `metrics.json` (full metrics summary)

## Recommendation
This teammate dataset is usable and already aligned to the assignment framing (features + composition labels across years). You can proceed with this as your core dataset, and later optionally add external auxiliary features (e.g., OSM road/building density) if you want a performance boost.
