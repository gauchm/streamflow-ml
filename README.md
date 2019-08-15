# Machine Learning for Streamflow Prediction

## Repository Organization
```
notebooks/
  initial_experiments/        -- initial data exploration, first models
  linear_models/              -- linear regression, ridge regression models
  xgboos/                     -- XGBoost models
  neural_networks_non_spatial -- neural network models that don't explicitly use gridded data, but run on a flattened grid
  visualization/              -- Visualizations of layers/kernels/...
  evaluation/                 -- Comparisons of physically-based and data-driven models
src/                          -- common source code
data/                         -- used datasets (not in the repository for copyright and size reasons)
  geophysical/                -- geophysical datasets
pickle/
  models/                     -- pickled trained models (not in the repository)
  results/                    -- pickled model results (not in the repository)
figures/                      -- generated figures
requirements.txt              -- python package requirements, to be installed with conda
```

