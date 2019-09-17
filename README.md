# Machine Learning for Streamflow Prediction

## Repository Organization
```
notebooks/
  initial_experiments/        -- initial data exploration, first models
  linear_models/              -- linear regression, ridge regression models
  xgboost/                    -- XGBoost models
  neural_networks_non-spatial -- neural network models that don't explicitly use gridded data, but run on a flattened grid
  neural_networks_convLSTM    -- convolutional LSTM network that runs on gridded data
  neural_networks_graph       -- spatio-temporal graph-convolutional network that runs on graph of subbasin-relationships
  visualization/              -- Visualizations of layers/kernels/...
  evaluation/                 -- Comparisons of physically-based and data-driven models
src/                          -- common source code
  conv_lstm.py                -- code for convolutional LSTM models
  datasets.py                 -- dataset classes to feed data into neural networks
  evaluate.py                 -- code to evaluate prediction performance
  load_data.py                -- code to load and save data from/to disk
  stgcn.py                    -- code for spatio-temporal graph-convolutional networks
  utils.py                    -- helper functions
  visualize.py                -- visualization code
data/                         -- used datasets (not in the repository for copyright and size reasons)
  geophysical/                -- geophysical datasets
pickle/
  models/                     -- pickled trained models (not in the repository)
  results/                    -- pickled model results (not in the repository)
figures/                      -- generated figures
requirements.txt              -- python package requirements, to be installed with conda
```

