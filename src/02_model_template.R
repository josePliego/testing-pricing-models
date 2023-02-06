# 02_model_template.R

library(tidyverse)
library(tidymodels)
library(bonsai)
library(parallel)
library(doFuture)

# all_cores <- parallel::detectCores(logical = FALSE)
registerDoFuture()
cl <- makeCluster(6)
plan(cluster, workers = cl)

dt <- read_rds("data/processed/dt_grouped.rds")

set.seed(42)
splits <- initial_split(dt)
dt_train <- training(splits)
dt_val <- testing(splits)

cv_splits <- vfold_cv(dt_train, v = 10)

param_grid <- grid_latin_hypercube(
  num_leaves(),
  min_n(),
  tree_depth(),
  size = 30
)

lightgbm_model <- boost_tree(
  mode = "regression",
  trees = 1000,
  min_n = tune(),
  tree_depth = tune()
) |>
  set_engine("lightgbm", num_leaves = tune())

lightgbm_recipe <- recipe(loss ~., data = splits) |>
  step_log(loss)

lightgbm_wf <- workflow() |>
  add_recipe(lightgbm_recipe) |>
  add_model(lightgbm_model)

tune_params <- tune_grid(
  lightgbm_wf,
  resamples = cv_splits,
  grid = param_grid,
  control = control_grid(verbose = TRUE, allow_par = TRUE)
)

show_best(tune_params, metric = "rmse", n = 5) |>
  write_rds("output/model_template_results.rds")

beepr::beep(sound = 8)

