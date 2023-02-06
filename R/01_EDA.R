# 01_EDA.R

library(tidyverse)

dt <- read_csv("data/raw/train.csv")

skimr::skim(dt)


# 1. Response -------------------------------------------------------------

dt |>
  ggplot(aes(x = loss)) +
  geom_density(fill = "steelblue1") +
  theme_bw()

dt |>
  ggplot(aes(x = log(loss))) +
  geom_density(fill = "steelblue1") +
  theme_bw()


# 2. Categorical Predictors -----------------------------------------------

dt |>
  select(starts_with("cat")) |>
  summarise(across(everything(), n_distinct)) |>
  pivot_longer(cols = everything()) |>
  count(value)

dt |>
  select(starts_with("cat")) |>
  summarise(across(everything(), n_distinct)) |>
  pivot_longer(cols = everything()) |>
  filter(value > 20) |>
  arrange(value)

cut_levels <- function(.dt, cat_col) {
  cum_prop <- .dt |>
    count({{cat_col}}) |>
    arrange(-n) |>
    mutate(across(n, ~cumsum(.x)/sum(.x)))

  first_group <- which(1 - cum_prop$n < cum_prop$n[[1]])[[2]]

  group_levels <- cum_prop |>
    slice(first_group:n()) |>
    pull({{cat_col}})

  .dt |>
    mutate(across({{cat_col}}, ~if_else(.x %in% group_levels, "Other", .x)))

}

dt_group <- dt |>
  cut_levels(cat115) |>
  cut_levels(cat112) |>
  cut_levels(cat113) |>
  cut_levels(cat109) |>
  cut_levels(cat110) |>
  cut_levels(cat116)

dt_group |>
  select(cat115, cat112, cat113, cat109, cat110, cat116) |>
  pivot_longer(cols = everything()) |>
  ggplot(aes(x = value)) +
  geom_bar() +
  facet_wrap(~name, scales = "free")

# dt_group is modeling data set so far


# 3. Numerical Predictors -------------------------------------------------

dt_group |>
  select(starts_with("cont")) |>
  pivot_longer(cols = everything()) |>
  ggplot(aes(x = value)) +
  geom_density() +
  facet_wrap(~name, scales = "free")


# 4. Write data -----------------------------------------------------------

cat("Writing file: data/processed/dt_grouped.rds\n")
write_rds(dt_group, "data/processed/dt_grouped.rds", compress = "gz")
