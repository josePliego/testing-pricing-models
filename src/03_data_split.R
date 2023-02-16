library("tidyverse")
library("rsample")

dt <- read_rds("data/processed/dt_grouped.rds")

set.seed(42)
splits <- group_initial_split(dt, prop = 0.8, group = "id")

training(splits) %>%
  write_csv("data/processed/dt_train.csv")

testing(splits) %>%
  write_csv("data/processed/dt_val.csv")