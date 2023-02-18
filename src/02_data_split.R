library("tidyverse")
library("rsample")
# library("fastDummies")

dt <- read_rds("data/processed/dt_grouped.rds")

set.seed(42)
splits <- dt %>%
  # dummy_cols(remove_selected_columns = TRUE) %>%
  group_initial_split(prop = 0.8, group = "id")

training(splits) %>%
  write_csv("data/processed/dt_train.csv")

testing(splits) %>%
  write_csv("data/processed/dt_val.csv")

testing(splits) %>%
  mutate(pred = mean(training(splits)$loss)) %>%
  mutate(sq_dif = (loss - pred)^2) %>%
  summarise(across(sq_dif, ~sqrt(mean(.x)))) %>%
  pull(sq_dif)

# Initial RMSE: 0.809581

for (i in colnames(training(splits))) {
  if (str_detect(i, 'cat')) {
    num = n_distinct(training(splits)[[i]])
    if (num != n_distinct(testing(splits)[[i]])) {
      print(num)
      print(i)
    }
  }
}