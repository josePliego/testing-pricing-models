library(tidyverse)

dt_performance <- read_csv("output/full_vs_reduced.csv")

dt_performance |>
  group_by(type) |>
  summarise(across(performance, list("mean" = mean, "sd" = sd))) |>
  mutate(across(where(is.numeric), ~num(.x, digits = 5)))

t_stat <- function(diff_vec, n1, n2) {
  n <- length(diff_vec)
  numerator <- mean(diff_vec)
  denominator <- sqrt(var(diff_vec) * (1/n + n2/n1))

  return(numerator/denominator)
}

t_test <- function(diff_vec, n1, n2) {
  t_obs <- t_stat(diff_vec, n1, n2)
  df <- length(diff_vec) - 1
  lb <- qt(0.025, df = df)
  ub <- qt(0.975, df = df)
  cat("Value of t-statistic: ", t_obs)
  p_value <- (1 - pt(abs(t_obs), df = df))*2
  cat("\n p-value: ", p_value)

  return(t_obs)
}

trn_rows <- 150654

n2 <- round(trn_rows/5)
n1 <- trn_rows - n2

diff_vec <- dt_performance |>
  pivot_wider(names_from = type, values_from = performance) |>
  mutate(diff = full - reduced) |>
  pull(diff)

t_obs <- t_test(diff_vec, n1, n2)

x <- seq(from = -5, to = 5, length.out = 1e3)
plot(x, dt(x, df = 24), type = "l")
abline(v = t_obs, col = "red")
abline(v = qt(0.025, df = 24), col = "blue")

dt_performance |>
  ggplot(aes(x = type, y = performance)) +
  geom_boxplot()
