library(tidyverse)

dt_performance <- read_csv("output/nested_vs_not.csv")

performance_summary <- dt_performance |>
  group_by(type) |>
  summarise(
    across(performance, list("mean" = mean, "sd" = sd))
    ) |>
  mutate(across(where(is.numeric), ~num(.x, digits = 5)))

dt_importance <- read_csv("output/nested_importance.csv")

model_num <- dt_importance |>
  select(model) |>
  distinct() |>
  arrange(model) |>
  mutate(model_num = 1:n())

dt_rank <- dt_importance |>
  arrange(model, -importance) |>
  group_by(model) |>
  mutate(rank = 1:n()) |>
  ungroup() |>
  left_join(model_num, by = "model")

dt_avg_rank <- dt_rank |>
  group_by(var) |>
  summarise(across(rank, list("mean" = mean, "sd" = sd)), .groups = "drop") |>
  arrange(rank_mean)

highlight_vars <- dt_avg_rank |>
  slice(c(1, 3, 128, 72, 82, 130, 66)) |>
  pull(var)

imp_plot <- dt_rank |>
  mutate(color = if_else(var %in% highlight_vars, TRUE, FALSE)) |>
  ggplot(aes(x = model_num, y = rank)) +
  geom_line(aes(group = var, alpha = color)) +
  scale_x_continuous(breaks = 1:9) +
  labs(
    y = "Importance Rank",
    x = "Model Index",
    title = "Cross validation can be used to better estimate variable importance",
    subtitle = "Importance shows more variability around the middle of the rankings"
    ) +
  theme_bw() +
  theme(legend.position = "none")

png(
  "output/graphs/importance.png",
  width = 25,
  height = 15,
  units = "cm",
  res = 300
  )
print(imp_plot)
dev.off()
