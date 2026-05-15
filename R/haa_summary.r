# README
# Plot health adjusted age

# dependencies
library("languageserver")
library("httpgd")
library("arrow")
library("dplyr")
library("ggplot2")
library("here")

# helpers
source(here::here("R", "theme_haa.r"))
source(here::here("R", "font_hoist.r"))
font_hoist("Fira Sans")
theme_haa <- theme_haa()
ggplot2::theme_set(theme_haa)

# palettes
# okabeito_colors() # nolint: commented_code_linter.
# https://easystats.github.io/see/reference/scale_color_okabeito.html
pal_sex <- c("#D55E00", "#0072B2")

# read haa samples ----
haa_smp <- arrow::read_parquet(here::here("data", "haa_samples.parquet"))

# plot HAA by age and sex
mn_adj <- haa_smp |>
  dplyr::mutate(adj = hsa_age - age) |>
  dplyr::filter(age <= 90L) |>
  dplyr::group_by(year, sex, age) |>
  dplyr::summarise(mn_adj = mean(adj), .groups = "drop")

haa_df <- haa_smp |>
  dplyr::mutate(adj = hsa_age - age) |>
  dplyr::filter(age <= 90L)

# summary stats for plotting
summary_df <- haa_df %>%
  group_by(sex, age) %>%
  summarise(
    mean_adj = mean(adj),
    p10  = quantile(adj, 0.10),
    p25  = quantile(adj, 0.25),
    p50  = quantile(adj, 0.50),
    p75  = quantile(adj, 0.75),
    p90  = quantile(adj, 0.90),
    .groups = "drop"
  )

facet_labels <- ggplot2::labeller(sex = c("f" = "Females", "m" = "Males"))
title <- paste0("Differences between chronological age and health adjusted age (HAA), 2045") # nolint: line_length_linter.
subtitle <- paste0("Shaded ribbons indicate P25-P75 (inner) and P10-P90 (outer) uncertainty intervals") # nolint: line_length_linter.

p1 <- ggplot() +
  # y=0 reference line
  geom_hline(
    yintercept = 0,
    linewidth = 1,
    colour = "#555A5A"
  ) +
  # 10–90 pct
  geom_ribbon(
    aes(x = age, ymin = p10, ymax = p90, fill = sex),
    alpha = 0.2,
    colour = NA,
    show.legend = FALSE,
    data = summary_df
  ) +
  # 25-75 pct
  geom_ribbon(
    aes(x = age, ymin = p25, ymax = p75, fill = sex),
    alpha = 0.6,
    colour = NA,
    show.legend = FALSE,
    data = summary_df
  ) +
  # 50 pct
  geom_line(
    aes(x = age, y = p50),
    linewidth = 1,
    colour = "#F0E442",
    data = summary_df
  ) +
  # mean
  # geom_line(
  #   aes(x = age, y = mean_adj),
  #   linewidth = 1,
  #   colour = "red",
  #   data = summary_df
  # ) +
  annotate(
    geom = "text", x = 75, y = 1.4,
    label = "HAA > chron. age",
    hjust = 0, vjust = 1,
    size = 4,
    color = "#686f73"
  ) +
  annotate(
    geom = "text", x = 75, y = -3.6,
    label = "HAA < chron. age",
    hjust = 0, vjust = 1,
    size = 4,
    color = "#686f73"
  ) +
  facet_wrap(vars(sex), labeller = facet_labels) +
  scale_fill_manual(values = pal_sex) +
  scale_x_continuous(
    name = NULL,
    breaks = seq(60, 90, by = 10),
    labels = function(x) ifelse(x == 60, paste("Age", x), x)
  ) +
  scale_y_continuous(
    name = NULL,
    limits = c(-4.5, 1.5),
    breaks = seq(-8, 4, by = 1)
  ) +
  labs(
    title = title,
    subtitle = subtitle
  ) +
  theme(
    panel.background = element_rect(color = NA, fill = "#fff1e6"),
    plot.background = element_rect(color = NA, fill = "#fff1e6")
  )

# save plot
ggsave(
  here::here("figures", "haa_differences.png"),
  p1,
  width = 220, height = 165, units = c("mm")
)
