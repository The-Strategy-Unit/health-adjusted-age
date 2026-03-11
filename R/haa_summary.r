# README
# Plot health adjusted age

# dependencies
library("languageserver")
library("httpgd")
library("arrow")
library("dplyr")
library("ggplot2")
library("here")


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

labeller <- ggplot2::labeller(sex = c("f" = "F", "m" = "M"))

ggplot2::ggplot(
  data = haa_df,
  mapping = ggplot2::aes(x = age, y = adj)
) +
  ggplot2::geom_boxplot(mapping = ggplot2::aes(group = age), outliers = FALSE) +
  ggplot2::geom_line(
    data = mn_adj,
    mapping = ggplot2::aes(x = age, y = mn_adj),
    color = "gold",
    linewidth = 1
  ) +
  ggplot2::annotate(
    geom = "text", x = 75, y = 2,
    label = "haa > chronol. age",
    hjust = 0, vjust = 1,
    size = 4,
    color = "red"
  ) +
  ggplot2::annotate(
    geom = "text", x = 75, y = -5,
    label = "haa < chronol. age",
    hjust = 0, vjust = 1,
    size = 4,
    color = "red"
  ) +
  ggplot2::facet_wrap(vars(sex), labeller = labeller) +
  ggplot2::geom_hline(yintercept = 0, color = "red") +
  ggplot2::scale_x_continuous(name = NULL) +
  ggplot2::scale_y_continuous(
    name = NULL,
    breaks = seq(-8, 4, by = 1)
  ) +
  ggplot2::labs(
    title = "Health adjusted age by chronological age and sex"
  ) +
  ggplot2::theme_light()
