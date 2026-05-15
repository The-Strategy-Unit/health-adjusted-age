# README
# Custom ggplot2 theme

# packages
library("ggplot2")


# theme_hse ----
theme_haa <- function(
  base_size = 10,
  base_family = "Fira Sans tnum"
) {

  half_line <- base_size / 2

  theme(
    # inherited
    line = element_line(),
    rect = element_rect(),
    text = element_text(
      family = base_family,
      face = "plain",
      colour = "#686f73",
      size = base_size,
      lineheight = .9,
      hjust = .5,
      vjust = .5,
      angle = 0,
      margin = margin(),
      debug = FALSE
    ),

    # axes
    # axis.line = element_line(linewidth = .2, color = "#d3d3d3"),
    axis.line = element_blank(),
    axis.text = element_text(
      size = base_size,
      color = "#686f73",
      hjust = .5
    ),
    axis.ticks.x = element_line(linewidth = .4, color = "#686f73"),
    axis.ticks.y = element_blank(),
    axis.ticks.length.x = unit(2, "mm"),
    axis.title = element_text(),

    # legend
    legend.background = element_rect(color = NA, fill = NA),
    legend.spacing = unit(4, "mm"),
    legend.margin = margin(2, 2, 2, 2, "mm"),
    legend.key = element_rect(color = NA, fill = NA),
    legend.text = element_text(size = base_size),
    legend.title = element_text(),
    legend.position = "right",
    legend.direction = NULL,
    legend.justification = "centre",
    legend.box = NULL,

    # panel
    panel.background = element_rect(color = NA, fill = NA),
    panel.border = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_line(linewidth = .2, color = "#d3d3d3"),
    panel.grid.minor = element_blank(),
    panel.spacing = unit(2, "mm"),
    panel.ontop = FALSE,

    # strip
    strip.background = element_rect(
      color = NA,
      fill = "#e7e7e7"
    ),
    strip.text = element_text(
      family = "Fira Sans",
      size = base_size,
      colour = "#1d1d1d",
      hjust = 0,
      margin = margin(2, 2, 2, 2, "mm")
    ),
    strip.placement = "inside",
    strip.switch.pad.grid = unit(2, "mm"),
    strip.switch.pad.wrap = unit(2, "mm"),

    # plot
    plot.background = element_blank(),
    plot.title.position = "plot",
    plot.caption.position = "plot",
    plot.title = element_text(
      family = "Fira Sans Medium",
      size = base_size * 1.2,
      color = "#2c2825",
      lineheight = 1.2,
      hjust = 0,
      vjust = 1,
      margin = margin(b = 4 * 2)
    ),
    plot.subtitle = element_text(
      family = "Fira Sans",
      size = base_size,
      color = "#686f73",
      lineheight = 1.2,
      hjust = 0,
      vjust = 1,
      margin = margin(b = 4 * 2)
    ),
    plot.caption = element_text(
      family = "Fira Sans",
      size = base_size,
      color = "#686f73",
      hjust = 0,
      margin = margin(t = 2, unit = "mm")
    ),
    plot.margin = margin(half_line, half_line, half_line, half_line),
    complete = TRUE
  )
}
