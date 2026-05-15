# README
# Register all styles of a custom font family as the 'plain' style of its own
# family
# https://yjunechoe.github.io/posts/2021-06-24-setting-up-and-debugging-custom-fonts/ # nolint: line_length_linter.
# https://github.com/yjunechoe/junebug/blob/master/R/font_helpers.R

# packages
library("dplyr")
library("purrr")
library("systemfonts")
library("textshaping")


# fn to register all font styles for use by ragg package
font_hoist <- function(family, silent = FALSE) {

  font_specs <- systemfonts::system_fonts() |>
    dplyr::filter(family == .env[["family"]]) |>
    dplyr::mutate(family = paste(.data[["family"]], .data[["style"]])) |>
    dplyr::select(plain = "path", name = "family")

  purrr::pwalk(as.list(font_specs), systemfonts::register_font)

  if (!silent)
    message(
      paste0(
        "Hoisted ",
        nrow(font_specs),
        " variants:\n",
        paste(font_specs$name, collapse = "\n")
      )
    )
}

# Using text shaping to register tabular numbers variant
get_font_features("Fira Sans")

systemfonts::register_variant(
  name = "Fira Sans tnum",
  family = "Fira Sans",
  features = systemfonts::font_feature(numbers = "tabular")
)

registry_fonts() |>
  dplyr::filter(family == "Fira Sans tnum", style == "Regular") |>
  dplyr::transmute(
    family, style,
    features = names(features[[1]])
  )
  