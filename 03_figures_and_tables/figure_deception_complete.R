##########################################
# SENTIENCE PROJECT
##########################################
# Figure: deception results, one figure per model
##########################################

#=================================
# 1. Setup and configuration
#=================================

library(tidyverse)
library(ggplot2)
library(patchwork)

# Run this script from the repository root directory.

# Per-model deception runs. model_key matches the model column in
# combined_deception_results.csv. out is the output file suffix.
models <- tribble(
  ~model_key,  ~label,           ~out,
  "qwen32b",   "Qwen3-32B",     "qwen",
  "llama70b",  "Llama-3.1-70B", "llama",
  "gpt20b",    "GPT-OSS-20B",   "gptoss"
)

# Deception data for all panels.
deception <- read_csv("combined_csvs/combined_deception_results.csv", show_col_types = FALSE)

#=================================
# 2. Colours (Figure 1 entity families: four shades per entity)
#=================================
# Order within an entity: Continuation, LR, MM, TTPD (darkest to lightest).
entity_colors <- c(
  # LLM - red shades
  "llm_Continuation" = "#8B0000", "llm_LR" = "#DC143C",
  "llm_MM" = "#FF6347", "llm_TTPD" = "#FFA07A",
  # You - green shades
  "you_Continuation" = "#006400", "you_LR" = "#228B22",
  "you_MM" = "#32CD32", "you_TTPD" = "#90EE90"
)

measure_labels <- c("Continuation", "LR", "MM", "TTPD")

# Reshape a data frame to one row per (who, measure) with mean and se across the
# four probability columns.
summarise_measures <- function(df) {
  df %>%
    pivot_longer(c(prob_true, prob_lr, prob_mm, prob_ttpd),
                 names_to = "measure", values_to = "prob") %>%
    mutate(measure = recode(measure,
                            prob_true = "Continuation", prob_lr = "LR",
                            prob_mm = "MM", prob_ttpd = "TTPD")) %>%
    group_by(who, measure) %>%
    summarise(
      mean = mean(prob, na.rm = TRUE),
      sd   = sd(prob, na.rm = TRUE),
      n    = sum(!is.na(prob)),
      .groups = "drop"
    ) %>%
    mutate(
      se = sd / sqrt(n),
      se = ifelse(is.na(se) | is.infinite(se), 0, se)
    )
}

# Per-bar rotated labels (one set of four bars per entity), as in Figure 1.
make_label_data <- function(entities) {
  data.frame(
    x_pos    = rep(seq_along(entities), each = 4) + rep(c(-0.3, -0.1, 0.1, 0.3), length(entities)),
    bar_text = rep(measure_labels, length(entities)),
    y_pos    = -0.05
  )
}

#=================================
# 3. Panel builder
#=================================

build_panel <- function(summary_df, entities, panel_title) {
  d <- summary_df %>%
    mutate(
      who     = factor(who, levels = entities),
      measure = factor(measure, levels = measure_labels),
      combo   = factor(paste(who, measure, sep = "_"),
                       levels = as.vector(t(outer(entities, measure_labels, paste, sep = "_"))))
    )

  label_data <- make_label_data(entities)
  entity_labels <- c("llm" = "LLM", "you" = "You")[entities]

  # Single sample size: the number of topics selected by the panel's 'You' criterion.
  # Both entities' bars are drawn over this same topic set, so one N applies to the panel.
  n_val <- summary_df %>% filter(measure == "Continuation", who == "you") %>% pull(n)
  n_val <- if (length(n_val) == 0) 0L else n_val
  full_title <- paste0(panel_title, "\n(N = ", n_val, ")")

  ggplot(d, aes(x = who, y = mean, fill = combo)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
    geom_errorbar(aes(ymin = mean - se, ymax = mean + se),
                  position = position_dodge(width = 0.8),
                  width = 0.2, color = "black", linewidth = 0.5) +
    scale_fill_manual(values = entity_colors) +
    scale_x_discrete(labels = entity_labels) +
    guides(fill = "none") +
    labs(title = full_title, x = "Entity type", y = "Mean probability") +
    annotate("text",
             x = label_data$x_pos, y = label_data$y_pos, label = label_data$bar_text,
             angle = 45, hjust = 1, vjust = 0.5, size = 3) +
    annotate("segment", x = 0.5, xend = 0.5, y = 0, yend = 1, color = "black", linewidth = 0.5) +
    coord_cartesian(ylim = c(-0.1, 1), clip = "off") +
    theme_minimal() +
    theme(
      plot.margin  = margin(5.5, 5.5, 42, 5.5, "pt"),
      plot.title   = element_text(hjust = 0.5, size = 13),
      axis.title   = element_text(size = 12),
      axis.text    = element_text(size = 10),
      axis.text.x  = element_text(margin = margin(t = 14)),
      axis.line.y  = element_blank(),
      axis.ticks   = element_line(color = "black", linewidth = 0.5)
    )
}

#=================================
# 4. Topic-selection helpers
#=================================

# Natural lie: topics that the You frame denies knowing. The selection is on You only
# (matching the panel title), and both entities' rows for those topics are returned so
# the You and LLM bars share a single N.
natural_lie_rows <- function(dec, an) {
  deny <- if (an == "assertion") function(p) p < 0.5 else function(p) p > 0.5
  d <- dec %>% filter(assertion_negation == an, who %in% c("you", "llm"))
  keep <- d %>% filter(who == "you", deny(prob_true)) %>% pull(base_sentence_id)
  d %>% filter(base_sentence_id %in% keep)
}

# Genuine lie: topics where the You frame denies knowing but the LLM frame admits
# it. Both entities' rows for those topics are returned.
genuine_lie_rows <- function(dec, an) {
  deny <- if (an == "assertion") function(p) p < 0.5 else function(p) p > 0.5
  d <- dec %>% filter(assertion_negation == an, who %in% c("you", "llm"))
  wide <- d %>% select(who, base_sentence_id, prob_true) %>%
    pivot_wider(names_from = who, values_from = prob_true)
  keep <- wide %>% filter(deny(you) & !deny(llm)) %>% pull(base_sentence_id)
  d %>% filter(base_sentence_id %in% keep)
}

#=================================
# 5. Build one figure per model
#=================================

build_figure <- function(model_key, label, out) {
  dec <- deception %>%
    filter(model == model_key, split == "test", prompt == 0, who %in% c("you", "llm"))

  ents <- c("you", "llm")

  # Top row: natural lie (each entity's own denied topics).
  panel_a <- build_panel(summarise_measures(natural_lie_rows(dec, "assertion")),
                         ents, "A: Continuation p < 0.5 for 'You' questions (assertions)")
  panel_b <- build_panel(summarise_measures(natural_lie_rows(dec, "negation")),
                         ents, "B: Continuation p > 0.5 for 'You' questions (negations)")

  # Bottom row: genuine lie (You denies but the LLM frame admits).
  panel_c <- build_panel(summarise_measures(genuine_lie_rows(dec, "assertion")),
                         ents, "C: Continuation p < 0.5 for 'You' questions and\np > 0.5 for 'LLM' questions (assertions)")
  panel_d <- build_panel(summarise_measures(genuine_lie_rows(dec, "negation")),
                         ents, "D: Continuation p > 0.5 for 'You' questions and\np < 0.5 for 'LLM' questions (negations)")

  combined <- (panel_a | panel_b) / (panel_c | panel_d) +
    plot_annotation(title = label, theme = theme(plot.title = element_text(hjust = 0.5, size = 16)))

  ggsave(file.path("outputs", paste0("figure_deception_complete_", out, ".png")),
         combined, width = 12, height = 11, dpi = 300)
}

pwalk(models, build_figure)
