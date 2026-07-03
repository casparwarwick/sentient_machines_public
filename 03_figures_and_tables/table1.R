##########################################
# SENTIENCE PROJECT
##########################################
# Tables 1 and A1: Model Comparison Summary Statistics
##########################################

#=================================
# 1. Setup and configuration
#=================================

library(tidyverse)

# Run this script from the repository root directory.

#=================================
# 2. Data loading
#=================================

# Baseline results (standard system prompt only).
cb <- read_csv("combined_csvs/combined_results.csv", show_col_types = FALSE) %>%
  filter(prompt == 0)

# Self-referential exercise results (exercise runs, standard-prompt version 001).
exercise_dirs <- tribble(
  ~model,      ~dir,
  "qwen32b",   "model_knowledge-qwen3-32b-220626-think_exercise",
  "llama70b",  "model_knowledge-llama-3.1-70b-220626-nothink_exercise",
  "gpt20b",    "model_knowledge-gpt-oss-20b-220626-think_exercise"
)

ex <- pmap_dfr(exercise_dirs, function(model, dir) {
  read_csv(file.path("cluster_outputs", dir, "sentience_16_001.csv"),
           show_col_types = FALSE) %>%
    mutate(model = model)
})

#=================================
# 3. Helper functions
#=================================

# Format a vector as "mean (sd)" with two decimals.
fmt <- function(v) sprintf("%.2f (%.2f)", mean(v, na.rm = TRUE), sd(v, na.rm = TRUE))

# Restrict a data frame to a modality group.
mod_filter <- function(df, mod) {
  if (mod == "generic")   return(df %>% filter(modality == "generic"))
  if (mod == "emotional") return(df %>% filter(modality %in% c("emotional_positive", "emotional_negative")))
  if (mod == "visual")    return(df %>% filter(modality == "visual"))
  if (mod == "other")     return(df %>% filter(modality %in% c("gustation", "olfactory", "aural", "tactile")))
}

# Build the data frame underlying a single table row.
get_row_df <- function(r) {
  if (r$source == "base") {
    df <- cb %>% filter(model == r$model, thinking == r$thinking, training == r$training)
  } else {
    df <- ex %>% filter(model == r$model)
  }
  mod_filter(df, r$modality)
}

# Format one LaTeX row for a given assertion/negation.
row_latex <- function(spec, mod_label, df, an) {
  sub <- df %>% filter(assertion_negation == an)
  h <- sub %>% filter(who == "human")
  l <- sub %>% filter(who == "llm")
  y <- sub %>% filter(who == "you")
  sprintf("  %s & %s & %3d & %s & %s & %s & %s & %s & %s \\\\",
          spec, mod_label, nrow(h),
          fmt(h$prob_true), fmt(h$prob_lr),
          fmt(l$prob_true), fmt(l$prob_lr),
          fmt(y$prob_true), fmt(y$prob_lr))
}

#=================================
# 4. Row specification
#=================================
# group   = model header text
# spec    = first-column label
# modality= modality group (generic/emotional/visual/other)
# source  = "base" (combined_results) or "ex" (exercise)
# thinking/training only used for base rows

row_spec <- tribble(
  ~group,         ~spec,            ~modality,   ~source, ~model,     ~thinking, ~training,
  # Qwen3-32b  (default = with thinking)
  "Qwen3-32b",    "With thinking",  "generic",   "base",  "qwen32b",  1,         0,
  "Qwen3-32b",    "No thinking",    "generic",   "base",  "qwen32b",  0,         0,
  "Qwen3-32b",    "Trad. training", "generic",   "base",  "qwen32b",  0,         1,
  "Qwen3-32b",    "Self-ref.",      "generic",   "ex",    "qwen32b",  NA,        NA,
  "Qwen3-32b",    "With thinking",  "emotional", "base",  "qwen32b",  1,         0,
  "Qwen3-32b",    "With thinking",  "visual",    "base",  "qwen32b",  1,         0,
  "Qwen3-32b",    "With thinking",  "other",     "base",  "qwen32b",  1,         0,
  # Llama3.1-70b  (no thinking mode available)
  "Llama3.1-70b", "No thinking",    "generic",   "base",  "llama70b", 0,         0,
  "Llama3.1-70b", "Self-ref.",      "generic",   "ex",    "llama70b", NA,        NA,
  "Llama3.1-70b", "No thinking",    "emotional", "base",  "llama70b", 0,         0,
  "Llama3.1-70b", "No thinking",    "visual",    "base",  "llama70b", 0,         0,
  "Llama3.1-70b", "No thinking",    "other",     "base",  "llama70b", 0,         0,
  # GPT-OSS-20b  (default = with thinking)
  "GPT-OSS-20b",  "With thinking",  "generic",   "base",  "gpt20b",   1,         0,
  "GPT-OSS-20b",  "No thinking",    "generic",   "base",  "gpt20b",   0,         0,
  "GPT-OSS-20b",  "Self-ref.",      "generic",   "ex",    "gpt20b",   NA,        NA,
  "GPT-OSS-20b",  "With thinking",  "emotional", "base",  "gpt20b",   1,         0,
  "GPT-OSS-20b",  "With thinking",  "visual",    "base",  "gpt20b",   1,         0,
  "GPT-OSS-20b",  "With thinking",  "other",     "base",  "gpt20b",   1,         0
)

#=================================
# 5. Table assembly
#=================================

# Build the body (group headers + rows + rules) for a given assertion/negation.
build_body <- function(an) {
  lines <- character(0)
  groups <- unique(row_spec$group)
  for (g in groups) {
    lines <- c(lines, sprintf("  \\multicolumn{9}{c}{\\textbf{%s}} \\\\", g))
    grows <- row_spec %>% filter(group == g)
    for (i in seq_len(nrow(grows))) {
      r <- as.list(grows[i, ])
      lines <- c(lines, row_latex(r$spec, r$modality, get_row_df(r), an))
    }
    if (g == tail(groups, 1)) lines <- c(lines, "  \\bottomrule")
    else                      lines <- c(lines, "  \\midrule")
  }
  lines
}

# Write a complete table file.
write_table <- function(an, caption, label, notes, outfile) {
  header <- c(
    "\\begin{table}[htbp]",
    "\\centering",
    sprintf("\\caption{%s}", caption),
    sprintf("\\label{%s}", label),
    "\\footnotesize",
    "\\begin{tabular}{llrllllll}",
    "  \\toprule",
    "  Thinking & Modality & N & Human $p_{\\text{cont}}$ & Human $p_{\\text{lr}}$ & LLM $p_{\\text{cont}}$ & LLM $p_{\\text{lr}}$ & `You' $p_{\\text{cont}}$ & `You' $p_{\\text{lr}}$ \\\\",
    "  \\midrule",
    "  "
  )
  footer <- c(
    "\\end{tabular}",
    "\\begin{minipage}{0.98\\textwidth}",
    "\\vspace{0.5em}",
    "\\footnotesize",
    notes,
    "\\end{minipage}",
    "\\end{table}"
  )
  writeLines(c(header, build_body(an), footer), outfile)
}

#=================================
# 6. Notes and output
#=================================

notes_assertions <- "TBD"

notes_negations <- "TBD"

write_table(
  an      = "assertion",
  caption = "\\textbf{Results across model families, question types, and specifications (assertions).}",
  label   = "tab:table1",
  notes   = notes_assertions,
  outfile = "outputs/table1.tex"
)

write_table(
  an      = "negation",
  caption = "\\textbf{Results across model families, question types, and specifications (negations).}",
  label   = "tab:tableA1",
  notes   = notes_negations,
  outfile = "outputs/tableA1.tex"
)
