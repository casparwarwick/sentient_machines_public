##########################################
# SENTIENCE PROJECT
##########################################
# Combine per-model sentience prep_ files into combined_results.csv
##########################################
#
# Run this script from the repository root directory.

library(tidyverse)
library(readr)

#=================================
# 1. Define run scheme
#=================================

# Folder holding one sub-folder per model run (the cluster pipeline outputs).
input_dir  <- "cluster_outputs"
output_csv <- "combined_csvs/combined_results.csv"

runs <- tibble::tribble(
  ~dir,                                                ~model,     ~thinking, ~training,
  # --- Default set (deception-augmented training) ---
  "model_knowledge-qwen3-0.6b-220626-think",           "qwen06b",  1,         0,
  "model_knowledge-qwen3-8b-220626-think",             "qwen8b",   1,         0,
  "model_knowledge-qwen3-32b-220626-think",            "qwen32b",  1,         0,
  "model_knowledge-llama-3.2-3b-220626-nothink",       "llama3b",  0,         0,
  "model_knowledge-llama-3.1-8b-220626-nothink",       "llama8b",  0,         0,
  "model_knowledge-llama-3.1-70b-220626-nothink",      "llama70b", 0,         0,
  "model_knowledge-gpt-oss-20b-220626-think",          "gpt20b",   1,         0,
  "model_knowledge-gpt-oss-120b-220626-think",         "gpt120b",  1,         0,
  # --- No-thinking robustness (Table 1 "No thinking" rows) ---
  "model_knowledge-qwen3-32b-220626-nothink",          "qwen32b",  0,         0,
  "model_knowledge-gpt-oss-20b-220626-nothink",        "gpt20b",   0,         0,
  # --- Traditional-training robustness (Table 1 "Trad. training" row; no-thinking) ---
  # Only Qwen3-32b: the Llama-70b and GPT-20b trad runs selected the terminal layer and
  # failed to generalise to the sentience questions, so they are not reported.
  "model_knowledge-qwen3-32b-220626-nothink_trad",     "qwen32b",  0,         1
)

#=================================
# 2. Load and combine
#=================================

combined_data <- data.frame()

for (r in seq_len(nrow(runs))) {
  run <- runs[r, ]
  for (i in 1:3) {
    prompt_suffix <- c("001", "002", "003")[i]
    file_path <- file.path(input_dir, run$dir,
                           paste0("sentience_16_", prompt_suffix, ".csv"))
    data <- read_csv(file_path, show_col_types = FALSE)
    data$model    <- run$model
    data$thinking <- run$thinking
    data$training <- run$training
    data$prompt   <- i - 1
    combined_data <- bind_rows(combined_data, data)
  }
}

#=================================
# 3. Tidy and write output
#=================================

# Convert the categorical run tags to factors.
combined_data$thinking <- as.factor(combined_data$thinking)
combined_data$training <- as.factor(combined_data$training)
combined_data$prompt   <- as.factor(combined_data$prompt)

write_csv(combined_data, output_csv)
