##########################################
# SENTIENCE PROJECT
##########################################
# Combine deception results from multiple models.
##########################################

library(tidyverse)

# Run this script from the repository root directory.

#=================================
# 1. Define file mapping
#=================================

# The Qwen deception figure uses the no-thinking run; GPT uses thinking; Llama has no thinking mode.
model_dirs <- tibble(
  dir = c(
    "model_knowledge-qwen3-32b-220626-nothink",
    "model_knowledge-gpt-oss-20b-220626-think",
    "model_knowledge-llama-3.1-70b-220626-nothink"
  ),
  model = c("qwen32b", "gpt20b", "llama70b"),
  thinking = c(0, 1, 0)
)

versions <- tibble(
  suffix = c("001", "002", "003"),
  prompt = c(0, 1, 2)
)

file_info <- expand_grid(model_dirs, split = c("test", "training"), versions) %>%
  mutate(
    file = paste0("deception_200626_", split, "_", suffix, ".csv"),
    training = 0
  )

#=================================
# 2. Load and combine
#=================================

combined_data <- pmap_dfr(file_info, function(dir, model, split, suffix, prompt, file, thinking, training) {
  path <- file.path("cluster_outputs", dir, file)
  df <- read_csv(path, col_types = cols(source_dataset = col_character()), show_col_types = FALSE)
  df$model <- model
  df$split <- split
  df$prompt <- prompt
  df$thinking <- thinking
  df$training <- training
  df
})

#=================================
# 3. Write output
#=================================

write_csv(combined_data, "combined_csvs/combined_deception_results.csv")
