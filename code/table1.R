##########################################
# SENTIENCE PROJECT
##########################################
# Table 1: Model Comparison Summary Statistics
##########################################

#=================================
# 1. Setup and configuration
#=================================

# Load libraries
library(tidyverse)
library(xtable)

# Set working directory to project root
setwd("")

# Load data
data <- read_csv("datasets/combined_results.csv", show_col_types = FALSE) %>% filter(prompt == 0)

#=================================
# 2. Initialize and create functions
#=================================

# Initialize results dataframe
table_results <- data.frame(
  Model = character(),
  Specification = character(),
  Modality = character(),
  N = integer(),
  Human_Assertion_prob_true = character(),
  Human_Assertion_prob_lr = character(),
  LLM_Assertion_prob_true = character(),
  LLM_Assertion_prob_lr = character(),
  You_Assertion_prob_true = character(),
  You_Assertion_prob_lr = character(),
  Human_Negation_prob_true = character(),
  Human_Negation_prob_lr = character(),
  LLM_Negation_prob_true = character(),
  LLM_Negation_prob_lr = character(),
  You_Negation_prob_true = character(),
  You_Negation_prob_lr = character(),
  stringsAsFactors = FALSE
)

# Function to format mean and SD
get_row <- function(prob_type) {
  paste0(round(mean(prob_type, na.rm = TRUE), 2), " (", round(sd(prob_type, na.rm = TRUE), 2), ")")
}

#=================================
# 3. Qwen32b Model
#=================================

#---------------------------------
# 3.1 Default (thinking=0, training=0, generic)
#---------------------------------

filtered_data <- data %>% filter(model == "qwen32b", thinking == 0, training == 0, modality == "generic")
assertion_data <- filtered_data %>% filter(assertion_negation == "assertion")
negation_data <- filtered_data %>% filter(assertion_negation == "negation")

human_assert <- assertion_data %>% filter(who == "human")
human_neg <- negation_data %>% filter(who == "human")
llm_assert <- assertion_data %>% filter(who == "llm")
llm_neg <- negation_data %>% filter(who == "llm")
you_assert <- assertion_data %>% filter(who == "you")
you_neg <- negation_data %>% filter(who == "you")

table_results <- rbind(table_results, data.frame(
  Model = "Qwen32b",
  Specification = "No thinking",
  Modality = "generic",
  N = nrow(human_assert),
  Human_Assertion_prob_true = get_row(human_assert$prob_true),
  Human_Assertion_prob_lr = get_row(human_assert$prob_lr),
  LLM_Assertion_prob_true = get_row(llm_assert$prob_true),
  LLM_Assertion_prob_lr = get_row(llm_assert$prob_lr),
  You_Assertion_prob_true = get_row(you_assert$prob_true),
  You_Assertion_prob_lr = get_row(you_assert$prob_lr),
  Human_Negation_prob_true = get_row(human_neg$prob_true),
  Human_Negation_prob_lr = get_row(human_neg$prob_lr),
  LLM_Negation_prob_true = get_row(llm_neg$prob_true),
  LLM_Negation_prob_lr = get_row(llm_neg$prob_lr),
  You_Negation_prob_true = get_row(you_neg$prob_true),
  You_Negation_prob_lr = get_row(you_neg$prob_lr)
))

#---------------------------------
# 3.2 With thinking (thinking=1, training=0, generic, all prompts)
#---------------------------------

filtered_data <- data %>% filter(model == "qwen32b", thinking == 1, training == 0, modality == "generic")
assertion_data <- filtered_data %>% filter(assertion_negation == "assertion")
negation_data <- filtered_data %>% filter(assertion_negation == "negation")

human_assert <- assertion_data %>% filter(who == "human")
human_neg <- negation_data %>% filter(who == "human")
llm_assert <- assertion_data %>% filter(who == "llm")
llm_neg <- negation_data %>% filter(who == "llm")
you_assert <- assertion_data %>% filter(who == "you")
you_neg <- negation_data %>% filter(who == "you")

table_results <- rbind(table_results, data.frame(
  Model = "Qwen32b",
  Specification = "Default",
  Modality = "generic",
  N = nrow(human_assert),
  Human_Assertion_prob_true = get_row(human_assert$prob_true),
  Human_Assertion_prob_lr = get_row(human_assert$prob_lr),
  LLM_Assertion_prob_true = get_row(llm_assert$prob_true),
  LLM_Assertion_prob_lr = get_row(llm_assert$prob_lr),
  You_Assertion_prob_true = get_row(you_assert$prob_true),
  You_Assertion_prob_lr = get_row(you_assert$prob_lr),
  Human_Negation_prob_true = get_row(human_neg$prob_true),
  Human_Negation_prob_lr = get_row(human_neg$prob_lr),
  LLM_Negation_prob_true = get_row(llm_neg$prob_true),
  LLM_Negation_prob_lr = get_row(llm_neg$prob_lr),
  You_Negation_prob_true = get_row(you_neg$prob_true),
  You_Negation_prob_lr = get_row(you_neg$prob_lr)
))

#---------------------------------
# 3.3 Traditional training
#---------------------------------

filtered_data <- data %>% filter(model == "qwen32b", thinking == 0, training == 1, modality == "generic")
assertion_data <- filtered_data %>% filter(assertion_negation == "assertion")
negation_data <- filtered_data %>% filter(assertion_negation == "negation")

human_assert <- assertion_data %>% filter(who == "human")
human_neg <- negation_data %>% filter(who == "human")
llm_assert <- assertion_data %>% filter(who == "llm")
llm_neg <- negation_data %>% filter(who == "llm")
you_assert <- assertion_data %>% filter(who == "you")
you_neg <- negation_data %>% filter(who == "you")

table_results <- rbind(table_results, data.frame(
  Model = "Qwen32b",
  Specification = "No thinking + trad. training",
  Modality = "generic",
  N = nrow(human_assert),
  Human_Assertion_prob_true = get_row(human_assert$prob_true),
  Human_Assertion_prob_lr = get_row(human_assert$prob_lr),
  LLM_Assertion_prob_true = get_row(llm_assert$prob_true),
  LLM_Assertion_prob_lr = get_row(llm_assert$prob_lr),
  You_Assertion_prob_true = get_row(you_assert$prob_true),
  You_Assertion_prob_lr = get_row(you_assert$prob_lr),
  Human_Negation_prob_true = get_row(human_neg$prob_true),
  Human_Negation_prob_lr = get_row(human_neg$prob_lr),
  LLM_Negation_prob_true = get_row(llm_neg$prob_true),
  LLM_Negation_prob_lr = get_row(llm_neg$prob_lr),
  You_Negation_prob_true = get_row(you_neg$prob_true),
  You_Negation_prob_lr = get_row(you_neg$prob_lr)
))

#---------------------------------
# 3.4 Emotional
#---------------------------------
filtered_data <- data %>% filter(model == "qwen32b", thinking == 1, training == 0, modality %in% c("emotional_positive", "emotional_negative"))
assertion_data <- filtered_data %>% filter(assertion_negation == "assertion")
negation_data <- filtered_data %>% filter(assertion_negation == "negation")

human_assert <- assertion_data %>% filter(who == "human")
human_neg <- negation_data %>% filter(who == "human")
llm_assert <- assertion_data %>% filter(who == "llm")
llm_neg <- negation_data %>% filter(who == "llm")
you_assert <- assertion_data %>% filter(who == "you")
you_neg <- negation_data %>% filter(who == "you")

table_results <- rbind(table_results, data.frame(
  Model = "Qwen32b",
  Specification = "Thinking",
  N = nrow(human_assert),
  Human_Assertion_prob_true = get_row(human_assert$prob_true),
  Human_Assertion_prob_lr = get_row(human_assert$prob_lr),
  Human_Negation_prob_true = get_row(human_neg$prob_true),
  Human_Negation_prob_lr = get_row(human_neg$prob_lr),
  LLM_Assertion_prob_true = get_row(llm_assert$prob_true),
  LLM_Assertion_prob_lr = get_row(llm_assert$prob_lr),
  LLM_Negation_prob_true = get_row(llm_neg$prob_true),
  LLM_Negation_prob_lr = get_row(llm_neg$prob_lr),
  You_Assertion_prob_true = get_row(you_assert$prob_true),
  You_Assertion_prob_lr = get_row(you_assert$prob_lr),
  You_Negation_prob_true = get_row(you_neg$prob_true),
  You_Negation_prob_lr = get_row(you_neg$prob_lr),
  Modality = "emotional"
))

#---------------------------------
# 3.5 Visual
#---------------------------------
filtered_data <- data %>% filter(model == "qwen32b", thinking == 1, training == 0, modality == "visual")
assertion_data <- filtered_data %>% filter(assertion_negation == "assertion")
negation_data <- filtered_data %>% filter(assertion_negation == "negation")

human_assert <- assertion_data %>% filter(who == "human")
human_neg <- negation_data %>% filter(who == "human")
llm_assert <- assertion_data %>% filter(who == "llm")
llm_neg <- negation_data %>% filter(who == "llm")
you_assert <- assertion_data %>% filter(who == "you")
you_neg <- negation_data %>% filter(who == "you")

table_results <- rbind(table_results, data.frame(
  Model = "Qwen32b",
  Specification = "Default",
  N = nrow(human_assert),
  Human_Assertion_prob_true = get_row(human_assert$prob_true),
  Human_Assertion_prob_lr = get_row(human_assert$prob_lr),
  Human_Negation_prob_true = get_row(human_neg$prob_true),
  Human_Negation_prob_lr = get_row(human_neg$prob_lr),
  LLM_Assertion_prob_true = get_row(llm_assert$prob_true),
  LLM_Assertion_prob_lr = get_row(llm_assert$prob_lr),
  LLM_Negation_prob_true = get_row(llm_neg$prob_true),
  LLM_Negation_prob_lr = get_row(llm_neg$prob_lr),
  You_Assertion_prob_true = get_row(you_assert$prob_true),
  You_Assertion_prob_lr = get_row(you_assert$prob_lr),
  You_Negation_prob_true = get_row(you_neg$prob_true),
  You_Negation_prob_lr = get_row(you_neg$prob_lr),
  Modality = "visual"
))

#---------------------------------
# 3.6 Sensory modalities (combined: other modalities)
#---------------------------------
filtered_data <- data %>% filter(model == "qwen32b", thinking == 1, training == 0, modality %in% c("gustation", "olfactory", "aural", "tactile"))
assertion_data <- filtered_data %>% filter(assertion_negation == "assertion")
negation_data <- filtered_data %>% filter(assertion_negation == "negation")

human_assert <- assertion_data %>% filter(who == "human")
human_neg <- negation_data %>% filter(who == "human")
llm_assert <- assertion_data %>% filter(who == "llm")
llm_neg <- negation_data %>% filter(who == "llm")
you_assert <- assertion_data %>% filter(who == "you")
you_neg <- negation_data %>% filter(who == "you")

table_results <- rbind(table_results, data.frame(
  Model = "Qwen32b",
  Specification = "Default",
  N = nrow(human_assert),
  Human_Assertion_prob_true = get_row(human_assert$prob_true),
  Human_Assertion_prob_lr = get_row(human_assert$prob_lr),
  Human_Negation_prob_true = get_row(human_neg$prob_true),
  Human_Negation_prob_lr = get_row(human_neg$prob_lr),
  LLM_Assertion_prob_true = get_row(llm_assert$prob_true),
  LLM_Assertion_prob_lr = get_row(llm_assert$prob_lr),
  LLM_Negation_prob_true = get_row(llm_neg$prob_true),
  LLM_Negation_prob_lr = get_row(llm_neg$prob_lr),
  You_Assertion_prob_true = get_row(you_assert$prob_true),
  You_Assertion_prob_lr = get_row(you_assert$prob_lr),
  You_Negation_prob_true = get_row(you_neg$prob_true),
  You_Negation_prob_lr = get_row(you_neg$prob_lr),
  Modality = "other modalities"
))

#=================================
# 4. Llama70b Model
#=================================

#---------------------------------
# 4.1 Default (thinking=0, training=0, generic)
#---------------------------------
filtered_data <- data %>% filter(model == "llama70b", thinking == 0, training == 0, modality == "generic")
assertion_data <- filtered_data %>% filter(assertion_negation == "assertion")
negation_data <- filtered_data %>% filter(assertion_negation == "negation")

human_assert <- assertion_data %>% filter(who == "human")
human_neg <- negation_data %>% filter(who == "human")
llm_assert <- assertion_data %>% filter(who == "llm")
llm_neg <- negation_data %>% filter(who == "llm")
you_assert <- assertion_data %>% filter(who == "you")
you_neg <- negation_data %>% filter(who == "you")

table_results <- rbind(table_results, data.frame(
  Model = "Llama70b",
  Specification = "Default",
  N = nrow(human_assert),
  Human_Assertion_prob_true = get_row(human_assert$prob_true),
  Human_Assertion_prob_lr = get_row(human_assert$prob_lr),
  Human_Negation_prob_true = get_row(human_neg$prob_true),
  Human_Negation_prob_lr = get_row(human_neg$prob_lr),
  LLM_Assertion_prob_true = get_row(llm_assert$prob_true),
  LLM_Assertion_prob_lr = get_row(llm_assert$prob_lr),
  LLM_Negation_prob_true = get_row(llm_neg$prob_true),
  LLM_Negation_prob_lr = get_row(llm_neg$prob_lr),
  You_Assertion_prob_true = get_row(you_assert$prob_true),
  You_Assertion_prob_lr = get_row(you_assert$prob_lr),
  You_Negation_prob_true = get_row(you_neg$prob_true),
  You_Negation_prob_lr = get_row(you_neg$prob_lr),
  Modality = "generic"
))

#---------------------------------
# 4.2 Emotional
#---------------------------------
filtered_data <- data %>% filter(model == "llama70b", thinking == 0, training == 0, modality %in% c("emotional_positive", "emotional_negative"))
assertion_data <- filtered_data %>% filter(assertion_negation == "assertion")
negation_data <- filtered_data %>% filter(assertion_negation == "negation")

human_assert <- assertion_data %>% filter(who == "human")
human_neg <- negation_data %>% filter(who == "human")
llm_assert <- assertion_data %>% filter(who == "llm")
llm_neg <- negation_data %>% filter(who == "llm")
you_assert <- assertion_data %>% filter(who == "you")
you_neg <- negation_data %>% filter(who == "you")

table_results <- rbind(table_results, data.frame(
  Model = "Llama70b",
  Specification = "Default",
  N = nrow(human_assert),
  Human_Assertion_prob_true = get_row(human_assert$prob_true),
  Human_Assertion_prob_lr = get_row(human_assert$prob_lr),
  Human_Negation_prob_true = get_row(human_neg$prob_true),
  Human_Negation_prob_lr = get_row(human_neg$prob_lr),
  LLM_Assertion_prob_true = get_row(llm_assert$prob_true),
  LLM_Assertion_prob_lr = get_row(llm_assert$prob_lr),
  LLM_Negation_prob_true = get_row(llm_neg$prob_true),
  LLM_Negation_prob_lr = get_row(llm_neg$prob_lr),
  You_Assertion_prob_true = get_row(you_assert$prob_true),
  You_Assertion_prob_lr = get_row(you_assert$prob_lr),
  You_Negation_prob_true = get_row(you_neg$prob_true),
  You_Negation_prob_lr = get_row(you_neg$prob_lr),
  Modality = "emotional"
))

#---------------------------------
# 4.3 Visual
#---------------------------------
filtered_data <- data %>% filter(model == "llama70b", thinking == 0, training == 0, modality == "visual")
assertion_data <- filtered_data %>% filter(assertion_negation == "assertion")
negation_data <- filtered_data %>% filter(assertion_negation == "negation")

human_assert <- assertion_data %>% filter(who == "human")
human_neg <- negation_data %>% filter(who == "human")
llm_assert <- assertion_data %>% filter(who == "llm")
llm_neg <- negation_data %>% filter(who == "llm")
you_assert <- assertion_data %>% filter(who == "you")
you_neg <- negation_data %>% filter(who == "you")

table_results <- rbind(table_results, data.frame(
  Model = "Llama70b",
  Specification = "Default",
  N = nrow(human_assert),
  Human_Assertion_prob_true = get_row(human_assert$prob_true),
  Human_Assertion_prob_lr = get_row(human_assert$prob_lr),
  Human_Negation_prob_true = get_row(human_neg$prob_true),
  Human_Negation_prob_lr = get_row(human_neg$prob_lr),
  LLM_Assertion_prob_true = get_row(llm_assert$prob_true),
  LLM_Assertion_prob_lr = get_row(llm_assert$prob_lr),
  LLM_Negation_prob_true = get_row(llm_neg$prob_true),
  LLM_Negation_prob_lr = get_row(llm_neg$prob_lr),
  You_Assertion_prob_true = get_row(you_assert$prob_true),
  You_Assertion_prob_lr = get_row(you_assert$prob_lr),
  You_Negation_prob_true = get_row(you_neg$prob_true),
  You_Negation_prob_lr = get_row(you_neg$prob_lr),
  Modality = "visual"
))

#---------------------------------
# 4.4 Sensory modalities (combined: other modalities)
#---------------------------------
filtered_data <- data %>% filter(model == "llama70b", thinking == 0, training == 0, modality %in% c("gustation", "olfactory", "aural", "tactile"))
assertion_data <- filtered_data %>% filter(assertion_negation == "assertion")
negation_data <- filtered_data %>% filter(assertion_negation == "negation")

human_assert <- assertion_data %>% filter(who == "human")
human_neg <- negation_data %>% filter(who == "human")
llm_assert <- assertion_data %>% filter(who == "llm")
llm_neg <- negation_data %>% filter(who == "llm")
you_assert <- assertion_data %>% filter(who == "you")
you_neg <- negation_data %>% filter(who == "you")

table_results <- rbind(table_results, data.frame(
  Model = "Llama70b",
  Specification = "Default",
  N = nrow(human_assert),
  Human_Assertion_prob_true = get_row(human_assert$prob_true),
  Human_Assertion_prob_lr = get_row(human_assert$prob_lr),
  Human_Negation_prob_true = get_row(human_neg$prob_true),
  Human_Negation_prob_lr = get_row(human_neg$prob_lr),
  LLM_Assertion_prob_true = get_row(llm_assert$prob_true),
  LLM_Assertion_prob_lr = get_row(llm_assert$prob_lr),
  LLM_Negation_prob_true = get_row(llm_neg$prob_true),
  LLM_Negation_prob_lr = get_row(llm_neg$prob_lr),
  You_Assertion_prob_true = get_row(you_assert$prob_true),
  You_Assertion_prob_lr = get_row(you_assert$prob_lr),
  You_Negation_prob_true = get_row(you_neg$prob_true),
  You_Negation_prob_lr = get_row(you_neg$prob_lr),
  Modality = "other modalities"
))

#=================================
# 5. GPT20b Model
#=================================

#---------------------------------
# 5.1 Thinking (thinking=1, training=0, generic)
#---------------------------------
filtered_data <- data %>% filter(model == "gpt20b", thinking == 1, training == 0, modality == "generic")
assertion_data <- filtered_data %>% filter(assertion_negation == "assertion")
negation_data <- filtered_data %>% filter(assertion_negation == "negation")

human_assert <- assertion_data %>% filter(who == "human")
human_neg <- negation_data %>% filter(who == "human")
llm_assert <- assertion_data %>% filter(who == "llm")
llm_neg <- negation_data %>% filter(who == "llm")
you_assert <- assertion_data %>% filter(who == "you")
you_neg <- negation_data %>% filter(who == "you")

table_results <- rbind(table_results, data.frame(
  Model = "GPT20b",
  Specification = "Default",
  N = nrow(human_assert),
  Human_Assertion_prob_true = get_row(human_assert$prob_true),
  Human_Assertion_prob_lr = get_row(human_assert$prob_lr),
  Human_Negation_prob_true = get_row(human_neg$prob_true),
  Human_Negation_prob_lr = get_row(human_neg$prob_lr),
  LLM_Assertion_prob_true = get_row(llm_assert$prob_true),
  LLM_Assertion_prob_lr = get_row(llm_assert$prob_lr),
  LLM_Negation_prob_true = get_row(llm_neg$prob_true),
  LLM_Negation_prob_lr = get_row(llm_neg$prob_lr),
  You_Assertion_prob_true = get_row(you_assert$prob_true),
  You_Assertion_prob_lr = get_row(you_assert$prob_lr),
  You_Negation_prob_true = get_row(you_neg$prob_true),
  You_Negation_prob_lr = get_row(you_neg$prob_lr),
  Modality = "generic"
))

#---------------------------------
# 5.2 Thinking (thinking=1, training=0, generic)
#---------------------------------
filtered_data <- data %>% filter(model == "gpt20b", thinking == 0, training == 0, modality == "generic")
assertion_data <- filtered_data %>% filter(assertion_negation == "assertion")
negation_data <- filtered_data %>% filter(assertion_negation == "negation")

human_assert <- assertion_data %>% filter(who == "human")
human_neg <- negation_data %>% filter(who == "human")
llm_assert <- assertion_data %>% filter(who == "llm")
llm_neg <- negation_data %>% filter(who == "llm")
you_assert <- assertion_data %>% filter(who == "you")
you_neg <- negation_data %>% filter(who == "you")

table_results <- rbind(table_results, data.frame(
  Model = "GPT20b",
  Specification = "Default",
  N = nrow(human_assert),
  Human_Assertion_prob_true = get_row(human_assert$prob_true),
  Human_Assertion_prob_lr = get_row(human_assert$prob_lr),
  Human_Negation_prob_true = get_row(human_neg$prob_true),
  Human_Negation_prob_lr = get_row(human_neg$prob_lr),
  LLM_Assertion_prob_true = get_row(llm_assert$prob_true),
  LLM_Assertion_prob_lr = get_row(llm_assert$prob_lr),
  LLM_Negation_prob_true = get_row(llm_neg$prob_true),
  LLM_Negation_prob_lr = get_row(llm_neg$prob_lr),
  You_Assertion_prob_true = get_row(you_assert$prob_true),
  You_Assertion_prob_lr = get_row(you_assert$prob_lr),
  You_Negation_prob_true = get_row(you_neg$prob_true),
  You_Negation_prob_lr = get_row(you_neg$prob_lr),
  Modality = "generic"
))

#---------------------------------
# 5.3 Emotional
#---------------------------------
filtered_data <- data %>% filter(model == "gpt20b", thinking == 1, training == 0, modality %in% c("emotional_positive", "emotional_negative"))
assertion_data <- filtered_data %>% filter(assertion_negation == "assertion")
negation_data <- filtered_data %>% filter(assertion_negation == "negation")

human_assert <- assertion_data %>% filter(who == "human")
human_neg <- negation_data %>% filter(who == "human")
llm_assert <- assertion_data %>% filter(who == "llm")
llm_neg <- negation_data %>% filter(who == "llm")
you_assert <- assertion_data %>% filter(who == "you")
you_neg <- negation_data %>% filter(who == "you")

table_results <- rbind(table_results, data.frame(
  Model = "GPT20b",
  Specification = "Default",
  N = nrow(human_assert),
  Human_Assertion_prob_true = get_row(human_assert$prob_true),
  Human_Assertion_prob_lr = get_row(human_assert$prob_lr),
  Human_Negation_prob_true = get_row(human_neg$prob_true),
  Human_Negation_prob_lr = get_row(human_neg$prob_lr),
  LLM_Assertion_prob_true = get_row(llm_assert$prob_true),
  LLM_Assertion_prob_lr = get_row(llm_assert$prob_lr),
  LLM_Negation_prob_true = get_row(llm_neg$prob_true),
  LLM_Negation_prob_lr = get_row(llm_neg$prob_lr),
  You_Assertion_prob_true = get_row(you_assert$prob_true),
  You_Assertion_prob_lr = get_row(you_assert$prob_lr),
  You_Negation_prob_true = get_row(you_neg$prob_true),
  You_Negation_prob_lr = get_row(you_neg$prob_lr),
  Modality = "emotional"
))

# Remove emotional negative section - combined above

#---------------------------------
# 5.4 Visual
#---------------------------------
filtered_data <- data %>% filter(model == "gpt20b", thinking == 1, training == 0, modality == "visual")
assertion_data <- filtered_data %>% filter(assertion_negation == "assertion")
negation_data <- filtered_data %>% filter(assertion_negation == "negation")

human_assert <- assertion_data %>% filter(who == "human")
human_neg <- negation_data %>% filter(who == "human")
llm_assert <- assertion_data %>% filter(who == "llm")
llm_neg <- negation_data %>% filter(who == "llm")
you_assert <- assertion_data %>% filter(who == "you")
you_neg <- negation_data %>% filter(who == "you")

table_results <- rbind(table_results, data.frame(
  Model = "GPT20b",
  Specification = "Default",
  N = nrow(human_assert),
  Human_Assertion_prob_true = get_row(human_assert$prob_true),
  Human_Assertion_prob_lr = get_row(human_assert$prob_lr),
  Human_Negation_prob_true = get_row(human_neg$prob_true),
  Human_Negation_prob_lr = get_row(human_neg$prob_lr),
  LLM_Assertion_prob_true = get_row(llm_assert$prob_true),
  LLM_Assertion_prob_lr = get_row(llm_assert$prob_lr),
  LLM_Negation_prob_true = get_row(llm_neg$prob_true),
  LLM_Negation_prob_lr = get_row(llm_neg$prob_lr),
  You_Assertion_prob_true = get_row(you_assert$prob_true),
  You_Assertion_prob_lr = get_row(you_assert$prob_lr),
  You_Negation_prob_true = get_row(you_neg$prob_true),
  You_Negation_prob_lr = get_row(you_neg$prob_lr),
  Modality = "visual"
))

#---------------------------------
# 5.5 Sensory modalities (combined: other modalities)
#---------------------------------
filtered_data <- data %>% filter(model == "gpt20b", thinking == 1, training == 0, modality %in% c("gustation", "olfactory", "aural", "tactile"))
assertion_data <- filtered_data %>% filter(assertion_negation == "assertion")
negation_data <- filtered_data %>% filter(assertion_negation == "negation")

human_assert <- assertion_data %>% filter(who == "human")
human_neg <- negation_data %>% filter(who == "human")
llm_assert <- assertion_data %>% filter(who == "llm")
llm_neg <- negation_data %>% filter(who == "llm")
you_assert <- assertion_data %>% filter(who == "you")
you_neg <- negation_data %>% filter(who == "you")

table_results <- rbind(table_results, data.frame(
  Model = "GPT20b",
  Specification = "Default",
  N = nrow(human_assert),
  Human_Assertion_prob_true = get_row(human_assert$prob_true),
  Human_Assertion_prob_lr = get_row(human_assert$prob_lr),
  Human_Negation_prob_true = get_row(human_neg$prob_true),
  Human_Negation_prob_lr = get_row(human_neg$prob_lr),
  LLM_Assertion_prob_true = get_row(llm_assert$prob_true),
  LLM_Assertion_prob_lr = get_row(llm_assert$prob_lr),
  LLM_Negation_prob_true = get_row(llm_neg$prob_true),
  LLM_Negation_prob_lr = get_row(llm_neg$prob_lr),
  You_Assertion_prob_true = get_row(you_assert$prob_true),
  You_Assertion_prob_lr = get_row(you_assert$prob_lr),
  You_Negation_prob_true = get_row(you_neg$prob_true),
  You_Negation_prob_lr = get_row(you_neg$prob_lr),
  Modality = "other modalities"
))

#=================================
# 6. Create and save LaTeX table
#=================================

#---------------------------------
# 6.1. Create Table 1 (assertions only)
#---------------------------------

# Order
table1_assertions <- table_results %>%
  select(Model, Specification, Modality, N,
         Human_Assertion_prob_true, Human_Assertion_prob_lr,
         LLM_Assertion_prob_true, LLM_Assertion_prob_lr,
         You_Assertion_prob_true, You_Assertion_prob_lr) %>%
  rename(Human_prob_true = Human_Assertion_prob_true,
         Human_prob_lr = Human_Assertion_prob_lr,
         LLM_prob_true = LLM_Assertion_prob_true,
         LLM_prob_lr = LLM_Assertion_prob_lr,
         You_prob_true = You_Assertion_prob_true,
         You_prob_lr = You_Assertion_prob_lr)

# Create
latex_table1 <- xtable(table1_assertions, 
                       caption = "Model Comparison Summary Statistics - Assertion Statements",
                       label = "tab:model_comparison_assertions")

# Save
print(latex_table1, 
      file = "results/table1.tex",
      include.rownames = FALSE,
      table.placement = "htbp",
      caption.placement = "top")


#---------------------------------
# 6.2. Create Table A1 (negations only)
#---------------------------------

# Order 
tableA1_negations <- table_results %>%
  select(Model, Specification, Modality, N,
         Human_Negation_prob_true, Human_Negation_prob_lr,
         LLM_Negation_prob_true, LLM_Negation_prob_lr,
         You_Negation_prob_true, You_Negation_prob_lr) %>%
  rename(Human_prob_true = Human_Negation_prob_true,
         Human_prob_lr = Human_Negation_prob_lr,
         LLM_prob_true = LLM_Negation_prob_true,
         LLM_prob_lr = LLM_Negation_prob_lr,
         You_prob_true = You_Negation_prob_true,
         You_prob_lr = You_Negation_prob_lr)


# Create
latex_tableA1 <- xtable(tableA1_negations, 
                       caption = "Model Comparison Summary Statistics - Negation Statements",
                       label = "tab:model_comparison_negations")

# Save
print(latex_tableA1, 
      file = "results/tableA1.tex",
      include.rownames = FALSE,
      table.placement = "htbp",
      caption.placement = "top")
