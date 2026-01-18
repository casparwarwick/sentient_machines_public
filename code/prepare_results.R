##########################################
# SENTIENCE PROJECT
##########################################
# Prepare Combined Results CSV
##########################################

#=================================
# 1. Setup and configuration
#=================================

# Load libraries
library(tidyverse)
library(readr)
library(dplyr)
library(stringr)

# Set working directory to project root
setwd("")

# Set path for all results
results_path <- "datasets/results"

# Initialize empty dataframe to store all data
combined_data <- data.frame()

#=================================
# 2. Main data processing
#=================================

#---------------------------------
# 2.1 gpt20b (regular)
#---------------------------------

model <- "gpt20b"
thinking <- 0
training <- 0

for (i in 1:3) {
  prompt_suffix <- c("001", "002", "003")[i]
  prompt_code <- i - 1
  
  data <- read_csv(paste0(results_path, "/gpt20b/prep_sentience_16_", prompt_suffix, ".csv"), show_col_types = FALSE)
  data$model <- model
  data$thinking <- thinking
  data$training <- training
  data$prompt <- prompt_code
  combined_data <- bind_rows(combined_data, data)
}

#---------------------------------
# 2.2 gpt20b (thinking)
#---------------------------------

model <- "gpt20b"
thinking <- 1
training <- 0

for (i in 1:3) {
  prompt_suffix <- c("001", "002", "003")[i]
  prompt_code <- i - 1
  
  data <- read_csv(paste0(results_path, "/gpt20b_think/prep_sentience_16_", prompt_suffix, ".csv"), show_col_types = FALSE)
  data$model <- model
  data$thinking <- thinking
  data$training <- training
  data$prompt <- prompt_code
  combined_data <- bind_rows(combined_data, data)
}

#---------------------------------
# 2.3 llama70b (regular)
#---------------------------------

model <- "llama70b"
thinking <- 0
training <- 0
for (i in 1:3) {
  prompt_suffix <- c("001", "002", "003")[i]
  prompt_code <- i - 1
  
  data <- read_csv(paste0(results_path, "/llama70b/prep_sentience_16_", prompt_suffix, ".csv"), show_col_types = FALSE)
  data$model <- model
  data$thinking <- thinking
  data$training <- training
  data$prompt <- prompt_code
  combined_data <- bind_rows(combined_data, data)
}

#---------------------------------
# 2.4 llama8b (regular)
#---------------------------------

model <- "llama8b"
thinking <- 0
training <- 0

for (i in 1:3) {
  prompt_suffix <- c("001", "002", "003")[i]
  prompt_code <- i - 1
  
  data <- read_csv(paste0(results_path, "/llama8b/prep_sentience_16_", prompt_suffix, ".csv"), show_col_types = FALSE)
  data$model <- model
  data$thinking <- thinking
  data$training <- training
  data$prompt <- prompt_code
  combined_data <- bind_rows(combined_data, data)
}

#---------------------------------
# 2.5 llama3b (regular)
#---------------------------------

model <- "llama3b"
thinking <- 0
training <- 0

for (i in 1:3) {
  prompt_suffix <- c("001", "002", "003")[i]
  prompt_code <- i - 1
  
  data <- read_csv(paste0(results_path, "/llama3b/prep_sentience_16_", prompt_suffix, ".csv"), show_col_types = FALSE)
  data$model <- model
  data$thinking <- thinking
  data$training <- training
  data$prompt <- prompt_code
  combined_data <- bind_rows(combined_data, data)
}

#---------------------------------
# 2.6 qwen06b (regular)
#---------------------------------

model <- "qwen06b"
thinking <- 0
training <- 0

for (i in 1:3) {
  prompt_suffix <- c("001", "002", "003")[i]
  prompt_code <- i - 1
  
  data <- read_csv(paste0(results_path, "/qwen06b/prep_sentience_16_", prompt_suffix, ".csv"), show_col_types = FALSE)
  data$model <- model
  data$thinking <- thinking
  data$training <- training
  data$prompt <- prompt_code
  combined_data <- bind_rows(combined_data, data)
}

#---------------------------------
# 2.7 qwen06b (thinking)
#---------------------------------

model <- "qwen06b"
thinking <- 1
training <- 0

for (i in 1:3) {
  prompt_suffix <- c("001", "002", "003")[i]
  prompt_code <- i - 1
  
  data <- read_csv(paste0(results_path, "/qwen06b_think/prep_sentience_16_", prompt_suffix, ".csv"), show_col_types = FALSE)
  data$model <- model
  data$thinking <- thinking
  data$training <- training
  data$prompt <- prompt_code
  combined_data <- bind_rows(combined_data, data)
}

#---------------------------------
# 2.8 qwen32b (regular)
#---------------------------------

model <- "qwen32b"
thinking <- 0
training <- 0

for (i in 1:3) {
  prompt_suffix <- c("001", "002", "003")[i]
  prompt_code <- i - 1
  
  data <- read_csv(paste0(results_path, "/qwen32b/prep_sentience_16_", prompt_suffix, ".csv"), show_col_types = FALSE)
  data$model <- model
  data$thinking <- thinking
  data$training <- training
  data$prompt <- prompt_code
  combined_data <- bind_rows(combined_data, data)
}

#---------------------------------
# 2.9 qwen32b (thinking)
#---------------------------------

model <- "qwen32b"
thinking <- 1
training <- 0

for (i in 1:3) {
  prompt_suffix <- c("001", "002", "003")[i]
  prompt_code <- i - 1
  
  data <- read_csv(paste0(results_path, "/qwen32b_think/prep_sentience_16_", prompt_suffix, ".csv"), show_col_types = FALSE)
  data$model <- model
  data$thinking <- thinking
  data$training <- training
  data$prompt <- prompt_code
  combined_data <- bind_rows(combined_data, data)
}

#---------------------------------
# 2.10 qwen32b (traditional training)
#---------------------------------

model <- "qwen32b"
thinking <- 0
training <- 1

for (i in 1:3) {
  prompt_suffix <- c("001", "002", "003")[i]
  prompt_code <- i - 1
  
  data <- read_csv(paste0(results_path, "/qwen32b_trad/prep_sentience_16_", prompt_suffix, ".csv"), show_col_types = FALSE)
  data$model <- model
  data$thinking <- thinking
  data$training <- training
  data$prompt <- prompt_code
  combined_data <- bind_rows(combined_data, data)
}

#---------------------------------
# 2.11 qwen8b (regular)
#---------------------------------

model <- "qwen8b"
thinking <- 0
training <- 0

for (i in 1:3) {
  prompt_suffix <- c("001", "002", "003")[i]
  prompt_code <- i - 1
  
  data <- read_csv(paste0(results_path, "/qwen8b/prep_sentience_16_", prompt_suffix, ".csv"), show_col_types = FALSE)
  data$model <- model
  data$thinking <- thinking
  data$training <- training
  data$prompt <- prompt_code
  combined_data <- bind_rows(combined_data, data)
}

#---------------------------------
# 2.12 qwen8b (thinking)
#---------------------------------

model <- "qwen8b"
thinking <- 1
training <- 0

for (i in 1:3) {
  prompt_suffix <- c("001", "002", "003")[i]
  prompt_code <- i - 1
  
  data <- read_csv(paste0(results_path, "/qwen8b_think/prep_sentience_16_", prompt_suffix, ".csv"), show_col_types = FALSE)
  data$model <- model
  data$thinking <- thinking
  data$training <- training
  data$prompt <- prompt_code
  combined_data <- bind_rows(combined_data, data)
}

#=================================
# 3. Data processing and output
#=================================

#---------------------------------
# 3.1 Factor conversion
#---------------------------------

# Convert new categorical variables to factors
combined_data$thinking <- as.factor(combined_data$thinking)
combined_data$training <- as.factor(combined_data$training) 
combined_data$prompt <- as.factor(combined_data$prompt)

#=================================
# 4. Save
#=================================

write_csv(combined_data, "datasets/combined_results.csv")

