##########################################
# SENTIENCE PROJECT
##########################################
# Training Performance - Layer-wise
##########################################

#=================================
# 1. Setup and configuration
#=================================

# Load libraries
library(tidyverse)
library(ggplot2)

# Set working directory to project root
setwd("")

#=================================
# 2. Data loading and combining
#=================================

# Set path for training accuracy layer data
layer_data_path <- "datasets/from_blythe/training_accuracy_layers"

# Initialize empty dataframe to store all data
combined_data <- data.frame()

# List of available model families (add more as they become available)
available_model_families <- c("llama", "qwen")

# Load and combine data from each classifier type
for (classifier in c("lr", "mm", "ttpd")) {
  for (model_family in available_model_families) {
    file_path <- file.path(layer_data_path, paste0("training_accuracy_layers_", classifier, "_", model_family, ".csv"))
    
    data <- read_csv(file_path, show_col_types = FALSE)
    data$classifier_type <- classifier
    data$model_family <- model_family
    combined_data <- bind_rows(combined_data, data)
  }
}

#=================================
# 3. Data preparation for plotting
#=================================


# Create factor levels for proper ordering
combined_data$classifier_type <- factor(combined_data$classifier_type, levels = c("lr", "mm", "ttpd"))

# Get accuracy column dynamically and create factor for model families
combined_data <- combined_data %>%
  mutate(
    # Get accuracy column dynamically
    accuracy = case_when(
      classifier_type == "lr" ~ acc_lr,
      classifier_type == "mm" ~ acc_mm,
      classifier_type == "ttpd" ~ acc_ttpd,
      TRUE ~ NA_real_
    )
  )

# Create factor for model families
combined_data$model_family <- factor(combined_data$model_family, levels = c("qwen", "llama", "gpt"))

#=================================
# 4. Create layer performance figure
#=================================

model_colors <- c(
  "qwen" = "#696969",   
  "llama" = "#DAA520",   
  "gpt" = "#6A0DAD"     
)

# Define line types for classifiers
classifier_linetypes <- c(
  "lr" = "solid",
  "mm" = "dashed", 
  "ttpd" = "dotted"
)

# Create the plot
layer_plot <- ggplot(combined_data, aes(x = layer, y = accuracy, 
                                       color = model_family, 
                                       linetype = classifier_type)) +
  geom_line(size = 0.8) +
  scale_color_manual(
    name = "Model Family",
    values = model_colors,
    labels = c("Qwen", "Llama", "GPT")
  ) +
  scale_linetype_manual(
    name = "Classifier Type", 
    values = classifier_linetypes,
    labels = c("LR", "MM", "TTPD")
  ) +
  scale_x_continuous(breaks = seq(0, 100, by = 10)) +
  labs(
    x = "Layer",
    y = "Accuracy"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 16),
    axis.title = element_text(size = 14),
    axis.text = element_text(size = 12),
    axis.line = element_line(color = "black", size = 0.5),
    axis.ticks = element_line(color = "black", size = 0.5),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10),
    legend.position = "bottom",
    panel.background = element_rect(fill = "white", color = NA),
    plot.background = element_rect(fill = "white", color = NA)
  ) +
  guides(
    color = guide_legend(override.aes = list(linetype = "solid")),
    linetype = guide_legend(override.aes = list(color = "black"))
  )

#=================================
# 5. Save output
#=================================

# Save the figure
ggsave("results/figure_training_performance_layers.png", layer_plot, width = 10, height = 6, dpi = 300)

# Display plot
layer_plot