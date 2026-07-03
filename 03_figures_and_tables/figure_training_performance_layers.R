##########################################
# SENTIENCE PROJECT
##########################################
# Training Performance Analysis - Layer-wise
##########################################

#=================================
# 1. Setup and configuration
#=================================

# Load libraries
library(tidyverse)
library(ggplot2)

#=================================
# 2. Data loading and combining
#=================================

layer_data_path <- "cluster_outputs/model_knowledge-qwen3-32b-220626-think"

# Initialize empty dataframe to store all data
combined_data <- data.frame()

# Load and combine data from each classifier type
for (classifier in c("lr", "mm", "ttpd")) {
  file_path <- file.path(layer_data_path, paste0("training_accuracy_layers_", classifier, ".csv"))

  data <- read_csv(file_path, show_col_types = FALSE)
  data$classifier_type <- classifier
  combined_data <- bind_rows(combined_data, data)
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

#=================================
# 4. Create layer performance figure
#=================================

# One line per classifier (single model: Qwen3-32b with thinking).
classifier_colors <- c(
  "lr" = "#1f77b4",
  "mm" = "#ff7f0e",
  "ttpd" = "#2ca02c"
)

# Create the plot
layer_plot <- ggplot(combined_data, aes(x = layer, y = accuracy,
                                       color = classifier_type)) +
  geom_line(size = 0.8) +
  scale_color_manual(
    name = "Classifier Type",
    values = classifier_colors,
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
  )

#=================================
# 5. Save output
#=================================

# Save the figure
ggsave("outputs/figure_training_performance_layers.png", layer_plot, width = 10, height = 6, dpi = 300)

# Display plot
layer_plot