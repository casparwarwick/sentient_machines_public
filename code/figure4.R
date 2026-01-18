##########################################
# SENTIENCE PROJECT
##########################################
# Figure 4: Model Scaling 
##########################################

#=================================
# 1. Setup and configuration
#=================================

# Load libraries
library(tidyverse)
library(ggplot2)
library(patchwork)

# Set working directory to project root
setwd("")

#=================================
# 2. Data loading and filtering
#=================================

# Load combined results
data <- read_csv("datasets/combined_results.csv", show_col_types = FALSE)

# Filter to specified conditions for thinking models
filtered_data <- data %>%
  filter(
    training == 0,
    prompt == 0,
    modality == "generic",
    assertion_negation == "assertion",
    # Use thinking=1 for GPT and Qwen, thinking=0 for Llama (no thinking data)
    ((thinking == 1 & model %in% c("gpt20b", "qwen06b", "qwen8b", "qwen32b")) |
     (thinking == 0 & model %in% c("llama3b", "llama8b", "llama70b")))
  )

#=================================
# 3. Data preparation for scaling plot
#=================================

# Create parameter count mapping
model_info <- data.frame(
  model = c("qwen06b", "qwen8b", "qwen32b", "llama3b", "llama8b", "llama70b", "gpt20b"),
  param_count = c(0.6, 8, 32, 3, 8, 70, 20),
  family = c("qwen", "qwen", "qwen", "llama", "llama", "llama", "gpt"),
  label = c("Qwen 0.6B", "Qwen 8B", "Qwen 32B", "Llama 3B", "LLama 8B", "Llama 70B", "GPT 20B"),
  shape = c("square", "square", "square", "circle", "circle", "circle", "cross"),
  stringsAsFactors = FALSE
)

# Calculate mean prob_true and prob_lr for each model and entity
model_summary <- filtered_data %>%
  group_by(model, who) %>%
  summarise(
    prob_true = mean(prob_true, na.rm = TRUE),
    prob_lr = mean(prob_lr, na.rm = TRUE),
    .groups = "drop"
  )

# Join with model information and create separate datasets for each entity
plot_data_you <- model_summary %>%
  filter(who == "you") %>%
  inner_join(model_info, by = "model") %>%
  arrange(family, param_count)

plot_data_human <- model_summary %>%
  filter(who == "human") %>%
  inner_join(model_info, by = "model") %>%
  arrange(family, param_count)

plot_data_llm <- model_summary %>%
  filter(who == "llm") %>%
  inner_join(model_info, by = "model") %>%
  arrange(family, param_count)

#=================================
# 4. Create scaling plot
#=================================

# Function to create scaling plot with entity-specific colors
create_scaling_plot <- function(data, y_var, plot_title, point_color) {
  line_data <- data %>%
    mutate(linetype = case_when(
      family == "llama" ~ "dashed",
      TRUE ~ "solid"
    ))
  
  ggplot(data, aes(x = param_count, y = .data[[y_var]])) +
    geom_line(data = line_data, aes(group = family, linetype = linetype), color = "black", size = 0.8) +
    scale_linetype_identity() +
    geom_point(aes(shape = shape), color = point_color, size = 3, stroke = 1.0) +
    scale_shape_manual(values = c("square" = 15, "circle" = 16, "cross" = 4)) +
    geom_text(aes(label = label, 
                  vjust = ifelse(model == "llama8b", 0.2, -0.8),
                  hjust = case_when(
                    model == "qwen8b" ~ -0.2,
                    model == "llama8b" ~ 1.1,
                    TRUE ~ 0.5
                  )), 
              size = 4,
              color = "black") +
    labs(
      title = plot_title,
      x = "Parameter Count (Billions)",
      y = "Probability True (Assertions)"
    ) +
    scale_x_continuous(
      breaks = c(0, 10, 20, 30, 40, 50, 60, 70, 80),
      limits = c(0, 80)
    ) +
    scale_y_continuous(limits = c(0, 1)) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 16),
      plot.subtitle = element_text(hjust = 0.5, size = 13),
      axis.title = element_text(size = 14),
      axis.text = element_text(size = 12),
      axis.line = element_line(color = "black", size = 0.5),
      axis.ticks = element_line(color = "black", size = 0.5),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      legend.position = "none"
    )
}

# Create individual plots for all entities
# Human (blue)  
scaling_plot_human_continuations <- create_scaling_plot(plot_data_human, "prob_true", "Panel A: Human - Continuations", "#4169E1")
scaling_plot_human_lr <- create_scaling_plot(plot_data_human, "prob_lr", "Panel D: Human - LR", "#4169E1")

# LLM (red)
scaling_plot_llm_continuations <- create_scaling_plot(plot_data_llm, "prob_true", "Panel B: LLM - Continuations", "#DC143C")
scaling_plot_llm_lr <- create_scaling_plot(plot_data_llm, "prob_lr", "Panel E: LLM - LR", "#DC143C")

# You (green)
scaling_plot_you_continuations <- create_scaling_plot(plot_data_you, "prob_true", "Panel C: You - Continuations", "#32CD32")
scaling_plot_you_lr <- create_scaling_plot(plot_data_you, "prob_lr", "Panel F: You - LR", "#32CD32")

#=================================
# 5. Combine and save output
#=================================

# Combine plots
combined_scaling_plot <-  (scaling_plot_human_continuations + scaling_plot_llm_continuations + scaling_plot_you_continuations) /
  (scaling_plot_human_lr + scaling_plot_llm_lr + scaling_plot_you_lr)
  plot_layout(ncol = 3)

# Save the combined figure
ggsave("results/figure4.png", combined_scaling_plot, width = 16, height = 10, dpi = 300)

# Display plots
combined_scaling_plot