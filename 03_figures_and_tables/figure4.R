##########################################
# SENTIENCE PROJECT
##########################################
# Figure 4: Model Scaling
# Produces both the assertions figure (outputs/figure4.png) and the corresponding
# negations figure (outputs/figure4_negations.png).
##########################################

#=================================
# 1. Setup and configuration
#=================================

# Load libraries
library(tidyverse)
library(ggplot2)
library(patchwork)

# Run this script from the repository root directory.

#=================================
# 2. Data loading
#=================================

# Load combined results
data <- read_csv("combined_csvs/combined_results.csv", show_col_types = FALSE)

#=================================
# 3. Model information
#=================================

# Create parameter count mapping
model_info <- data.frame(
  model = c("qwen06b", "qwen8b", "qwen32b", "llama3b", "llama8b", "llama70b", "gpt20b", "gpt120b"),
  param_count = c(0.6, 8, 32, 3, 8, 70, 20, 120),
  family = c("qwen", "qwen", "qwen", "llama", "llama", "llama", "gpt", "gpt"),
  label = c("Qwen 0.6B", "Qwen 8B", "Qwen 32B", "Llama 3B", "LLama 8B", "Llama 70B", "GPT 20B", "GPT 120B"),
  shape = c("square", "square", "square", "circle", "circle", "circle", "cross", "cross"),
  stringsAsFactors = FALSE
)

#=================================
# 4. Plotting function
#=================================

# Function to create scaling plot with entity-specific colors
create_scaling_plot <- function(data, y_var, plot_title, point_color, y_label,
                                vjust_overrides = list(), hjust_overrides = list()) {
  line_data <- data %>%
    mutate(linetype = case_when(
      family == "llama" ~ "dashed",
      family == "gpt" ~ "dotted",
      TRUE ~ "solid"
    ),
    line_group = family)

  data <- data %>%
    rowwise() %>%
    mutate(
      label_vjust = if (model %in% names(vjust_overrides)) vjust_overrides[[model]]
                    else if (model == "llama8b") 0.2
                    else -0.8,
      label_hjust = if (model %in% names(hjust_overrides)) hjust_overrides[[model]]
                    else if (model == "qwen8b") -0.2
                    else if (model == "llama8b") 1.1
                    else if (model == "qwen06b") -0.2
                    else 0.5
    ) %>%
    ungroup()

  ggplot(data, aes(x = param_count, y = .data[[y_var]])) +
    geom_line(data = line_data, aes(group = line_group, linetype = linetype), color = "black", size = 0.8) +
    scale_linetype_identity() +
    geom_point(aes(shape = shape), color = point_color, size = 3, stroke = 1.0) +
    scale_shape_manual(values = c("square" = 15, "circle" = 16, "cross" = 4)) +
    geom_text(aes(label = label,
                  vjust = label_vjust,
                  hjust = label_hjust),
              size = 4,
              color = "black") +
    labs(
      title = plot_title,
      x = "Parameter Count (Billions, log scale)",
      y = y_label
    ) +
    scale_x_log10(
      breaks = c(1, 3, 10, 30, 100),
      limits = c(0.5, 200),
      labels = c("1", "3", "10", "30", "100")
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

#=================================
# 5. Figure builder
#=================================

# Builds the full 2x3 scaling figure for a given statement type ("assertion"
# or "negation") and saves it to out_file. y_label sets the shared y-axis title.
build_scaling_figure <- function(an_value, y_label, out_file) {

  # Filter to specified conditions for thinking models
  # Use thinking=1 for GPT and Qwen, thinking=0 for Llama (no thinking data)
  filtered_data <- data %>%
    filter(
      training == 0,
      prompt == 0,
      modality == "generic",
      assertion_negation == an_value,
      ((thinking == 1 & model %in% c("gpt20b", "gpt120b", "qwen06b", "qwen8b", "qwen32b")) |
         (thinking == 0 & model %in% c("llama3b", "llama8b", "llama70b")))
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

  # Create individual plots for all entities
  # Human (blue)
  scaling_plot_human_continuations <- create_scaling_plot(plot_data_human, "prob_true", "Panel A: Human - Continuations", "#4169E1", y_label,
    vjust_overrides = list(llama70b = 1.5, qwen8b = 1.5),
    hjust_overrides = list(qwen8b = 0.5, gpt20b = -0.2))
  scaling_plot_human_lr <- create_scaling_plot(plot_data_human, "prob_lr", "Panel D: Human - LR", "#4169E1", y_label,
    vjust_overrides = list(qwen32b = 1.5, qwen8b = 1.5, gpt120b = 1.5))

  # LLM (red)
  scaling_plot_llm_continuations <- create_scaling_plot(plot_data_llm, "prob_true", "Panel B: LLM - Continuations", "#DC143C", y_label,
    vjust_overrides = list(gpt120b = 1.5))
  scaling_plot_llm_lr <- create_scaling_plot(plot_data_llm, "prob_lr", "Panel E: LLM - LR", "#DC143C", y_label)

  # You (green)
  scaling_plot_you_continuations <- create_scaling_plot(plot_data_you, "prob_true", "Panel C: You - Continuations", "#32CD32", y_label,
    vjust_overrides = list(gpt20b = 1.5, gpt120b = 1.5))
  scaling_plot_you_lr <- create_scaling_plot(plot_data_you, "prob_lr", "Panel F: You - LR", "#32CD32", y_label,
    vjust_overrides = list(gpt20b = 1.5, gpt120b = 1.5))

  # Combine plots
  combined_scaling_plot <- (scaling_plot_human_continuations + scaling_plot_llm_continuations + scaling_plot_you_continuations) /
    (scaling_plot_human_lr + scaling_plot_llm_lr + scaling_plot_you_lr)

  # Save the combined figure
  ggsave(out_file, combined_scaling_plot, width = 16, height = 10, dpi = 300)

  combined_scaling_plot
}

#=================================
# 6. Build and save both figures
#=================================

build_scaling_figure("assertion", "Probability True (Assertions)", "outputs/figure4.png")
build_scaling_figure("negation",  "Probability True (Negations)",  "outputs/figure4_negations.png")
