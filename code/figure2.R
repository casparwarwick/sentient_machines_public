##########################################
# SENTIENCE PROJECT
##########################################
# Figure 2: Assertion vs Negation Comparison
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

# Filter to qwen32b model with specified conditions
filtered_data <- data %>%
  filter(
    model == "qwen32b",
    thinking == 0,
    training == 0,
    prompt == 0,
    modality == "generic"
  )

#=================================
# 3. Data preparation for scatterplot
#=================================

# Separate assertion and negation data
assertion_data <- filtered_data %>%
  filter(assertion_negation == "assertion") %>%
  select(who, base_sentence_id, prob_true, prob_lr, prob_mm, prob_ttpd) %>%
  rename(
    prob_true_assertion = prob_true,
    prob_lr_assertion = prob_lr,
    prob_mm_assertion = prob_mm,
    prob_ttpd_assertion = prob_ttpd
  )

negation_data <- filtered_data %>%
  filter(assertion_negation == "negation") %>%
  select(who, base_sentence_id, prob_true, prob_lr, prob_mm, prob_ttpd) %>%
  rename(
    prob_true_negation = prob_true,
    prob_lr_negation = prob_lr,
    prob_mm_negation = prob_mm,
    prob_ttpd_negation = prob_ttpd
  )

# Join assertion and negation data by entity and base sentence
scatter_data <- assertion_data %>%
  inner_join(negation_data, by = c("who", "base_sentence_id"))

# Create factor for entity colors
scatter_data$who <- factor(scatter_data$who, levels = c("human", "llm", "you"))

#=================================
# 4. Regression analysis
#=================================

# Fit linear regression models for all probability types
regression_model_true <- lm(prob_true_assertion ~ prob_true_negation, data = scatter_data)
regression_model_lr <- lm(prob_lr_assertion ~ prob_lr_negation, data = scatter_data)
regression_model_mm <- lm(prob_mm_assertion ~ prob_mm_negation, data = scatter_data)
regression_model_ttpd <- lm(prob_ttpd_assertion ~ prob_ttpd_negation, data = scatter_data)

# Extract regression coefficients and R-squared values
get_regression_stats <- function(model) {
  intercept <- round(coef(model)[1], 3)
  slope <- round(coef(model)[2], 3)
  r_squared <- round(summary(model)$r.squared, 3)
  
  # Format equation string
  equation <- paste0("y = ", 
                     ifelse(intercept >= 0, "", ""), 
                     intercept, 
                     ifelse(slope >= 0, " + ", " - "), 
                     abs(slope), "x")
  r_squared_text <- paste0("R² = ", r_squared)
  
  return(list(equation = equation, r_squared = r_squared_text))
}

# Get statistics for each model
stats_true <- get_regression_stats(regression_model_true)
stats_lr <- get_regression_stats(regression_model_lr)
stats_mm <- get_regression_stats(regression_model_mm)
stats_ttpd <- get_regression_stats(regression_model_ttpd)

#=================================
# 5. Create scatterplots
#=================================

# Function to create consistent scatterplot
create_scatterplot <- function(data, x_var, y_var, title, equation, r_squared) {
  ggplot(data, aes(x = .data[[x_var]], y = .data[[y_var]], color = who)) +
    geom_point(size = 2, alpha = 0.7) +
    geom_smooth(method = "lm", se = TRUE, color = "black", linetype = "solid", linewidth = 0.8) +
    scale_color_manual(
      name = "Entity type",
      values = c("human" = "#4169E1", "llm" = "#DC143C", "you" = "#32CD32"),
      labels = c("Human", "LLM", "You")
    ) +
    labs(
      title = title,
      x = "Probability (negation)",
      y = "Probability (assertion)") +
    # Add regression equation and R-squared
    annotate("text", x = 0.05, y = 0.95, label = equation, 
             hjust = 0, vjust = 1, size = 3.5, color = "black") +
    annotate("text", x = 0.05, y = 0.90, label = r_squared, 
             hjust = 0, vjust = 1, size = 3.5, color = "black") +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 16),
      axis.title = element_text(size = 14),
      axis.text = element_text(size = 12),
      axis.line = element_line(color = "black", size = 0.5),
      axis.ticks = element_line(color = "black", size = 0.5),
      legend.title = element_text(size = 14),
      legend.text = element_text(size = 12),
      legend.position = "bottom",
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA)
    ) +
    coord_fixed(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE)
}

# Create individual plots
scatter_plot_true <- create_scatterplot(scatter_data, "prob_true_negation", "prob_true_assertion", "Panel A: Continuation", stats_true$equation, stats_true$r_squared)
scatter_plot_lr <- create_scatterplot(scatter_data, "prob_lr_negation", "prob_lr_assertion", "Panel B: LR", stats_lr$equation, stats_lr$r_squared)
scatter_plot_mm <- create_scatterplot(scatter_data, "prob_mm_negation", "prob_mm_assertion", "Panel C: MM", stats_mm$equation, stats_mm$r_squared)
scatter_plot_ttpd <- create_scatterplot(scatter_data, "prob_ttpd_negation", "prob_ttpd_assertion", "Panel D: TTPD", stats_ttpd$equation, stats_ttpd$r_squared)

#=================================
# 6. Combine plots and save output
#=================================

# Combine all plots in 2x2 layout using patchwork
combined_plot <- (scatter_plot_true + scatter_plot_lr + scatter_plot_mm + scatter_plot_ttpd) +
  #plot_annotation(
  #  title = "Consistency across assertion and negation"
  #) +
  plot_layout(guides = "collect", ncol = 4) & 
  theme(
    legend.position = "bottom",
    plot.title = element_text(hjust = 0.5, size = 18),
    legend.title = element_text(size = 14),
    legend.text = element_text(size = 12)
  )
  

# Save combined figure
ggsave("results/figure2.png", combined_plot, width = 16, height = 5, dpi = 300)

# Display combined plot
combined_plot

# Display regression summaries
summary(regression_model_true)
summary(regression_model_lr)
summary(regression_model_mm)
summary(regression_model_ttpd)
