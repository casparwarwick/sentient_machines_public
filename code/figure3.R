##########################################
# SENTIENCE PROJECT
##########################################
# Figure 3: Prompt Condition Comparison (Qwen32b)
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

# Filter to qwen32b model with thinking=1 conditions
filtered_data <- data %>%
  filter(
    thinking == 0,
    training == 0,
    modality == "generic",
    model == "qwen32b"
  )

#=================================
# 3. Data preparation for plotting
#=================================

# Separate data by prompt condition and assertion/negation
prompt0_assertion_data <- filtered_data %>%
  filter(prompt == 0, assertion_negation == "assertion") %>%
  filter(who %in% c("human", "llm", "you")) %>%
  group_by(who) %>%
  summarise(
    # Means
    prob_true_mean = mean(prob_true, na.rm = TRUE),
    prob_lr_mean = mean(prob_lr, na.rm = TRUE),
    prob_mm_mean = mean(prob_mm, na.rm = TRUE),
    prob_ttpd_mean = mean(prob_ttpd, na.rm = TRUE),
    # Standard deviations
    prob_true_sd = sd(prob_true, na.rm = TRUE),
    prob_lr_sd = sd(prob_lr, na.rm = TRUE),
    prob_mm_sd = sd(prob_mm, na.rm = TRUE),
    prob_ttpd_sd = sd(prob_ttpd, na.rm = TRUE),
    # Counts
    prob_true_n = sum(!is.na(prob_true)),
    prob_lr_n = sum(!is.na(prob_lr)),
    prob_mm_n = sum(!is.na(prob_mm)),
    prob_ttpd_n = sum(!is.na(prob_ttpd)),
    .groups = "drop"
  ) %>%
  # Reshape to long format
  pivot_longer(
    cols = -who,
    names_to = c("probability_type", "stat"),
    names_sep = "_(?=mean|sd|n$)",
    values_to = "value"
  ) %>%
  pivot_wider(
    names_from = stat,
    values_from = value
  ) %>%
  # Calculate standard error
  mutate(
    se = sd / sqrt(n),
    se = ifelse(is.na(se) | is.infinite(se), 0, se),
    probability = mean
  ) %>%
  select(-mean)

prompt0_negation_data <- filtered_data %>%
  filter(prompt == 0, assertion_negation == "negation") %>%
  filter(who %in% c("human", "llm", "you")) %>%
  group_by(who) %>%
  summarise(
    # Means
    prob_true_mean = mean(prob_true, na.rm = TRUE),
    prob_lr_mean = mean(prob_lr, na.rm = TRUE),
    prob_mm_mean = mean(prob_mm, na.rm = TRUE),
    prob_ttpd_mean = mean(prob_ttpd, na.rm = TRUE),
    # Standard deviations
    prob_true_sd = sd(prob_true, na.rm = TRUE),
    prob_lr_sd = sd(prob_lr, na.rm = TRUE),
    prob_mm_sd = sd(prob_mm, na.rm = TRUE),
    prob_ttpd_sd = sd(prob_ttpd, na.rm = TRUE),
    # Counts
    prob_true_n = sum(!is.na(prob_true)),
    prob_lr_n = sum(!is.na(prob_lr)),
    prob_mm_n = sum(!is.na(prob_mm)),
    prob_ttpd_n = sum(!is.na(prob_ttpd)),
    .groups = "drop"
  ) %>%
  # Reshape to long format
  pivot_longer(
    cols = -who,
    names_to = c("probability_type", "stat"),
    names_sep = "_(?=mean|sd|n$)",
    values_to = "value"
  ) %>%
  pivot_wider(
    names_from = stat,
    values_from = value
  ) %>%
  # Calculate standard error
  mutate(
    se = sd / sqrt(n),
    se = ifelse(is.na(se) | is.infinite(se), 0, se),
    probability = mean
  ) %>%
  select(-mean)

prompt1_assertion_data <- filtered_data %>%
  filter(prompt == 1, assertion_negation == "assertion") %>%
  filter(who %in% c("human", "llm", "you")) %>%
  group_by(who) %>%
  summarise(
    # Means
    prob_true_mean = mean(prob_true, na.rm = TRUE),
    prob_lr_mean = mean(prob_lr, na.rm = TRUE),
    prob_mm_mean = mean(prob_mm, na.rm = TRUE),
    prob_ttpd_mean = mean(prob_ttpd, na.rm = TRUE),
    # Standard deviations
    prob_true_sd = sd(prob_true, na.rm = TRUE),
    prob_lr_sd = sd(prob_lr, na.rm = TRUE),
    prob_mm_sd = sd(prob_mm, na.rm = TRUE),
    prob_ttpd_sd = sd(prob_ttpd, na.rm = TRUE),
    # Counts
    prob_true_n = sum(!is.na(prob_true)),
    prob_lr_n = sum(!is.na(prob_lr)),
    prob_mm_n = sum(!is.na(prob_mm)),
    prob_ttpd_n = sum(!is.na(prob_ttpd)),
    .groups = "drop"
  ) %>%
  # Reshape to long format
  pivot_longer(
    cols = -who,
    names_to = c("probability_type", "stat"),
    names_sep = "_(?=mean|sd|n$)",
    values_to = "value"
  ) %>%
  pivot_wider(
    names_from = stat,
    values_from = value
  ) %>%
  # Calculate standard error
  mutate(
    se = sd / sqrt(n),
    se = ifelse(is.na(se) | is.infinite(se), 0, se),
    probability = mean
  ) %>%
  select(-mean)

prompt1_negation_data <- filtered_data %>%
  filter(prompt == 1, assertion_negation == "negation") %>%
  filter(who %in% c("human", "llm", "you")) %>%
  group_by(who) %>%
  summarise(
    # Means
    prob_true_mean = mean(prob_true, na.rm = TRUE),
    prob_lr_mean = mean(prob_lr, na.rm = TRUE),
    prob_mm_mean = mean(prob_mm, na.rm = TRUE),
    prob_ttpd_mean = mean(prob_ttpd, na.rm = TRUE),
    # Standard deviations
    prob_true_sd = sd(prob_true, na.rm = TRUE),
    prob_lr_sd = sd(prob_lr, na.rm = TRUE),
    prob_mm_sd = sd(prob_mm, na.rm = TRUE),
    prob_ttpd_sd = sd(prob_ttpd, na.rm = TRUE),
    # Counts
    prob_true_n = sum(!is.na(prob_true)),
    prob_lr_n = sum(!is.na(prob_lr)),
    prob_mm_n = sum(!is.na(prob_mm)),
    prob_ttpd_n = sum(!is.na(prob_ttpd)),
    .groups = "drop"
  ) %>%
  # Reshape to long format
  pivot_longer(
    cols = -who,
    names_to = c("probability_type", "stat"),
    names_sep = "_(?=mean|sd|n$)",
    values_to = "value"
  ) %>%
  pivot_wider(
    names_from = stat,
    values_from = value
  ) %>%
  # Calculate standard error
  mutate(
    se = sd / sqrt(n),
    se = ifelse(is.na(se) | is.infinite(se), 0, se),
    probability = mean
  ) %>%
  select(-mean)

prompt2_assertion_data <- filtered_data %>%
  filter(prompt == 2, assertion_negation == "assertion") %>%
  filter(who %in% c("human", "llm", "you")) %>%
  group_by(who) %>%
  summarise(
    # Means
    prob_true_mean = mean(prob_true, na.rm = TRUE),
    prob_lr_mean = mean(prob_lr, na.rm = TRUE),
    prob_mm_mean = mean(prob_mm, na.rm = TRUE),
    prob_ttpd_mean = mean(prob_ttpd, na.rm = TRUE),
    # Standard deviations
    prob_true_sd = sd(prob_true, na.rm = TRUE),
    prob_lr_sd = sd(prob_lr, na.rm = TRUE),
    prob_mm_sd = sd(prob_mm, na.rm = TRUE),
    prob_ttpd_sd = sd(prob_ttpd, na.rm = TRUE),
    # Counts
    prob_true_n = sum(!is.na(prob_true)),
    prob_lr_n = sum(!is.na(prob_lr)),
    prob_mm_n = sum(!is.na(prob_mm)),
    prob_ttpd_n = sum(!is.na(prob_ttpd)),
    .groups = "drop"
  ) %>%
  # Reshape to long format
  pivot_longer(
    cols = -who,
    names_to = c("probability_type", "stat"),
    names_sep = "_(?=mean|sd|n$)",
    values_to = "value"
  ) %>%
  pivot_wider(
    names_from = stat,
    values_from = value
  ) %>%
  # Calculate standard error
  mutate(
    se = sd / sqrt(n),
    se = ifelse(is.na(se) | is.infinite(se), 0, se),
    probability = mean
  ) %>%
  select(-mean)

prompt2_negation_data <- filtered_data %>%
  filter(prompt == 2, assertion_negation == "negation") %>%
  filter(who %in% c("human", "llm", "you")) %>%
  group_by(who) %>%
  summarise(
    # Means
    prob_true_mean = mean(prob_true, na.rm = TRUE),
    prob_lr_mean = mean(prob_lr, na.rm = TRUE),
    prob_mm_mean = mean(prob_mm, na.rm = TRUE),
    prob_ttpd_mean = mean(prob_ttpd, na.rm = TRUE),
    # Standard deviations
    prob_true_sd = sd(prob_true, na.rm = TRUE),
    prob_lr_sd = sd(prob_lr, na.rm = TRUE),
    prob_mm_sd = sd(prob_mm, na.rm = TRUE),
    prob_ttpd_sd = sd(prob_ttpd, na.rm = TRUE),
    # Counts
    prob_true_n = sum(!is.na(prob_true)),
    prob_lr_n = sum(!is.na(prob_lr)),
    prob_mm_n = sum(!is.na(prob_mm)),
    prob_ttpd_n = sum(!is.na(prob_ttpd)),
    .groups = "drop"
  ) %>%
  # Reshape to long format
  pivot_longer(
    cols = -who,
    names_to = c("probability_type", "stat"),
    names_sep = "_(?=mean|sd|n$)",
    values_to = "value"
  ) %>%
  pivot_wider(
    names_from = stat,
    values_from = value
  ) %>%
  # Calculate standard error
  mutate(
    se = sd / sqrt(n),
    se = ifelse(is.na(se) | is.infinite(se), 0, se),
    probability = mean
  ) %>%
  select(-mean)

# Create factor levels for proper ordering
all_datasets <- list(prompt0_assertion_data, prompt0_negation_data, 
                     prompt1_assertion_data, prompt1_negation_data,
                     prompt2_assertion_data, prompt2_negation_data)

for(i in 1:length(all_datasets)) {
  all_datasets[[i]]$who <- factor(all_datasets[[i]]$who, levels = c("human", "llm", "you"))
  all_datasets[[i]]$probability_type <- factor(all_datasets[[i]]$probability_type, 
                                               levels = c("prob_true", "prob_lr", "prob_mm", "prob_ttpd"))
}

# Create entity-probability combination for color mapping
for(i in 1:length(all_datasets)) {
  all_datasets[[i]]$entity_prob_combo <- paste(all_datasets[[i]]$who, all_datasets[[i]]$probability_type, sep = "_")
}

# Create factor with explicit ordering for correct bar positions
combo_levels <- c(
  "human_prob_true", "human_prob_lr", "human_prob_mm", "human_prob_ttpd",
  "llm_prob_true", "llm_prob_lr", "llm_prob_mm", "llm_prob_ttpd",
  "you_prob_true", "you_prob_lr", "you_prob_mm", "you_prob_ttpd"
)

for(i in 1:length(all_datasets)) {
  all_datasets[[i]]$entity_prob_combo <- factor(all_datasets[[i]]$entity_prob_combo, levels = combo_levels)
}

# Reassign back to individual variables
prompt0_assertion_data <- all_datasets[[1]]
prompt0_negation_data <- all_datasets[[2]]
prompt1_assertion_data <- all_datasets[[3]]
prompt1_negation_data <- all_datasets[[4]]
prompt2_assertion_data <- all_datasets[[5]]
prompt2_negation_data <- all_datasets[[6]]

# Define entity-specific color families (darkest to lightest)
entity_colors <- c(
  # Human - blue shades (darkest to lightest)
  "human_prob_true" = "#191970",
  "human_prob_lr" = "#000080", 
  "human_prob_mm" = "#4169E1",
  "human_prob_ttpd" = "#ADD8E6",
  # LLM - red shades (darkest to lightest)
  "llm_prob_true" = "#8B0000",
  "llm_prob_lr" = "#DC143C",
  "llm_prob_mm" = "#FF6347",
  "llm_prob_ttpd" = "#FFA07A",
  # You - green shades (darkest to lightest)
  "you_prob_true" = "#006400",
  "you_prob_lr" = "#228B22",
  "you_prob_mm" = "#32CD32",
  "you_prob_ttpd" = "#90EE90"
)

#=================================
# 4. Create bar plots
#=================================

# Create label positions for angled text
label_data <- data.frame(
  who = rep(c("human", "llm", "you"), each = 4),
  prob_type = rep(c("prob_true", "prob_lr", "prob_mm", "prob_ttpd"), 3),
  label = rep(c("Continuation", "LR", "MM", "TTPD"), 3),
  stringsAsFactors = FALSE
)

# Calculate positions to match the dodged bar positions
entity_pos <- c(1, 2, 3)  # human=1, llm=2, you=3
dodge_offsets <- c(-0.3, -0.1, 0.1, 0.3)  # for prob_true, prob_lr, prob_mm, prob_ttpd

label_data$x_pos <- rep(entity_pos, each = 4) + rep(dodge_offsets, 3)
label_data$y_pos <- -0.05

# Function to create bar plot
create_bar_plot <- function(data, plot_title) {
  ggplot(data, aes(x = who, y = probability, fill = entity_prob_combo)) +
    geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
    geom_errorbar(aes(ymin = probability - se, ymax = probability + se), 
                  position = position_dodge(width = 0.8), 
                  width = 0.2, 
                  color = "black", 
                  size = 0.5) +
    scale_fill_manual(values = entity_colors) +
    scale_x_discrete(labels = c("human" = "Human", "llm" = "LLM", "you" = "You")) +
    guides(fill = "none") +
    labs(
      title = plot_title,
      x = "Entity type",
      y = "Mean probability"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10),
      axis.line.y = element_blank(),
      axis.ticks = element_line(color = "black", size = 0.5),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA)
    ) +
    annotate("text", 
             x = label_data$x_pos, 
             y = label_data$y_pos, 
             label = label_data$label,
             angle = 45, 
             hjust = 1, 
             vjust = 0.5, 
             size = 4) +
    annotate("segment", x = 0.5, xend = 0.5, y = 0, yend = 1, color = "black", size = 0.5) +
    coord_cartesian(ylim = c(-0.1, 1), clip = "off") +
    theme(plot.margin = margin(5.5, 5.5, 25, 5.5, "pt"))
}

# Create individual plots for assertion and negation
bar_plot_prompt0_assertion <- create_bar_plot(prompt0_assertion_data, "Panel A: Standard - Assertions")
bar_plot_prompt1_assertion <- create_bar_plot(prompt1_assertion_data, "Panel B: Force True - Assertions")  
bar_plot_prompt2_assertion <- create_bar_plot(prompt2_assertion_data, "Panel C: Force False - Assertions")

bar_plot_prompt0_negation <- create_bar_plot(prompt0_negation_data, "Panel D: Standard - Negations")
bar_plot_prompt1_negation <- create_bar_plot(prompt1_negation_data, "Panel E: Force True - Negations")  
bar_plot_prompt2_negation <- create_bar_plot(prompt2_negation_data, "Panel F: Force False - Negations")

#=================================
# 5. Combine and save output
#=================================

# Combine plots in 2x3 layout (assertions top row, negations bottom row)
combined_plot <- (bar_plot_prompt0_assertion + bar_plot_prompt1_assertion + bar_plot_prompt2_assertion) /
                 (bar_plot_prompt0_negation + bar_plot_prompt1_negation + bar_plot_prompt2_negation) +
  plot_layout(nrow = 2)

# Save the combined figure
ggsave("results/figure3.png", combined_plot, width = 18, height = 12, dpi = 300)

# Display plot
combined_plot