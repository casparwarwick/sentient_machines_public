##########################################
# SENTIENCE PROJECT
##########################################
# Figure 1: Qwen32b Results
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
# 3. Data preparation for plotting
#=================================

# Aggregate data by entity and probability type with standard errors
plot_data <- filtered_data %>%
  filter(who %in% c("human", "llm", "you")) %>%
  group_by(who, assertion_negation) %>%
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
    cols = -c(who, assertion_negation),
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
  select(-mean) %>%
  ungroup()

# Create factor levels for proper ordering (prob_true first, prob_ttpd last)
plot_data$who <- factor(plot_data$who, levels = c("human", "llm", "you"))
plot_data$probability_type <- factor(plot_data$probability_type, levels = c("prob_true", "prob_lr", "prob_mm", "prob_ttpd"))

# Create entity-probability combination for color mapping
plot_data$entity_prob_combo <- paste(plot_data$who, plot_data$probability_type, sep = "_")

# Create factor with explicit ordering for correct bar positions
combo_levels <- c(
  "human_prob_true", "human_prob_lr", "human_prob_mm", "human_prob_ttpd",
  "llm_prob_true", "llm_prob_lr", "llm_prob_mm", "llm_prob_ttpd",
  "you_prob_true", "you_prob_lr", "you_prob_mm", "you_prob_ttpd"
)
plot_data$entity_prob_combo <- factor(plot_data$entity_prob_combo, levels = combo_levels)

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
# 4. Create assertion figure
#=================================

assertion_data <- plot_data %>% filter(assertion_negation == "assertion")

assertion_plot <- ggplot(assertion_data, aes(x = who, y = probability, fill = entity_prob_combo)) +
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
    title = "Panel A: Assertions",
    x = "Entity type",
    y = "Mean probability"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5)
  )
assertion_plot

#=================================
# 5. Create negation figure
#=================================

negation_data <- plot_data %>% filter(assertion_negation == "negation")

negation_plot <- ggplot(negation_data, aes(x = who, y = probability, fill = entity_prob_combo)) +
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
    title = "Panel B: Negations",
    x = "Entity type",
    y = "Mean probability"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5)
  )
negation_plot

#=================================
# 6. Combine and save figure
#=================================

# Add text labels for probability types
# Create label positions that match the actual bar positions
label_data <- data.frame(
  who = rep(c("human", "llm", "you"), each = 4),
  prob_type = rep(c("prob_true", "prob_lr", "prob_mm", "prob_ttpd"), 3),
  label = rep(c("Continuation", "LR", "MM", "TTPD"), 3),
  stringsAsFactors = FALSE
)

# Calculate positions to match the dodged bar positions
# Each entity gets positions at -0.3, -0.1, 0.1, 0.3 relative to entity center
entity_pos <- c(1, 2, 3)  # human=1, llm=2, you=3
dodge_offsets <- c(-0.3, -0.1, 0.1, 0.3)  # for prob_true, prob_lr, prob_mm, prob_ttpd

label_data$x_pos <- rep(entity_pos, each = 4) + rep(dodge_offsets, 3)
label_data$y_pos <- -0.05

# Update assertion plot with labels
assertion_plot <- assertion_plot +
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
  theme(
    plot.margin = margin(5.5, 5.5, 25, 5.5, "pt"),
    plot.title = element_text(hjust = 0.5, size = 14),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    axis.line.y = element_blank(),
    axis.ticks = element_line(color = "black", size = 0.5)
  )

# Update negation plot with labels  
negation_plot <- negation_plot +
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
  theme(
    plot.margin = margin(5.5, 5.5, 25, 5.5, "pt"),
    plot.title = element_text(hjust = 0.5, size = 14),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10),
    axis.line.y = element_blank(),
    axis.ticks = element_line(color = "black", size = 0.5)
  )

# Combine plots without legend
combined_plot <- assertion_plot + negation_plot +
  plot_layout(ncol = 2)

# Save the combined figure
combined_plot
ggsave("results/figure1.png", combined_plot, width = 12, height = 6, dpi = 300)