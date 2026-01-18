##########################################
# SENTIENCE PROJECT
##########################################
# Training Performance for best layer
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

# Load combined training results
data <- read_csv("datasets/combined_results_training.csv", show_col_types = FALSE)

# Filter to test split data (20% holdout) and remove NA splits
filtered_data <- data %>%
  filter(
    split == "test",
    !is.na(prob_lr),
    !is.na(prob_mm),
    !is.na(prob_ttpd)
  )

#=================================
# 3. Data preparation for plotting
#=================================

# Calculate accuracy for each classifier and add assertion/negation labels
accuracy_data <- filtered_data %>%
  mutate(
    # Convert polarity to assertion/negation labels
    assertion_negation = ifelse(polarity == 1, "assertion", "negation"),
    # Calculate accuracy for each classifier (predicted vs actual)
    acc_lr = as.numeric((prob_lr > 0.5) == (label == 1)),
    acc_mm = as.numeric((prob_mm > 0.5) == (label == 1)),
    acc_ttpd = as.numeric((prob_ttpd > 0.5) == (label == 1))
  )

# Aggregate accuracy data by prompt condition and assertion/negation with standard errors
plot_data <- accuracy_data %>%
  group_by(prompt, assertion_negation) %>%
  summarise(
    # Means
    acc_lr_mean = mean(acc_lr, na.rm = TRUE),
    acc_mm_mean = mean(acc_mm, na.rm = TRUE),
    acc_ttpd_mean = mean(acc_ttpd, na.rm = TRUE),
    # Standard deviations
    acc_lr_sd = sd(acc_lr, na.rm = TRUE),
    acc_mm_sd = sd(acc_mm, na.rm = TRUE),
    acc_ttpd_sd = sd(acc_ttpd, na.rm = TRUE),
    # Counts
    acc_lr_n = sum(!is.na(acc_lr)),
    acc_mm_n = sum(!is.na(acc_mm)),
    acc_ttpd_n = sum(!is.na(acc_ttpd)),
    .groups = "drop"
  ) %>%
  # Reshape to long format
  pivot_longer(
    cols = -c(prompt, assertion_negation),
    names_to = c("classifier_type", "stat"),
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
    accuracy = mean
  ) %>%
  select(-mean) %>%
  ungroup()

# Create factor levels for proper ordering
plot_data$classifier_type <- factor(plot_data$classifier_type, levels = c("acc_lr", "acc_mm", "acc_ttpd"))
plot_data$prompt <- factor(plot_data$prompt, levels = c("0", "1", "2"))

# Create prompt-classifier combination for color mapping
plot_data$prompt_classifier_combo <- paste(plot_data$prompt, plot_data$classifier_type, sep = "_")

# Create factor with explicit ordering for correct bar positions
combo_levels <- c(
  "0_acc_lr", "1_acc_lr", "2_acc_lr",
  "0_acc_mm", "1_acc_mm", "2_acc_mm", 
  "0_acc_ttpd", "1_acc_ttpd", "2_acc_ttpd"
)
plot_data$prompt_classifier_combo <- factor(plot_data$prompt_classifier_combo, levels = combo_levels)

# Define prompt-specific colors (Grey, Yellow, Purple)
prompt_colors <- c(
  # Standard (prompt=0) - grey shades
  "0_acc_lr" = "#696969",
  "0_acc_mm" = "#808080", 
  "0_acc_ttpd" = "#A9A9A9",
  # Force True (prompt=1) - yellow shades
  "1_acc_lr" = "#DAA520",
  "1_acc_mm" = "#FFD700",
  "1_acc_ttpd" = "#FFFF99",
  # Force False (prompt=2) - purple shades  
  "2_acc_lr" = "#6A0DAD",
  "2_acc_mm" = "#8A2BE2",
  "2_acc_ttpd" = "#DA70D6"
)

#=================================
# 4. Create assertion figure
#=================================

assertion_data <- plot_data %>% filter(assertion_negation == "assertion")

assertion_plot <- ggplot(assertion_data, aes(x = classifier_type, y = accuracy, fill = prompt_classifier_combo)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(aes(ymin = accuracy - se, ymax = accuracy + se), 
                position = position_dodge(width = 0.8), 
                width = 0.2, 
                color = "black", 
                size = 0.5) +
  scale_fill_manual(values = prompt_colors) +
  scale_x_discrete(labels = c("acc_lr" = "LR", "acc_mm" = "MM", "acc_ttpd" = "TTPD")) +
  guides(fill = "none") +
  labs(
    title = "Panel A: Assertions",
    x = "Classifier type",
    y = "Accuracy"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

#=================================
# 5. Create negation figure
#=================================

negation_data <- plot_data %>% filter(assertion_negation == "negation")

negation_plot <- ggplot(negation_data, aes(x = classifier_type, y = accuracy, fill = prompt_classifier_combo)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7) +
  geom_errorbar(aes(ymin = accuracy - se, ymax = accuracy + se), 
                position = position_dodge(width = 0.8), 
                width = 0.2, 
                color = "black", 
                size = 0.5) +
  scale_fill_manual(values = prompt_colors) +
  scale_x_discrete(labels = c("acc_lr" = "LR", "acc_mm" = "MM", "acc_ttpd" = "TTPD")) +
  guides(fill = "none") +
  labs(
    title = "Panel B: Negations",
    x = "Classifier type",
    y = "Accuracy"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

#=================================
# 6. Combine and save figure
#=================================

# Add text labels for prompt conditions
# Create label positions that match the actual bar positions
label_data <- data.frame(
  classifier_type = rep(c("acc_lr", "acc_mm", "acc_ttpd"), each = 3),
  prompt_type = rep(c("0", "1", "2"), 3),
  label = rep(c("Standard", "Force True", "Force False"), 3),
  stringsAsFactors = FALSE
)

# Calculate positions to match the dodged bar positions
# Each classifier gets positions at -0.25, 0, 0.25 relative to classifier center
classifier_pos <- c(1, 2, 3)  # lr=1, mm=2, ttpd=3
dodge_offsets <- c(-0.25, 0, 0.25)  # for prompt 0, 1, 2

label_data$x_pos <- rep(classifier_pos, each = 3) + rep(dodge_offsets, 3)
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
           size = 3.5) +
  coord_cartesian(ylim = c(-0.1, 1), clip = "off") +
  theme(
    plot.margin = margin(5.5, 5.5, 25, 5.5, "pt"),
    plot.title = element_text(hjust = 0.5, size = 14),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10)
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
           size = 3.5) +
  coord_cartesian(ylim = c(-0.1, 1), clip = "off") +
  theme(
    plot.margin = margin(5.5, 5.5, 25, 5.5, "pt"),
    plot.title = element_text(hjust = 0.5, size = 14),
    axis.title = element_text(size = 12),
    axis.text = element_text(size = 10)
  )

# Combine plots without legend
combined_plot <- assertion_plot + negation_plot +
  plot_layout(ncol = 2)

# Save the combined figure
combined_plot
ggsave("results/figure_training_performance.png", combined_plot, width = 12, height = 6, dpi = 300)