# Load libraries
library(tidyverse)
library(h2o)
library(rsample)
library(recipes)
library(PerformanceAnalytics)
library(cowplot)
library(glue)

product_backorders <- read_csv("product_backorders.csv")

glimpse(product_backorders)

colSums(is.na(product_backorders))

set.seed(1113)
split_obj <- initial_split(product_backorders, prop = 0.85)

train_tbl <- training(split_obj)
test_tbl  <- testing(split_obj)

recipe_obj <- recipe(went_on_backorder ~ ., data = train_tbl) %>%
  step_zv(all_predictors()) %>%
  step_YeoJohnson(all_numeric_predictors()) %>%
  step_center(all_numeric_predictors()) %>%
  step_scale(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  prep()

train_processed_tbl <- bake(recipe_obj, new_data = train_tbl)
test_processed_tbl  <- bake(recipe_obj, new_data = test_tbl)

h2o.init()

train_h2o <- as.h2o(train_processed_tbl)
test_h2o  <- as.h2o(test_processed_tbl)

split_h2o <- h2o.splitFrame(train_h2o, ratios = 0.85, seed = 1234)
train_h2o <- split_h2o[[1]]
valid_h2o <- split_h2o[[2]]

y <- "went_on_backorder"
x <- setdiff(names(train_h2o), y)

automl_models_h2o <- h2o.automl(
  x = x,
  y = y,
  training_frame = train_h2o,
  validation_frame = valid_h2o,
  leaderboard_frame = test_h2o,
  max_runtime_secs = 30,
  nfolds = 5
)

automl_models_h2o@leaderboard

best_model <- automl_models_h2o@leader
predictions <- h2o.predict(best_model, newdata = test_h2o)
predictions_tbl <- as_tibble(predictions)
head(predictions_tbl)

model_path <- h2o.saveModel(best_model, path = "h2o_models/", force = TRUE)
model_path

leaderboard_tbl <- automl_models_h2o@leaderboard %>%
  as_tibble() %>%
  select(-c(aucpr, mean_per_class_error, rmse, mse)) %>% 
  mutate(model_type = str_extract(model_id, "[^_]+")) %>%
  slice(1:15) %>% 
  rownames_to_column(var = "rowname") %>%
  mutate(
    model_id   = as_factor(model_id) %>% reorder(auc),
    model_type = as.factor(model_type)
  ) %>% 
  pivot_longer(cols = -c(model_id, model_type, rowname), 
               names_to = "key", 
               values_to = "value", 
               names_transform = list(key = forcats::fct_inorder)
  ) %>% 
  mutate(model_id = paste0(rowname, ". ", model_id) %>% as_factor() %>% fct_rev())

leaderboard_tbl %>%
  ggplot(aes(value, model_id, color = model_type)) +
  geom_point(size = 3) +
  geom_label(aes(label = round(value, 2), hjust = "inward")) +
  
  facet_wrap(~ key, scales = "free_x") +
  labs(title = "Leaderboard Metrics",
       subtitle = paste0("Ordered by: ", "auc"),
       y = "Model Postion, Model ID", x = "") + 
  theme(legend.position = "bottom")


# --- Challenge 5 ---
deeplearning_model_id <- automl_models_h2o@leaderboard %>% 
       as_tibble() %>% 
       select(-c(mean_per_class_error, rmse, mse)) %>%
        filter(str_detect(model_id, "DeepLearning")) %>%
        slice(1) %>%
        pull(model_id) 
  
dl_model <- h2o.getModel(deeplearning_model_id)
dl_performance <- h2o.performance(dl_model, newdata = as.h2o(test_processed_tbl))

dl_model_grid_1 <- h2o.grid(
  algorithm = "deeplearning",
  grid_id = "deeplearning_grid_01",
  x = x,
  y = y,
  training_frame   = train_h2o,
  validation_frame = valid_h2o,
  nfolds = 5,
  hyper_params = list(
    hidden = list(c(10, 10, 10), c(50, 20, 10), c(20, 20, 20)),
    epochs = c(10, 50, 100)
  )
)

theme_new <- theme(
  legend.position  = "bottom",
  legend.key       = element_blank(),,
  panel.background = element_rect(fill   = "transparent"),
  panel.border     = element_rect(color = "black", fill = NA, size = 0.5),
  panel.grid.major = element_line(color = "grey", size = 0.333)
) 

tuned_dl_model <- h2o.getModel("deeplearning_grid_01_model_1")

performance_h2o <- h2o.performance(tuned_dl_model, newdata = as.h2o(test_processed_tbl))


performance_tbl <- performance_h2o %>%
  h2o.metric() %>%
  as.tibble() 

performance_tbl %>% 
  glimpse()

performance_tbl %>%
  filter(f1 == max(f1))

h2o.no_progress()
p0 <- performance_tbl %>%
  ggplot(aes(x = threshold)) +
  geom_line(aes(y = precision), color = "blue", size = 1) +
  geom_line(aes(y = recall), color = "red", size = 1) +
  
  # Insert line where precision and recall are harmonically optimized
  geom_vline(xintercept = h2o.find_threshold_by_max_metric(performance_h2o, "f1")) +
  labs(title = "Precision vs Recall", y = "value") +
  theme_new


load_model_performance_metrics <- function(model_h2o, test_tbl) {
  perf_h2o  <- h2o.performance(model_h2o, newdata = as.h2o(test_tbl)) 
  
  perf_h2o %>%
    h2o.metric() %>%
    as_tibble() %>%
    mutate(auc = h2o.auc(perf_h2o)) %>%
    select(tpr, fpr, auc, precision, recall)
  
}

# create tabbl from dl_model and tuned_dl_model
model_metrics_tbl <- tibble(
  model_id = c("Deep Learning", "Tuned Deep Learning"),
  metrics  = map(c(dl_model, tuned_dl_model), load_model_performance_metrics, test_processed_tbl)
) %>% unnest(cols = metrics)

p1 <- model_metrics_tbl %>%
  mutate(
    auc  = auc %>% round(3) %>% as.character() %>% as_factor()
  ) %>%
  ggplot(aes(fpr, tpr, color = model_id, linetype = auc)) +
  geom_line(size = 1) +
  
  # just for demonstration purposes
  geom_abline(color = "red", linetype = "dotted") +
  
  theme_new +
  theme(
    legend.direction = "vertical",
  ) +
  labs(
    title = "ROC Plot",
    subtitle = "Performance of Models Pre and Post Grid Tuning"
  )


p2 <- model_metrics_tbl %>%
  mutate(
    auc  = auc %>% round(3) %>% as.character() %>% as_factor()
  ) %>%
  ggplot(aes(recall, precision, color = model_id, linetype = auc)) +
  geom_line(size = 1) +
  theme_new + 
  theme(
    legend.direction = "vertical",
  ) +
  labs(
    title = "Precision vs Recall Plot",
    subtitle = "Performance of Models Pre and Post Grid Tuning"
  )

gain_lift_tbl <- performance_h2o %>%
  h2o.gainsLift() %>%
  as.tibble()

## Gain Chart

gain_transformed_tbl <- gain_lift_tbl %>% 
  select(group, cumulative_data_fraction, cumulative_capture_rate, cumulative_lift) %>%
  select(-contains("lift")) %>%
  mutate(baseline = cumulative_data_fraction) %>%
  rename(gain     = cumulative_capture_rate) %>%
  # prepare the data for the plotting (for the color and group aesthetics)
  pivot_longer(cols = c(gain, baseline), values_to = "value", names_to = "key")

p3 <- gain_transformed_tbl %>%
  ggplot(aes(x = cumulative_data_fraction, y = value, color = key)) +
  geom_line(size = 1.5) +
  labs(
    title = "Gain Chart",
    x = "Cumulative Data Fraction",
    y = "Gain"
  ) +
  theme_new

## Lift Plot

lift_transformed_tbl <- gain_lift_tbl %>% 
  select(group, cumulative_data_fraction, cumulative_capture_rate, cumulative_lift) %>%
  select(-contains("capture")) %>%
  mutate(baseline = 1) %>%
  rename(lift = cumulative_lift) %>%
  pivot_longer(cols = c(lift, baseline), values_to = "value", names_to = "key")

p4 <- lift_transformed_tbl %>%
  ggplot(aes(x = cumulative_data_fraction, y = value, color = key)) +
  geom_line(size = 1.5) +
  labs(
    title = "Lift Chart",
    x = "Cumulative Data Fraction",
    y = "Lift"
  ) +
  theme_new

#--------------------------------
  # Combine using cowplot
  
  # cowplot::get_legend extracts a legend from a ggplot object
  p_legend <- get_legend(p1)
  # Remove legend from p1
  p1 <- p1 + theme(legend.position = "none")
  
  # cowplot::plt_grid() combines multiple ggplots into a single cowplot object
  p <- cowplot::plot_grid(p1, p2, p3, p4, ncol = 2)
  
  # cowplot::ggdraw() sets up a drawing layer
  p_title <- ggdraw() + 
    
    # cowplot::draw_label() draws text on a ggdraw layer / ggplot object
    draw_label("H2O Model Metrics", size = 18, fontface = "bold", 
               color = "#2C3E50")
  
  p_subtitle <- ggdraw() + 
    draw_label(glue("Ordered by {toupper(order_by)}"), size = 10,  
               color = "#2C3E50")
  
  # Combine everything
  plot_grid(p_title, p_subtitle, p, p_legend, 
                   
                   # Adjust the relative spacing, so that the legends always fits
                   ncol = 1, rel_heights = c(0.05, 0.05, 1, 0.05 * max_models))
  
  h2o.show_progress()
  

