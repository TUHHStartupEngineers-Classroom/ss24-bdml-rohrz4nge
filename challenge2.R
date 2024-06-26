library(recipes)
library(workflows)
library(dplyr)
library(stringr)

bike_features_tbl <- readRDS("bike_features_tbl.rds")  %>% 
  select(model:url, `Rear Derailleur`, `Shift Lever`) %>% 
  mutate(derailleur = `Rear Derailleur`) %>%
  # Remove original columns  
  select(-c(`Rear Derailleur`, `Shift Lever`)) %>% 
  # Set all NAs to 0
  mutate_if(is.numeric, ~replace(., is.na(.), 0)) %>%
  mutate(id = row_number())

bike_features_tbl %>% distinct(category_2)

# run both following commands at the same time
set.seed(seed = 28)
split_obj <- rsample::initial_split(bike_features_tbl, prop   = 0.80, 
                                    strata = "category_2")

# Check if testing contains all category_2 values
split_obj %>% training() %>% distinct(category_2)
split_obj %>% testing() %>% distinct(category_2)

# Assign training and test data
train_tbl <- training(split_obj)
test_tbl  <- testing(split_obj)

# We have to remove spaces and dashes from the column names
train_data <- train_tbl %>% set_names(str_replace_all(names(train_tbl), " |-", "_"))
test_data  <- test_tbl  %>% set_names(str_replace_all(names(test_tbl),  " |-", "_"))
recipe_obj <-
  recipe(price ~ ., data = bike_features_tbl) %>%
  update_role(id, new_role="ID") %>%
  step_rm(url) %>%
  step_dummy(derailleur, one_hot = TRUE) %>%
  step_zv(all_predictors()) # %>%
recipe_obj %>% prep() %>% bake(new_data = train_data) %>% glimpse()
lr_mod <- 
  linear_reg() %>% 
  set_engine("glm")

bike_wflow <- 
  workflow() %>% 
  add_model(lr_mod) %>% 
  add_recipe(recipe_obj)

bike_fit <- 
  bike_wflow %>% 
  fit(data = train_data)

# 3.1.3 Function to Calculate Metrics ----

# Generalized into a function
calc_metrics <- function(model, new_data = test_tbl) {
  
  model %>%
    predict(new_data = new_data) %>%
    
    bind_cols(new_data %>% select(price)) %>%
    yardstick::metrics(truth = price, estimate = .pred)
  
}

bike_fit %>% calc_metrics(test_tbl)