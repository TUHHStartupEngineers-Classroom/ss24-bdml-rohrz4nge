# Load necessary libraries
library(tidyverse)
library(tidyquant)
library(broom)
library(umap)

sp_500_prices_tbl <- read_rds("challenge1/sp_500_prices_tbl.rds")

sp_500_daily_returns_tbl <- sp_500_prices_tbl %>%
  select(symbol, date, adjusted) %>%
  filter(date >= as.Date("2018-01-01")) %>%
  group_by(symbol) %>%
  mutate(lag_adjusted = lag(adjusted)) %>%
  filter(!is.na(lag_adjusted)) %>%
  mutate(pct_return = (adjusted - lag_adjusted) / lag_adjusted) %>%
  select(symbol, date, pct_return) %>%
  ungroup()


sp_500_daily_returns_tbl <- read_rds("challenge1/sp_500_daily_returns_tbl.rds")

stock_date_matrix_tbl <- sp_500_daily_returns_tbl %>%
  spread(key = date, value = pct_return, fill = 0)

stock_date_matrix_tbl <- read_rds("challenge1/stock_date_matrix_tbl.rds")
stock_date_matrix <- stock_date_matrix_tbl %>%
  select(-symbol)

kmeans_obj <- kmeans(stock_date_matrix, centers = 4, nstart = 20)

tot_withinss <- glance(kmeans_obj)$tot.withinss
tot_withinss

kmeans_mapper <- function(center = 3) {
  stock_date_matrix_tbl %>%
    select(-symbol) %>%
    kmeans(centers = center, nstart = 20)
}
k_means_mapped_tbl <- tibble(centers = 1:30) %>%
  mutate(k_means = map(centers, kmeans_mapper)) %>%
  mutate(glance = map(k_means, glance)) 

k_means_mapped_tbl %>%
  unnest(glance) %>%
  ggplot(aes(x = centers, y = tot.withinss)) +
  geom_point() +
  geom_line() +
  ggtitle("Scree Plot") +
  theme_minimal()

k_means_mapped_tbl <- read_rds("challenge1/k_means_mapped_tbl.rds")

umap_results <- stock_date_matrix_tbl %>%
  select(-symbol) %>%
  umap()

umap_results_tbl <- as_tibble(umap_results$layout) %>%
  bind_cols(stock_date_matrix_tbl %>% select(symbol))

umap_results_tbl %>%
  ggplot(aes(x = V1, y = V2)) +
  geom_point(alpha = 0.5) +
  ggtitle("UMAP Projection") +
  theme_minimal()

umap_results_tbl   <- read_rds("challenge1/umap_results_tbl.rds")
k_means_mapped_tbl <- read_rds("challenge1/k_means_mapped_tbl.rds")
sp_500_index_tbl <- read_rds("challenge1/sp_500_index_tbl.rds")
k_means_obj <- kmeans_mapper(10)
kmeans_augment <- augment(k_means_obj, stock_date_matrix_tbl)

umap_kmeans_results_tbl <- umap_results_tbl %>%
  left_join(kmeans_augment %>% select(symbol, .cluster), by = "symbol") %>%
  left_join(sp_500_index_tbl %>% select(symbol, company, sector), by = "symbol")

umap_kmeans_results_tbl %>%
  mutate(label_text = str_glue("Symbol: {symbol}
                                Cluster: {.cluster}")) %>%
  ggplot(aes(x = V1, y = V2, color = .cluster)) +
    geom_point(alpha = 0.5) +
    #ggrepel::geom_label_repel(aes(label = label_text), size = 2, fill = "#282A36") +
  scale_color_manual(values = c("#2d72d6", "#2dc6d6", "#2dd692", "#d62d2d", "#d6a72d", "#2dd6a7", "#722dd6", "#d62da7", "#d62d72", "#a7d62d")) +
  
  
  labs(title = "Company Segmentation: 2D Projection",
       subtitle = "UMAP 2D Projection with K-Means Cluster Assignment")


