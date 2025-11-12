library(ggplot2)
library(dplyr)
library(lubridate)
library(stringr)  # 仅加载 stringr
library(patchwork)
library(purrr)  # 加载purrr包
library(ggsci)
library(glue)



# Interval score function
interval_score <- function(y, lower, upper, alpha) {
  lower <- as.numeric(lower)
  upper <- as.numeric(upper)
  y <- as.numeric(y)
  alpha <- as.numeric(alpha)
  width <- upper - lower
  penalty_lower <- (2 / alpha) * (lower - y) * (y < lower)
  penalty_upper <- (2 / alpha) * (y - upper) * (y > upper)
  return(width + penalty_lower + penalty_upper)
}

# Weighted Interval Score (WIS) function
calculate_wis <- function(row) {
  intervals <- list(
    list(lower = "q_1", upper = "q_99", alpha = 0.02),  # 98% PI
    list(lower = "q_2", upper = "q_97", alpha = 0.05),  # 95% PI
    list(lower = "q_5", upper = "q_95", alpha = 0.1),   # 90% PI
    list(lower = "q_10", upper = "q_90", alpha = 0.2),  # 80% PI
    list(lower = "q_15", upper = "q_85", alpha = 0.3),  # 70% PI
    list(lower = "q_20", upper = "q_80", alpha = 0.4),  # 60% PI
    list(lower = "q_25", upper = "q_75", alpha = 0.5),  # 50% PI
    list(lower = "q_30", upper = "q_70", alpha = 0.6),  # 40% PI
    list(lower = "q_35", upper = "q_65", alpha = 0.7),  # 30% PI
    list(lower = "q_40", upper = "q_60", alpha = 0.8),  # 20% PI
    list(lower = "q_45", upper = "q_55", alpha = 0.9)   # 10% PI
  )
  y <- row[["true"]]
  y <- as.numeric(y)
  median <- y  # Assuming median is the true value as in the original code
  K <- length(intervals)  # Number of intervals
  
  # Calculate interval scores
  interval_scores <- sum(sapply(intervals, function(interval) {
    0.5 * interval$alpha * interval_score(y, row[[interval$lower]], row[[interval$upper]], interval$alpha)
  }))
  
  # Median penalty
  median_penalty <- 0.5 * abs(y - median)
  
  # WIS calculation
  wis <- (interval_scores + median_penalty) / (K + 0.5)
  return(wis)
}

# 定义计算评估指标的函数
calculate_metrics <- function(true, pred) {
  # 均方根误差 (RMSE)
  rmse <- sqrt(mean((true - pred)^2))
  
  # 平均绝对误差 (MAE)
  mae <- mean(abs(true - pred))
  
  # 平均绝对百分比误差 (MAPE)
  mape <- mean(abs((true - pred)/true)) * 100
  
  # 对称平均绝对百分比误差 (SMAPE)
  smape <- mean(2 * abs(pred - true) / (abs(true) + abs(pred) + 1e-8)) * 100
  
  # 中位数绝对误差 (Median AE)
  median_ae <- median(abs(true - pred))
  
  return(list(rmse = rmse, 
              mae = mae, 
              mape = mape, 
              smape = smape, 
              median_ae = median_ae))
}
# t =  read.csv('../res/gru_base_stage_holi.csv')
get_hk_plot_data <- function(metric_label,model_use = 'gru') {
  models <- c('constant',  
              'GRU_base', 'TCN_base','itransformer_base','Nbeats_base',
              'GRU_SIR', 'TCN_SIR','itransformer_SIR','Nbeats_SIR',
              'ensemble'
  )
  model_names <- c(
    'constant' = "Constant",
    'GRU_base' = "GRU base",
    'TCN_base' = "TCN base",
    'itransformer_base' = "itransformer base",
    'Nbeats_base' = "Nbeats base",
    'GRU_SIR' = "GRU SIRSPF-H",
    'TCN_SIR' = "TCN SIRSPF-H",
    'itransformer_SIR' = "itransformer SIRSPF-H",
    'Nbeats_SIR' = "Nbeats SIRSPF-H",
    'ensemble' = "Average Ensemble"
    
  )
  
  pred_steps <- 9
  
  # 预定义文件路径映射'../src/{model_use}_Base_stage_holi_pf_42test.csv')
  path_map <- c(
    'constant' = 'forc_baseline.csv',
    'GRU_base' = glue('../src/gru_Base_42test.csv'),
    'TCN_base' = glue('../src/tcn_Base_42test.csv'),
    'itransformer_base' = glue('../src/itransformer_Base_42test.csv'),
    'Nbeats_base' = glue('../src/Nbeats_Base_42test.csv'),
    'GRU_SIR' = glue('../src/gru_Base_stage_holi_pf_ol_42test.csv'),
    'TCN_SIR' = glue('../src/tcn_Base_stage_holi_pf_ol_42test.csv'),
    'itransformer_SIR' = glue('../src/itransformer_Base_stage_holi_pf_ol_42test.csv'),
    'Nbeats_SIR' = glue('../src/Nbeats_Base_stage_holi_pf_ol_42test.csv'),
    'ensemble' = glue('ensemble_model_with_intervals.csv')
  )
  
  # 统一处理数据的函数
  process_data <- function(path) {
    res <- read.csv(path) %>%
      {if("mode" %in% colnames(.)) filter(., mode == "train") else .} %>%
      mutate(date = as.Date(date),
             date_origin = date - week_ahead * 7)
    
    invalid_dates <- res %>%
      group_by(date) %>%
      filter(any(true <= 1)) %>%
      pull(date) %>%
      unique()
    
    res %>%
      filter(!(date_origin %in% invalid_dates),
             date_origin >= as.Date('2023-11-26')) %>%
      na.omit()
  }
  
  # 计算指标的统一方法
  calculate_metric <- function(t, metric) {
    metrics <- calculate_metrics(t$true, t$point)
    
    switch(metric,
           'RMSE' = metrics$rmse,
           'MAE' = metrics$mae,
           'MAPE' = metrics$mape,
           'SMAPE' = metrics$smape,
           'median_ae' = metrics$median_ae,
           'WIS' = {
             t$wis <- apply(t, 1, calculate_wis)
             mean(t$wis)
           },
           stop("Unknown metric: ", metric))
  }
  
  # 主处理逻辑
  plot_data <- map_dfr(models, function(model) {
    res <- process_data(path_map[model])
    
    metric_values <- map_dbl(0:(pred_steps-1), function(i) {
      t <- res %>% filter(week_ahead == i)
      calculate_metric(t, metric_label)
    })
    
    data.frame(
      WeekAhead = 0:(pred_steps-1),
      Value = metric_values,  # 使用通用列名
      Model = model_names[model]
    )
  })
  
  # 重命名Value列为实际使用的指标名称
  names(plot_data)[names(plot_data) == "Value"] <- metric_label
  
  return(plot_data)
}
  
get_usa_plot_data <- function(metric_label,model_use = 'gru') {
  models <- c('constant',  
              'GRU_base', 'TCN_base','itransformer_base','Nbeats_base',
              'GRU_SIR', 'TCN_SIR','itransformer_SIR','Nbeats_SIR',
              'ensemble'
              )
  model_names <- c(
    'constant' = "Constant",
    'GRU_base' = "GRU base",
    'TCN_base' = "TCN base",
    'itransformer_base' = "itransformer base",
    'Nbeats_base' = "Nbeats base",
    'GRU_SIR' = "GRU SIRSPF-H",
    'TCN_SIR' = "TCN SIRSPF-H",
    'itransformer_SIR' = "itransformer SIRSPF-H",
    'Nbeats_SIR' = "Nbeats SIRSPF-H",
    'ensemble' = "Average Ensemble"
    
  )
  
  pred_steps <- 9
  
  # 预定义文件路径映射'../src/{model_use}_Base_stage_holi_pf_42test.csv')
  path_map <- c(
    'constant' = '../../USA_data72/res/forc_baseline.csv',
    'GRU_base' = glue('../../USA_data72/scr/gru_Base_week_42test.csv'),
    'TCN_base' = glue('../../USA_data72/scr/tcn_Base_week_42test.csv'),
    'itransformer_base' = glue('../../USA_data72/scr/itransformer_Base_week_42test.csv'),
    'Nbeats_base' = glue('../../USA_data72/scr/Nbeats_Base_week_42test.csv'),
    'GRU_SIR' = glue('../../USA_data72/scr/gru_Base_week_holi_pf_ol_42test.csv'),
    'TCN_SIR' = glue('../../USA_data72/scr/tcn_Base_week_holi_pf_ol_42test.csv'),
    'itransformer_SIR' = glue('../../USA_data72/scr/itransformer_Base_week_holi_pf_ol_42test.csv'),
    'Nbeats_SIR' = glue('../../USA_data72/scr/Nbeats_Base_week_holi_pf_ol_42test.csv'),
    'ensemble' = glue('../../USA_data72/res/ensemble_model_with_intervals.csv')
  )
  
  # 统一处理数据的函数
  process_data <- function(path) {
    res <- read.csv(path) %>%
      {if("mode" %in% colnames(.)) filter(., mode == "train") else .} %>%
      mutate(date = as.Date(date),
             date_origin = date - week_ahead * 7)
    
    invalid_dates <- res %>%
      group_by(date) %>%
      filter(any(true <= 1)) %>%
      pull(date) %>%
      unique()
    
    res %>%
      filter(!(date_origin %in% invalid_dates),
             date_origin >= as.Date('2023-10-01')) %>%
      na.omit()
  }
  
  # 计算指标的统一方法
  calculate_metric <- function(t, metric) {
    metrics <- calculate_metrics(t$true, t$point)
    
    switch(metric,
           'RMSE' = metrics$rmse,
           'MAE' = metrics$mae,
           'MAPE' = metrics$mape,
           'SMAPE' = metrics$smape,
           'median_ae' = metrics$median_ae,
           'WIS' = {
             t$wis <- apply(t, 1, calculate_wis)
             mean(t$wis)
           },
           stop("Unknown metric: ", metric))
  }
  
  # 主处理逻辑
  plot_data <- map_dfr(models, function(model) {
    res <- process_data(path_map[model])
    
    metric_values <- map_dbl(0:(pred_steps-1), function(i) {
      t <- res %>% filter(week_ahead == i)
      calculate_metric(t, metric_label)
    })
    
    data.frame(
      WeekAhead = 0:(pred_steps-1),
      Value = metric_values,  # 使用通用列名
      Model = model_names[model]
    )
  })
  
  # 重命名Value列为实际使用的指标名称
  names(plot_data)[names(plot_data) == "Value"] <- metric_label
  
  return(plot_data)
}



model_alphas <- c(
  "Constant" = 1,
  "GRU base" = 0.7,
  "TCN base" = 0.7,
  "Nbeats base" = 0.7,
  "itransformer base" = 0.7,
  "GRU SIRSPF-H" = 0.5,
  "TCN SIRSPF-H" = 0.5,
  "Nbeats SIRSPF-H" = 0.5,
  "itransformer SIRSPF-H" = 0.5,
  "Average Ensemble" = 1
)

model_use = 'GRU'
color = c(
  "Constant" = "#D62728",
  "GRU base" = "#1F77B4",
  "TCN base" = "#4C78A8",
  "Nbeats base" = "#6BAED6",
  "itransformer base" = "#9ECAE1",
  "GRU SIRSPF-H" = "#1B9E77",       # Deep teal
  "TCN SIRSPF-H" = "#4CB7A5",       # Medium teal
  "Nbeats SIRSPF-H" = "#7ECDC3",    # Light teal
  "itransformer SIRSPF-H" = "#B7E2DB", # Very light teal
  "Average Ensemble" = "#FF6B6B"
)


p1 <- ggplot(get_hk_plot_data("WIS",model_use = model_use), aes(x = WeekAhead, y = WIS, color = Model)) +
  geom_line( linewidth = 1) +
  geom_point( size = 2) +
  scale_color_manual(
    limits = c("Constant", "GRU base", "TCN base", "Nbeats base", "itransformer base",
               "GRU SIRSPF-H", "TCN SIRSPF-H", "Nbeats SIRSPF-H", "itransformer SIRSPF-H", "Average Ensemble"),
    values = color
  )+
  labs(title = "WIS",x = 'Week ahead',y='WIS') +
  theme_bw()+
  theme(legend.position = "none")

p2 <- ggplot(get_hk_plot_data("MAE",model_use = model_use), aes(x = WeekAhead, y = MAE, color = Model)) +
  geom_line( linewidth = 1) +
  geom_point( size = 2) +
  scale_color_manual(
    limits = c("Constant", "GRU base", "TCN base", "Nbeats base", "itransformer base",
               "GRU SIRSPF-H", "TCN SIRSPF-H", "Nbeats SIRSPF-H", "itransformer SIRSPF-H", "Average Ensemble"),
    values = color
  )+
  labs(title = "MAE",x = 'Week ahead',y='MAE') +
  theme_bw()+
  theme(legend.position = "none")

p3 <- ggplot(get_hk_plot_data("SMAPE",model_use = model_use), aes(x = WeekAhead, y = SMAPE, color = Model)) +
  geom_line( linewidth = 1) +
  geom_point( size = 2) +
  scale_color_manual(
    limits = c("Constant", "GRU base", "TCN base", "Nbeats base", "itransformer base",
               "GRU SIRSPF-H", "TCN SIRSPF-H", "Nbeats SIRSPF-H", "itransformer SIRSPF-H", "Average Ensemble"),
    values = color
  )+
  labs(title = "SMAPE",x = 'Week ahead',y='SMAPE') +
  theme_bw()+
  theme(legend.position = "none")

p21 <- ggplot(get_usa_plot_data("WIS",model_use = model_use), aes(x = WeekAhead, y = WIS, color = Model)) +
  geom_line( linewidth = 1) +
  geom_point( size = 1.5) +
  scale_color_manual(
    limits = c("Constant", "GRU base", "TCN base", "Nbeats base", "itransformer base",
               "GRU SIRSPF-H", "TCN SIRSPF-H", "Nbeats SIRSPF-H", "itransformer SIRSPF-H", "Average Ensemble"),
    values = color
  )+

  labs(title = "WIS",x = 'Week ahead',y='WIS') +
  theme_bw()+
  theme(legend.position = "none")

p22 <- ggplot(get_usa_plot_data("MAE",model_use = model_use), aes(x = WeekAhead, y = MAE, color = Model)) +
  geom_line( linewidth = 1) +
  geom_point( size = 1.5) +
  scale_color_manual(
    limits = c("Constant", "GRU base", "TCN base", "Nbeats base", "itransformer base",
               "GRU SIRSPF-H", "TCN SIRSPF-H", "Nbeats SIRSPF-H", "itransformer SIRSPF-H", "Average Ensemble"),
    values = color
  )+
  labs(title = "MAE",x = 'Week ahead',y='MAE') +
  theme_bw()+
  theme(legend.position = "none")


p23 <- ggplot(get_usa_plot_data("SMAPE",model_use = model_use), aes(x = WeekAhead, y = SMAPE, color = Model)) +
  geom_line( linewidth = 1) +
  geom_point( size = 1.5) +
  scale_color_manual(
    limits = c("Constant", "GRU base", "TCN base", "Nbeats base", "itransformer base",
               "GRU SIRSPF-H", "TCN SIRSPF-H", "Nbeats SIRSPF-H", "itransformer SIRSPF-H", "Average Ensemble"),
    values = color
  )+
  labs(title = "SMAPE",x = 'Week ahead',y='SMAPE') +
  theme_bw()

combined_plot <- (p1 + p2 + p3 + p21 + p22 + p23) +
  plot_annotation(tag_levels = list(c("HK", "", "", "US", "", ""))) &  # 前三个加USA标签
  theme(
    plot.tag = element_text(size = 14, face = "bold", color = "black"),  # 标签样式
    plot.tag.position = c(0.03, 0.99)  # 标签位置（左上角）
  )


combined_plot



#   
