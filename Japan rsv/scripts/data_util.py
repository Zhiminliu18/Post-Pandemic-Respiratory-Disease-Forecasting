import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import torch
import random
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from datetime import timedelta
from sklearn.metrics import mean_squared_error
from scipy.stats import beta
QUANTILES = [0.01,0.025,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95,0.975,0.99]

def set_seed(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_flu_weekid_and_season(date):
    year = date.year
    # 找到当前年份11月的第一个星期日
    nov_first = pd.Timestamp(year=year, month=11, day=1)
    first_sunday_nov = nov_first + pd.Timedelta(days=(6 - nov_first.dayofweek) % 7)

    # 如果日期在11月第一个星期日之前，则属于上一年的流感季节
    if date < first_sunday_nov:
        season_year = year - 1
        nov_first = pd.Timestamp(year=season_year, month=11, day=1)
        first_sunday_nov = nov_first + pd.Timedelta(days=(6 - nov_first.dayofweek) % 7)
    else:
        season_year = year

    # 计算周数
    week_id = ((date - first_sunday_nov).days // 7) + 1
    week_id = min(week_id, 53)  # 确保不超过53周

    return pd.Series([season_year, week_id])



def get_data(scale_k = 1.0,data_path = '../data/Japan_rsv_data.csv', pf_path='../res/particle_filter_results_721.pkl'):
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    data[['flu_season_year', 'flu_weekid']] = data['date'].apply(get_flu_weekid_and_season)
    data['flu_season_year'] = data['flu_season_year'].astype(int) + 1
    data['month'] = data['date'].dt.month

    def assign_season(month):
        if month in [12, 1, 2]:
            return 1  # 冬春季节
        elif month in [3, 4, 5]:
            return 2
        elif month in [6, 7, 8]:
            return 3
        else:
            return 0

    data['season'] = data['month'].apply(assign_season)
    data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
    data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)
    # data['week_sin'] = np.sin(2 * np.pi * data['flu_weekid'] / 52.5)
    # data['week_cos'] = np.cos(2 * np.pi * data['flu_weekid'] / 52.5)
    with open(pf_path, 'rb') as f:  # 注意是 'rb'（二进制读取模式）
        results = pickle.load(f)

    data['ILI_est'] = results['ILI_estimates'] / 100
    data['beta'] = results['param_estimates'][:, 0]
    data['gamma'] = results['param_estimates'][:, 1]
    data['xi'] = results['param_estimates'][:, 2]

    data['S'] = results['state_estimates'][:, 0]
    data['I'] = results['state_estimates'][:, 1]
    data['R'] = results['state_estimates'][:, 2]
    # data['total_inf'] = data.apply(lambda row: calculate_total_inf(row), axis=1)

    data = data.rename(columns={'ILI': 'ILI+'})
    # 无量纲化
    data['ILI+'] = data['ILI+'] * scale_k
    # mask = data['date'] < pd.to_datetime('2020-01-01')
    # data.loc[mask, 'ILI+'] = data.loc[mask, 'ILI+'] * 0.5

    data['ILI_growth_rate'] = data['ILI+'].diff()
    data['ILI_growth_rate'] = data['ILI_growth_rate'].fillna(0)  # 填充第一个Na
    data['diff_spike'] = (
                data['ILI_growth_rate'] > data['ILI_growth_rate'].rolling(4).mean() + 1.5*data['ILI_growth_rate'].rolling(
            4).std()).astype(int)



    N = 1
    data['S'] = data['S'] / N
    data['I'] = data['I'] / N
    data['R'] = data['R'] / N
    data['status'] = np.where(

        data['Year'] < 2023, 1,0

    )

    data['ili_status'] = np.where(
        data['ILI+'] < 2, 0, 1
    )
    # data['beta'] = (data['beta']-data['status'])/data['status']



    data['SIR'] = data['S'] * data['I'] * data['beta']
    data['ILI*beta'] = data['beta'] * data['ILI+']
    data['SIR_new'] = data['S'] * data['beta']
    data['I_gr'] = data['I'].diff()  # .diff()
    data['I_gr'] = data['I_gr'].fillna(0)  # 填充第一个Na
    data['SIR_gr'] = data['SIR'].diff()  # .diff()
    data['SIR_gr'] = data['SIR_gr'].fillna(0)  # 填充第一个Na
    data['ILI_est_gr'] = data['ILI_est'].diff()  # .pct_change()
    data['ILI_est_gr'] = data['ILI_est_gr'].fillna(0)  # 填充第一个Na
    data['beta_gr'] = data['beta'].diff()
    data['beta_gr'] = data['beta_gr'].fillna(0)  # 填充第一个Na
    data['beta_diff2'] = data['beta_gr'].diff()  # 差分的变化率，反映加速/减速
    data['beta_diff2'] = data['beta_diff2'].fillna(0)  # 差分的变化率，反映加速/减速

    data['mean_2'] = data['ILI+'].rolling(2).mean()  # 过去4周平均变化
    data['mean_2'] = data['mean_2'].fillna(0)  # 填充第一个Na
    data['max_2'] = data['ILI+'].rolling(2).max()  # 过去4周平均变化
    data['max_2'] = data['max_2'].fillna(0)  # 填充第一个Na
    data['min_2'] = data['beta'].rolling(2).min()  # 过去4周平均变化
    data['min_2'] = data['min_2'].fillna(0)  # 填充第一个Na
    data['avg_2'] = data['beta'].rolling(2).max()  # 过去4周平均变化
    data['avg_2'] = data['avg_2'].fillna(0)  # 填充第一个Na

    data['ILI'] = data['ILI+'].rolling(2).mean()  # 过去4周平均变化
    data['ILI'] = data['ILI'].fillna(0)  # 填充第一个Na

    data['ILI_diff'] = data['ILI'].diff()
    data['ILI_diff'] = data['ILI_diff'].fillna(0)  # 填充第一个Na

    data['diff_mean_4'] = data['ILI_growth_rate'].rolling(3).mean()  # 过去4周平均变化
    data['diff_mean_4'] = data['diff_mean_4'].fillna(3)  # 填充第一个Na
    data['diff_std_4'] = data['ILI_growth_rate'].rolling(3).std()  # 过去4周变化波动性
    data['diff_std_4'] = data['diff_std_4'].fillna(0)  # 填充第一个Na
    data['diff_max_4'] = data['ILI_growth_rate'].rolling(3).max()  # 过去4周最大变化
    data['diff_max_4'] = data['diff_max_4'].fillna(0)  # 填充第一个Na
    data['seq_diff2'] = data['ILI_growth_rate'].diff()  # 差分的变化率，反映加速/减速
    data['seq_diff2'] = data['seq_diff2'].fillna(0)  # 差分的变化率，反映加速/减速


    data['diff_mean_4_est'] = data['ILI_est_gr'].rolling(3).mean()  # 过去4周平均变化
    data['diff_mean_4_est'] = data['diff_mean_4_est'].fillna(3)  # 填充第一个Na
    data['diff_std_4_est'] = data['ILI_est_gr'].rolling(3).std()  # 过去4周变化波动性
    data['diff_std_4_est'] = data['diff_std_4_est'].fillna(0)  # 填充第一个Na
    data['diff_max_4_est'] = data['ILI_est_gr'].rolling(3).max()  # 过去4周最大变化
    data['diff_max_4_est'] = data['diff_max_4_est'].fillna(0)  # 填充第一个Na
    data['seq_diff2_est'] = data['ILI_est_gr'].diff()  # 差分的变化率，反映加速/减速
    data['seq_diff2_est'] = data['seq_diff2_est'].fillna(0)  # 差分的变化率，反映加速/减速
    # 标记差分突然增大的点（可能预示爆发）
    lags = [1, 2, 3]  # 可根据需要调整滞后阶数
    for lag in lags:
        data[f'beta_lag{lag}'] = data['beta'].shift(lag)
        data[f'ILI_lag{lag}'] = data['ILI+'].shift(lag)

    data.loc[data['holiday_effect'] > 0, 'holiday_effect'] = 1
    data['beta_lag1_gr'] = data['beta_lag1'].diff()
    data['beta_lag1_gr'] = data['beta_lag1_gr'].fillna(0)  # 填充第一个Na

    data['ILI_lag1_gr'] = data['ILI_lag1'].diff()
    data['ILI_lag1_gr'] = data['ILI_lag1_gr'].fillna(0)  # 填充第一个Na
    # data['beta_lag1_gr'] = np.where(data['beta_lag1_gr'] > 0, 1, 0)
    # data['beta_gr'] = np.where(data['beta_gr'] > 0, 1, -1)
    data.loc[data['holiday_effect'] > 0, 'holiday_effect'] = 1

    data['time_idx'] = range(len(data))
    data['time_sin'] = np.sin(2 * np.pi * data['time_idx'] / data['time_idx'].max())
    data['time_cos'] = np.cos(2 * np.pi * data['time_idx'] / data['time_idx'].max())

    def add_exponential_decay_distance(group):
        group = group.reset_index(drop=True)  # 确保索引从0开始

        # 找到所有holiday_effect=1的位置
        ones_indices = group.index[group['holiday_effect'] == 1].tolist()
        decay_rate = 0.5  # 衰减率，设置为0.5
        distances = []
        for i in range(len(group)):
            if group.at[i, 'holiday_effect'] == 1:
                distances.append(1.0)  # 当前位置是1，权重最大
            else:
                if ones_indices:
                    # 计算到所有1的有符号距离（i - idx）
                    signed_dists = [i - idx for idx in ones_indices]
                    # 找到绝对值最小的距离（最近的1）
                    nearest_dist = min(signed_dists, key=abs)
                    # 如果距离是正的，应用指数衰减；如果负的，设为0
                    # decayed = (decay_rate ** abs(nearest_dist)) if nearest_dist > 0 else 0.0
                    decayed = np.sign(nearest_dist) * (decay_rate ** abs(nearest_dist))
                    # decayed = decay_rate ** abs(nearest_dist)
                else:
                    decayed = 0.0  # 如果没有1，设为0
                distances.append(decayed)

        group['decay_distance'] = distances
        return group
    data = data.groupby('Year').apply(add_exponential_decay_distance)
    return data

def calc_quantiles(row, quantiles=None, k=220):
    if quantiles is None:
        quantiles = QUANTILES
    t = row['week_ahead']
    k = k - 15 * (t + 1)
    if row['point'] <= 0:
        row['point'] = 1e-3
    scale = (row['point'] / 100)
    a = k * scale
    b = k * (1 - scale)
    quant_values = beta.ppf(quantiles, a, b)
    quant_values = quant_values * 100
    return pd.Series(quant_values, index=[f'q_{int(q * 100)}' for q in quantiles])


def sirs_derivatives(state, beta, gamma, xi, N):
    S, I, R = state
    dS = -beta * S * I / N + xi * R
    dI = beta * S * I / N - gamma * I
    dR = gamma * I - xi * R
    new_infections_rate = beta * S * I / N  # 新增感染率（不含康复项）
    return np.array([dS, dI, dR]), new_infections_rate


def upsample_data(all_data, upsample_ratio=2, random_state=42):
    n_samples = len(all_data['X'])
    # 生成随机索引（允许重复采样）
    idx = np.random.RandomState(random_state).choice(
        n_samples,
        size=n_samples * upsample_ratio,  # 目标样本量 = 原数据量 × 2
        replace=True  # 关键：允许重复抽取
    )

    # 对所有字段按相同索引采样
    upsampled_data = {
        'X': all_data['X'][idx],
        'holiday': all_data['holiday'][idx],
        'y': all_data['y'][idx],
        'mask': all_data['mask'][idx],
        'dates': [all_data['dates'][i] for i in idx]
    }
    return upsampled_data

# 计算一周的新增人数
def calculate_total_inf(row):
    state = np.array([row['S'], row['I'], row['R']])
    _, total_infected = rk4_step(state, row['beta'], row['gamma'], row['xi'])
    return total_infected

# RK4方法
def rk4_step(state, beta, gamma, xi, N=100000, dt=1):
    total_infected = 0
    current_state = state.copy()

    for day in range(7):
        # RK4步骤：计算斜率和新增感染率
        k1, new_inf1 = sirs_derivatives(current_state, beta, gamma, xi, N)
        k2, new_inf2 = sirs_derivatives(current_state + 0.5 * dt * k1, beta, gamma, xi, N)
        k3, new_inf3 = sirs_derivatives(current_state + 0.5 * dt * k2, beta, gamma, xi, N)
        k4, new_inf4 = sirs_derivatives(current_state + dt * k3, beta, gamma, xi, N)

        # 累计感染数（RK4加权平均）
        avg_new_inf = (new_inf1 + 2 * new_inf2 + 2 * new_inf3 + new_inf4) / 6.0
        total_infected += avg_new_inf * dt

        # 更新状态
        current_state = current_state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        current_state = np.clip(current_state, 0, N)

    return current_state, total_infected

class DataPrepare:
    def __init__(self, df, features, future_holiday, target, seq, horizon):
        self.df = df.sort_values(by='date').reset_index(drop=True)
        self.features = features
        self.holiday_col = future_holiday
        self.target = target
        self.seq = seq
        self.horizon = horizon
        self.X = None
        self.holiday = None
        self.y = None
        self.seq_dates = None
        self.filter = None
        self.prepare_data()

    def prepare_data(self):
        """仅准备原始数据，不进行标准化"""
        features = self.df[self.features].values  # (sample, n_features)
        holiday = self.df[self.holiday_col].values # (sample,1)
        target = self.df[self.target].values  # (sample, 1)

        X, holiday_seq, y, seq_dates, filter_x = [], [], [], [], []

        for i in range(len(features) - self.seq - self.horizon + 1):
            # now_filter = target[i:i + self.seq].flatten()
            regular_features = features[i:i + self.seq]  # (lookback, n_features)
            holiday_mask = holiday[i:i + self.seq + self.horizon]  # (output_time_steps+lookback, 1)
            target_current = target[i + self.seq:i + self.seq + self.horizon]  #(output_time_steps,n_targets)
            # filter_x.append(now_filter)
            X.append(regular_features)
            holiday_seq.append(holiday_mask)
            y.append(target_current)
            seq_dates.append(self.df['date'][i + self.seq - 1]) # last date for X

        # self.filter = np.array(filter_x)
        self.X = np.array(X)  # (samples, lookback, n_features)
        self.holiday = np.array(holiday_seq)  # (samples, output_time_steps+lookback, 1)
        self.y = np.array(y)  # (samples, output_time_steps,n_targets)
        self.seq_dates = seq_dates

    def get_data(self, start_date, end_date,threshold=0, now_forecast = False):
        """划分训练集和验证集，并在此时进行标准化（如果 scaler 未 fit，则先 fit）"""
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        start_id = self.seq_dates.index(start_date)
        end_id = self.seq_dates.index(end_date)
        mask = np.ones((end_id+1, self.horizon, 1))  
        if now_forecast:
            for i in range(self.horizon):
                pos = self.horizon-i
                last_pos = i+1
                mask[-pos,-last_pos:,:] = 0 # batch,horizon,1

        # 找出第一个特征最后一步小于1的样本的索引
        mask1 = self.X[start_id:end_id+ 1, -1, 0] >= threshold #(左闭右闭)
        valid_indices = np.where(mask1)[0] + start_id  # 截至到知道end_date的数据

        # 训练集数据（未标准化）
        x_raw = self.X[valid_indices]
        future_features_raw = self.holiday[valid_indices]
        y_raw = self.y[valid_indices]
        mask = mask[valid_indices]
        

        dates = [self.seq_dates[i] for i in valid_indices.tolist()]
        dates = [dt + timedelta(days=7) for dt in dates]  #dates代表seq的下一天
        dates_x = [[date+timedelta(days=-7*(14-i)) for i in range(1,14)] for date in dates]
        dates_y = [[date+timedelta(days=7 * i) for i in range(8)] for date in dates]
        all_data = {
            'X': x_raw,
            'holiday': future_features_raw,
            'y': y_raw,
            'mask': mask,
            'dates': dates
        }
        return all_data


import numpy as np


def merge_data(data1, data2):
    # 合并 numpy 数组
    merged_data = {}
    merged_data['X'] = np.concatenate((data1['X'], data2['X']), axis=0)  # 沿 batch 维度拼接
    merged_data['holiday'] = np.concatenate((data1['holiday'], data2['holiday']), axis=0)
    merged_data['y'] = np.concatenate((data1['y'], data2['y']), axis=0)
    merged_data['mask'] = np.concatenate((data1['mask'], data2['mask']), axis=0)

    # 合并 dates（列表）
    if isinstance(data1['dates'], list) and isinstance(data2['dates'], list):
        merged_data['dates'] = data1['dates'] + data2['dates']
    else:
        raise ValueError("'dates' 必须是 Python 列表")
    return merged_data


def standardize_data(data, scaler=None, normalize=False,get_last=0):
    standardized_data = {}
    if scaler is None:
        scaler = {}

    # 处理 X
    x_shape = data['X'].shape
    x_reshaped = data['X'].reshape(-1, x_shape[-1])
    
    if normalize:
        if 'X' not in scaler:
            scaler_x = StandardScaler()
            scaler_x.fit(x_reshaped)
            # for i in range(2):
            #     scaler_x.mean_[i] = 0  # 第二列均值设为0
            #     scaler_x.scale_[i] = 1  # 第二列标准差设为1

            scaler['X'] = scaler_x
        else:
            scaler_x = scaler['X']
        transformed_x = scaler_x.transform(x_reshaped)
    else:
        # 不标准化时，直接使用原数据
        transformed_x = x_reshaped
        # 创建一个不改变数据的scaler
        if 'X' not in scaler:
            scaler_x = StandardScaler()
            scaler_x.mean_ = np.zeros(x_reshaped.shape[1])
            scaler_x.scale_ = np.ones(x_reshaped.shape[1])
            scaler['X'] = scaler_x
    
    standardized_data['X'] = transformed_x.reshape(x_shape)

    # 处理 y
    y_shape = data['y'].shape
    y_reshaped = data['y'].reshape(-1, y_shape[-1])
    
    if normalize:
        if 'y' not in scaler:
            scaler_y = StandardScaler()
            scaler_y.fit(y_reshaped)
            scaler['y'] = scaler_y
        else:
            scaler_y = scaler['y']
        transformed_y = scaler_y.transform(y_reshaped)
    else:
        # 不标准化时，直接使用原数据
        transformed_y = y_reshaped
        # 创建一个不改变数据的scaler
        if 'y' not in scaler:
            scaler_y = StandardScaler()
            scaler_y.mean_ = np.zeros(y_reshaped.shape[1])
            scaler_y.scale_ = np.ones(y_reshaped.shape[1])
            scaler['y'] = scaler_y
    
    standardized_data['y'] = transformed_y.reshape(y_shape)

    # 不处理 holiday 和 dates
    scaler_holiday = StandardScaler()
    h_shape = data['holiday'].shape
    h_reshaped = data['holiday'].reshape(-1, h_shape[-1])
    scaler_holiday.fit(h_reshaped)
    transformed_h = scaler_holiday.transform(h_reshaped)
    standardized_data['holiday'] = transformed_h.reshape(h_shape)
    # standardized_data['holiday'] = data['holiday']
    standardized_data['dates'] = data['dates']
    standardized_data['mask'] = data['mask']

    if get_last>0:
        standardized_data['X'] = standardized_data['X'][-get_last:,:,:]
        standardized_data['y'] = standardized_data['y'][-get_last:, :, :]
        standardized_data['holiday'] = standardized_data['holiday'][-get_last:, :, :]
        standardized_data['mask'] = standardized_data['mask'][-get_last:, :, :]
        standardized_data['dates'] = data['dates'][-get_last:]
    return standardized_data, scaler


if __name__ == '__main__':
    data = pd.read_csv('../data/flu_data_final.csv')
    data['date'] = pd.to_datetime(data['date'])
    data_util = DataPrepare(df = data,
                            features = ['ILI+','WEEK'],
                            future_holiday= ['holiday_effect'],
                            target = ['ILI+','pos'],
                            seq = 12,
                            horizon = 4)
    t = data_util.get_data('2002-12-22','2003-09-14')
