import copy
import os
from n_beats_model import *
from data_util import *
import numpy as np
import torch
from sklearn.metrics import mean_squared_error
from dataclasses import dataclass, field


def create_prediction_dataframe(model, data, scaler, next_date, mode, shuffle_feature=False, feature_idx=None,time_idx=None):
    # 获取预测结果
    y_true, y_pred = get_val_predictions(
        model, data, scaler,
        shuffle_feature=shuffle_feature,
        feature_idx=feature_idx,
        timestep_idx=time_idx
    )
    # 创建DataFrame
    df_true = pd.DataFrame(y_true, columns=[f"week_ahead{i}" for i in range(horizon)])
    df_pred = pd.DataFrame(y_pred, columns=[f"week_ahead{i}" for i in range(horizon)])
    # 添加日期列
    df_true['date'] = data['dates']
    df_pred['date'] = data['dates']
    # 筛选指定日期的数据
    df_true = df_true[df_true['date'] == next_date]
    df_pred = df_pred[df_pred['date'] == next_date].copy()
    # 添加模式标识
    time_step = None
    if time_idx is not None:
        time_step = 13-time_idx
    df_pred['mode'] = mode+ f'_time{time_step}' if time_step is not None else mode
    return df_true, df_pred


def main(data,permutation, horizon, mode, model_name, seed, params,train_end_date,val_end_date,test_end_date,year):
    set_seed(seed)
    features_col = ['ILI+']

    if 'pf' in mode:
        features_col.append('ILI_est')
        features_col.append('I')
        features_col.append('beta')
    if 'holi' in mode:
        features_col.append('decay_distance')
    if 'stage' in mode:
        features_col.append('status')
    if 'week' in mode:
        features_col.append('flu_weekid')
    print(features_col)

    data_util = DataPrepare(df=data,
                            features=features_col,
                            future_holiday=['decay_distance'],
                            target=['ILI+','ILI_est','I','beta'],
                            seq=14,
                            horizon=horizon)

    GRU_model = RNNPredictor1(input_size=len(data_util.features),
                                hidden_size=108, output_size=1,
                                num_layers=1, output_time_steps=horizon,
                                dropout=0.3,
                                rnn_type=model_name,
                                if_I=True)


    #train_data
    train_data1 = data_util.get_data('1998-12-27', '2009-06-14')
    if train_end_date>pd.to_datetime('2020-05-12'):
        train_data2 = data_util.get_data('2010-06-06', '2019-05-12')#'2019-05-12' '2020-03-01'
        train_data3 = data_util.get_data('2023-03-19', train_end_date,now_forecast=True)  
        train_data = merge_data(train_data1, train_data2)
        train_data = merge_data(train_data, train_data3)
    else:
        train_data3 = data_util.get_data('2010-06-06', train_end_date,now_forecast=True)
        train_data = merge_data(train_data1, train_data3)

    train_data, train_scaler = standardize_data(train_data, normalize=True)

    #val_data
    val_data = data_util.get_data(train_end_date, val_end_date,now_forecast=True)
    val_data, _ = standardize_data(val_data, train_scaler, normalize=True)

    #test_data
    test_data = data_util.get_data(val_end_date, test_end_date)
    test_data, _ = standardize_data(test_data, train_scaler, normalize=True)


    GRU_old = train(GRU_model, train_data, val_data, num_epochs=params['epoch'], batch_size=params['batch_size'], lr=params['lr'],weight_decay=params['weight_decay'], plot=False)   #0.0011

    # online learning
    all_predictions = []
    all_actuals = []
    ol_start_date = pd.to_datetime(val_end_date)
    for i in range(90):
        current_date = ol_start_date+timedelta(days=i*7)
        if current_date>pd.to_datetime(test_end_date):
            break
        next_date = current_date + timedelta(days=7)

        #get latest data
        new_data = data_util.get_data('2013-10-27', str(current_date),now_forecast=True)
        new_train_data, _ = standardize_data(new_data, train_scaler, normalize=True, get_last=8)
        if '_ol' in mode:
            GRU = train(copy.deepcopy(GRU_old), new_train_data, new_train_data, num_epochs=2, batch_size=4, lr=params['lr']/5,plot=False,froze=True) # epoch=1,lr=0.0001
        else:
            GRU = GRU_old

        # do permutation
        if permutation:
            for i in range(len(data_util.features)):
                for j in range(14):
                    pm_mode = f'feature{i}'
                    time_idx = j
                    df_true, df_pred = create_prediction_dataframe(
                        GRU, test_data, train_scaler, next_date, pm_mode,
                        shuffle_feature=True, feature_idx=i,time_idx=time_idx
                    )
                    all_predictions.append(df_pred)
                    all_actuals.append(df_true)

        # prediction
        df_true, df_pred = create_prediction_dataframe(
            GRU, new_train_data, train_scaler, next_date, 'train',
            shuffle_feature=False
        )

        all_predictions.append(df_pred)
        all_actuals.append(df_true)


    final_actuals = pd.concat(all_actuals, ignore_index=True)
    final_predictions = pd.concat(all_predictions, ignore_index=True)
    final_predictions.fillna(0, inplace=True)

    # calculate metric
    new_method_res1 = []
    for i in range(horizon):
        var = f'week_ahead{i}'
        y_true = final_actuals[var]
        y_pred = final_predictions[var]
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        new_method_res1.append(rmse)
        print(f"RMSE for ILI_lead{i}: {rmse:.4f}")
    print(np.mean(new_method_res1))

    # restore results
    result_dfs = []
    for i in range(horizon):
        temp_df = pd.DataFrame({
            'date': final_predictions['date'] + timedelta(days=i * 7),
            'week_ahead': i,
            'point': final_predictions[f'week_ahead{i}'],
            'mode': final_predictions['mode'],
        })
        result_dfs.append(temp_df)

    final_pred_df = pd.concat(result_dfs, ignore_index=True)

    true_df = data[['date', "ILI+"]]
    format_pred_df = pd.merge(
        final_pred_df,  # 左表（预测值）
        true_df,  # 右表（真实值）
        on='date',  # 连接键
        how='left')
    format_pred_df = format_pred_df.rename(columns={'ILI+': 'true'})
    format_pred_df['date_origin'] = format_pred_df['date'] - format_pred_df['week_ahead'] * timedelta(days=7)
    result_cols = [f'q_{int(q*100)}' for q in QUANTILES]
    format_pred_df[result_cols] = format_pred_df.apply(calc_quantiles, axis=1)
    output_dir = f'../res/{model_name}'
    output_file = f'{output_dir}/{model_name}_{mode}_{seed}test{year}.csv'
    os.makedirs(output_dir, exist_ok=True)
    format_pred_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    # TODO data util optimize
    data = get_data()
    horizon = 9
    model_name = 'gru'
    # modes = ['base_holi_pf','base_holi','base_pf','base']
    modes = ['base_holi_pf']
    permutation = True
    seeds = [42] #123,456
    params = {
        'lr' : 0.0001,
        'batch_size' : 8,
        'epoch' : 50,
        'weight_decay': 10e-8
    }
   # todo define date according to year
    years = [i for i in range(2017,2020)]
    for year in years:
        print(f"----Processing year: {year}----")
        train_end_date = data[data['flu_season_year'] == year - 2]['date'].max()    
        val_end_date = data[data['flu_season_year'] == year - 1]['date'].max()
        test_end_date = data[data['flu_season_year'] == year]['date'].max()
        print(f"train_end_date: {train_end_date}")
        print(f"val_end_date: {val_end_date}")
        print(f"test_end_date: {test_end_date}")
        print("-" * 40)  # 分隔线
        for mode in modes:
            for seed in seeds:
                main(data, permutation ,horizon, mode, model_name,seed, params,train_end_date, val_end_date ,test_end_date,year)
          
    # concat data
    files = []
    for mode in modes:
        for seed in seeds:
            mode_files = []
            for year in years:
                path = f'../res/{model_name}/{model_name}_{mode}_{seed}test{year}.csv'
                mode_files.append(path)
            files.append((f'../res/{model_name}/{model_name}_{mode}_{seed}test.csv', mode_files))  # 存储最终文件名和临时文件列表

    for final_name, mode_files in files:
        data_list = [pd.read_csv(file) for file in mode_files]  # 读取所有文件
        data = pd.concat(data_list)  # 合并所有 DataFrame
        data.to_csv(final_name, index=False)  # 保存合并后的文件

