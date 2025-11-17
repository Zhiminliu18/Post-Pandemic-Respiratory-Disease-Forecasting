import copy
from n_beats_model import *
from data_util import *
import pickle
from neuralforecast.models import iTransformer
from neuralforecast.losses.pytorch import MSE
import numpy as np
import torch
from sklearn.metrics import mean_squared_error



def create_prediction_dataframe(model, data, scaler, next_date, mode, shuffle_feature=False, feature_idx=None,time_idx=None):
    # 获取预测结果
    df_pred_list = []
    end_seed = 62 if shuffle_feature else 43
    for seed in range(42,end_seed):
        y_true, y_pred = get_val_predictions_itrans(
            model, data, scaler,
            shuffle_feature=shuffle_feature,
            feature_idx=feature_idx,
            timestep_idx=time_idx,
            seed = seed
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
        mode = mode+ f'_time{time_step}' if time_step is not None else mode
        df_pred['mode'] = mode+ f'_seed{seed}'
        df_pred_list.append(df_pred)
    df_pred = pd.concat(df_pred_list, ignore_index=True)
    return df_true, df_pred


def main(horizon, mode, model_name, seed, params,train_end_date,val_end_date,test_end_date,year):
    set_seed(seed)
    features_col = ['ILI+']

    if 'pf' in mode:
        features_col.append('ILI_est')
        features_col.append('I')
        features_col.append('beta')
    if 'holi' in mode:
        features_col.append('decay_distance')
    if 'week' in mode:
        features_col.append('Week')
    print(features_col)

    data_util = DataPrepare(df=data,
                            features=features_col,
                            future_holiday=['decay_distance'],
                            target=['ILI+','ILI_est','I','beta'],
                            seq=14,
                            horizon=horizon)

    iTransformer_model = iTransformer(h=horizon,
                         input_size=14,
                         n_series=len(data_util.features),
                         hidden_size=32,
                         n_heads=1,
                         e_layers=1,
                         loss=MSE(),
                         d_ff=32,
                         dropout=0.3,
                         use_norm=False,
                         )



    train_data2 = data_util.get_data('2013-04-07', '2019-12-15')#'2019-05-12' '2020-03-01'
    train_data3 = data_util.get_data('2023-03-19', train_end_date,now_forecast=True)  # 2024-09-22  2023-10-29 #2023-06-04
    train_data = merge_data(train_data2, train_data3)
    train_data, train_scaler = standardize_data(train_data, normalize=True)


    val_data3 = data_util.get_data(train_end_date, val_end_date,now_forecast=True) #'2024-04-07'
    val_data, _ = standardize_data(val_data3, train_scaler, normalize=True)

    # if year==2024:
    #     test_start_date = pd.to_datetime('2024-02-11')
    # else:
    #     test_start_date = val_end_date
    test_data = data_util.get_data(val_end_date, test_end_date)
    test_data, _ = standardize_data(test_data, train_scaler, normalize=True)

    itrans_old = train_itrans(iTransformer_model, train_data, val_data,num_epochs=params['epoch'], batch_size=params['batch_size'], lr=params['lr'],weight_decay=params['weight_decay'],plot=False) #96 96 2 3 30 32 0.00061

    # ol
    all_predictions = []
    all_actuals = []
    ol_start_date = pd.to_datetime(val_end_date)

    for i in range(90):
        current_date = ol_start_date+timedelta(days=i*7)
        if current_date>pd.to_datetime(test_end_date):
            break
        next_date = current_date + timedelta(days=7)
        new_data = data_util.get_data('2023-03-19', str(current_date),now_forecast=True)
        new_train_data, _ = standardize_data(new_data, train_scaler, normalize=True, get_last=8)
        # use mask to prevent info leak
        if '_ol' in mode:
            itrans_new = train_itrans(copy.deepcopy(itrans_old), new_train_data, new_train_data, num_epochs=2, batch_size=4, lr=params['lr']/5,plot=False,froze=True)
        else:
            itrans_new = itrans_old

        # for i in range(len(data_util.features)):
        #     pm_mode = f'feature{i}'
        #     df_true, df_pred = create_prediction_dataframe(
        #         itrans_new, test_data, train_scaler, next_date, pm_mode,
        #         shuffle_feature=True, feature_idx=i
        #     )
        #     all_predictions.append(df_pred)
        #     all_actuals.append(df_true)

        df_true, df_pred = create_prediction_dataframe(
            itrans_new, new_train_data, train_scaler, next_date, 'train',
            shuffle_feature=False
        )
        all_predictions.append(df_pred)
        all_actuals.append(df_true)

    final_actuals = pd.concat(all_actuals, ignore_index=True)
    final_predictions = pd.concat(all_predictions, ignore_index=True)
    final_predictions.fillna(0, inplace=True)

    # new_method_res1 = []
    # for i in range(horizon):
    #     var = f'week_ahead{i}'
    #     y_true = final_actuals[var]
    #     y_pred = final_predictions[var]
    #     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    #     new_method_res1.append(rmse)
    #     print(f"RMSE for ILI_lead{i}: {rmse:.4f}")
    # print(np.mean(new_method_res1))

    
    result_dfs = []
    for i in range(horizon):
        temp_df = pd.DataFrame({
            'date': final_predictions['date'] + timedelta(days=i * 7),
            'week_ahead': i,  # 第1周、第2周...
            'point': final_predictions[f'week_ahead{i}'],
            'mode': final_predictions['mode']

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
    data = get_data()
    horizon = 9
    model_name = 'itransformer'
    modes = ['base_week_holi_pf']
    seeds = [42]
    params1 = {
        'lr' : 0.0006,
        'batch_size' : 32,
        'epoch' : 50,
        'weight_decay': 10e-3
    }

    params2 = {
        'lr' : 0.00006,
        'batch_size' : 32,
        'epoch' : 50,
        'weight_decay' : 10e-3
    }

    for mode in modes:
        for seed in seeds:
            main(horizon, mode, model_name,seed, params1,train_end_date = '2023-07-02', val_end_date = '2023-09-24',
                 test_end_date = '2024-10-06',year=2024)
            main(horizon, mode, model_name, seed, params2, train_end_date='2023-09-24', val_end_date='2024-10-06',
                 test_end_date='2025-06-22', year=2025)

    # concat data
    years = [2024, 2025]
    files = []
    for mode in modes:
        for seed in seeds:
            mode_files = []
            for year in years:
                path = f'../res/{model_name}/{model_name}_{mode}_{seed}test{year}.csv'
                mode_files.append(path)
            files.append((f'../res/{model_name}/{model_name}_{mode}_{seed}test.csv', mode_files))  # 存储最终文件名和临时文件列表

    for final_name, mode_files in files:
        data1 = pd.read_csv(mode_files[0])
        data2 = pd.read_csv(mode_files[1])
        data = pd.concat([data1, data2])
        data.to_csv(final_name, index=False)  # 使用不包含年份的文件名保存