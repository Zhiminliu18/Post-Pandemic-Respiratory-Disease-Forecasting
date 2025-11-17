import copy
from n_beats_model import *
from neuralforecast.models import TCN
from data_util import *
import pickle
import numpy as np
from sklearn.metrics import mean_squared_error
from neuralforecast.losses.pytorch import  DistributionLoss
from dataclasses import dataclass, field


def create_prediction_dataframe(model, data, scaler, next_date, mode, shuffle_feature=False, feature_idx=None,time_idx=None):
    # 获取预测结果
    df_pred_list = []

    for seed in range(42,62):
        y_true, y_pred = get_tcn_val_predictions(
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
    hist_exog_list = []
    if 'pf' in mode:
        hist_exog_list.append('ILI_est')
        hist_exog_list.append('I')
        hist_exog_list.append('beta')
    if 'holi' in mode:
        hist_exog_list.append('decay_distance')
    if 'stage' in mode:
        hist_exog_list.append('status')
    print(hist_exog_list)

    seq = 14
    data_util = DataPrepare(df=data,
                            features=['ILI+']+hist_exog_list, #'diff_mean_4','diff_std_4','diff_max_4',
                            future_holiday=['decay_distance'],
                            target=['ILI+','ILI_est','I','beta'],
                            seq=seq,
                            horizon=horizon)

    tcn_model = TCN( h=horizon,
                  input_size=seq,
                  kernel_size=2,
                  dilations=[1,2,3],
                  encoder_hidden_size=128,
                  decoder_hidden_size=128,
                  decoder_layers=2,
                  futr_exog_list=None,
                  hist_exog_list=hist_exog_list,
                  stat_exog_list=None
                  )


    train_end_date = '2023-08-06'
    train_data1 = data_util.get_data('1998-12-27', '2009-06-14')
    train_data2 = data_util.get_data('2010-06-06', '2019-05-12')#'2019-05-12' '2020-03-01'
    train_data3 = data_util.get_data('2023-03-19', train_end_date,now_forecast=True)
    # train_data3 = upsample_data(train_data3)
    train_data = merge_data(train_data1, train_data2)
    train_data = merge_data(train_data, train_data3)
    train_data, train_scaler = standardize_data(train_data, normalize=True)


    #用截至到end_date的数据，用mask遮盖未来数据，防止信息leak
    val_data3 = data_util.get_data(train_end_date, val_end_date,now_forecast=True) #'2024-04-07'
    val_data, _ = standardize_data(val_data3, train_scaler, normalize=True)

    test_data = data_util.get_data(val_end_date, test_end_date)#'2025-04-06'
    test_data, _ = standardize_data(test_data, train_scaler, normalize=True)

    TCN_old = train_tcn(tcn_model, train_data, val_data,  num_epochs=params['epoch'], batch_size=params['batch_size'], lr=params['lr'],weight_decay=params['weight_decay'], plot=False)  #128,128,2,30,16,0.0005 10e-8

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
        new_train_data, _ = standardize_data(new_data, train_scaler, normalize=True, get_last=8) #16
        if '_ol' in mode:
            TCN_new = train_tcn(copy.deepcopy(TCN_old), new_train_data, new_train_data, num_epochs=2, batch_size=4, lr=params['lr']/5,plot=False,froze=False)
        else:
            TCN_new = TCN_old

        # permutation feature
        # for i in range(len(data_util.features)):
        #     pm_mode = f'feature{i}'
        #     df_true, df_pred = create_prediction_dataframe(
        #         TCN_new, test_data, train_scaler, next_date, pm_mode,
        #         shuffle_feature=True, feature_idx=i
        #     )
        #     all_predictions.append(df_pred)
        #     all_actuals.append(df_true)
        
        df_true, df_pred = create_prediction_dataframe(
            TCN_new, new_train_data, train_scaler, next_date, 'train',
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

    # 动态生成列名并赋值
    result_cols = [f'q_{int(q*100)}' for q in QUANTILES]

    format_pred_df[result_cols] = format_pred_df.apply(calc_quantiles, axis=1)
    output_dir = f'../res/{model_name}'
    output_file = f'{output_dir}/{model_name}_{mode}_{seed}test{year}.csv'
    os.makedirs(output_dir, exist_ok=True)
    format_pred_df.to_csv(output_file, index=False)



if __name__ == "__main__":
    data = get_data()
    horizon = 9
    model_name = 'tcn'
    # modes = ['base_stage_holi_pf_ol', 'base_stage_holi_pf','base_stage_holi','base_stage_pf','base_stage','base']
    modes = ['base','base_stage_holi_pf']
    seeds = [42]
    params1 = {
        'lr' : 0.0009,
        'batch_size' : 16,
        'epoch' : 50,
        'weight_decay': 10e-6
    }

    params2 = {
        'lr' : 0.0009,
        'batch_size' : 16,
        'epoch' : 50,
        'weight_decay' : 10e-6
    }

    for mode in modes:
        for seed in seeds:
            main(horizon, mode, model_name,seed, params1,train_end_date = '2023-08-06', val_end_date = '2023-11-26',
                 test_end_date = '2024-12-01',year=2024)
            main(horizon, mode, model_name, seed, params2, train_end_date='2023-11-26', val_end_date='2024-09-15',
                 test_end_date='2025-03-30', year=2025)

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
