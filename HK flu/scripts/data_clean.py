import pandas as pd
import numpy as np
from datetime import timedelta,datetime
from lunarcalendar import Converter, Lunar



df = pd.read_csv('data/new_analysis_data.csv')
df['date'] = pd.to_datetime(df['date'])

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


# 应用函数并创建两列
df[['flu_season_year', 'flu_weekid']] = df['date'].apply(get_flu_weekid_and_season)
df['flu_season_year'] = df['flu_season_year'].astype(int) + 1


df['status'] = np.where(
    df['flu_season_year'].between(1997, 2019), 0,  # 1998-2019 → 1
    np.where(
        df['flu_season_year'] > 2022, 1,          # <2016 → 0
        2                                         # >2022 → 3
    )
)

data = df.sort_values("date").reset_index(drop=True)


# 元旦后的开学日
def get_first_workday_after_new_year(year):
    new_year = datetime(year, 1, 2)
    current_date = new_year
    # 如果元旦是周末，找到下一个工作日
    while current_date.weekday() >= 5:  # 5=周六，6=周日
        current_date += timedelta(days=1)
    return current_date

# 春节后的开学日
def get_first_monday_after_spring_festival(year):
    lunar_date = Lunar(year, 1, 1)
    solar_date = Converter.Lunar2Solar(lunar_date)
    spring_festival = datetime(solar_date.year, solar_date.month, solar_date.day)
    # seventh_day = spring_festival + timedelta(days=7)
    # # 如果初七是周末,找到下一个工作日
    # while seventh_day.weekday() >= 5:  # 5=周六，6=周日
    #     seventh_day += timedelta(days=1)
    return spring_festival

# 找到实际开学日期对应的ILI值（插值或取最近周日的ILI值）
def get_ili_for_date(target_date, data):
    data_dates = data['date']
    time_diffs = [abs((target_date - d).days) for d in data_dates]
    min_diff_idx = time_diffs.index(min(time_diffs))
    return data['ILI'].iloc[min_diff_idx]


data['holiday_effect'] = 0

for year in range(1997,2026):
    # new_year_workday = get_first_workday_after_new_year(year)
    # days_diff = (data['date'] - new_year_workday).dt.days
    # data.loc[(days_diff >= 0) & (days_diff < 7), 'holiday_effect'] = 1


    spring_festival_workday = get_first_monday_after_spring_festival(year)
    days_diff = (data['date'] - spring_festival_workday).dt.days
    data.loc[(days_diff >=0) & (days_diff <= 10), 'holiday_effect'] = 1

data = data[['flu_season_year','flu_weekid','holiday_effect','date','rate.All','pos','ILI','H1N1','H3N2','BIV']]

data.to_csv('data/flu_data_final.csv', index=False)