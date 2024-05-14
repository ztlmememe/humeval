import pandas as pd
import statsmodels.api as sm
import krippendorff
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np
import pingouin as pg
import os
# 假设文件名为 file1.csv, file2.csv, file3.csv, file4.csv, file5.csv

# files = [
# '/cpfs01/shared/public/ztl/NIPS/humeval/ori_data_before/ratings_zyc.csv',
# '/cpfs01/shared/public/ztl/NIPS/humeval/ori_data_before/ratings_yyc2.csv',
# '/cpfs01/shared/public/ztl/NIPS/humeval/ori_data_before/ratings_zyc2.csv',
# '/cpfs01/shared/public/ztl/NIPS/humeval/ori_data_before/ratings_ztl.csv',
# '/cpfs01/shared/public/ztl/NIPS/humeval/ori_data_before/ratings_YYC.csv']

# batch_num =100

# files = [
# f'/cpfs01/shared/public/ztl/NIPS/amt/processed/batch{batch_num}/data_for_0.csv',
# f'/cpfs01/shared/public/ztl/NIPS/amt/processed/batch{batch_num}/data_for_1.csv',
# f'/cpfs01/shared/public/ztl/NIPS/amt/processed/batch{batch_num}/data_for_2.csv',
# f'/cpfs01/shared/public/ztl/NIPS/amt/processed/batch{batch_num}/data_for_3.csv',
# f'/cpfs01/shared/public/ztl/NIPS/amt/processed/batch{batch_num}/data_for_4.csv']



# # 读取并合并所有CSV文件
# data_frames = [pd.read_csv(file) for file in files]
# combined_data = pd.concat(data_frames)

# 设置文件夹路径
folder_path = r'D:\pythonProject\humeval\amt_exp\humeval\ori_data_after_sopt'
files = []
# 读取所有CSV文件并将它们合并成一个DataFrame
dfs = []
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        files.append(file_path)
        df = pd.read_csv(file_path)
        dfs.append(df)

# 合并DataFrame
combined_data = pd.concat(dfs, ignore_index=True)

# 按dimension分组，计算每个维度的Fleiss's Kappa
kappa_values = {}

for dim in combined_data['dimension'].unique():
    # 从合并的数据中筛选特定维度的数据
    dim_data = combined_data[combined_data['dimension'] == dim]

    # 根据pairs_id和dimension分组，统计每种评分（0, 1, 2）的频率
    grouped = dim_data.groupby(['pairs_id'])
    rating_counts = grouped['rating'].value_counts().unstack(fill_value=0)

    # 计算并存储该维度的Fleiss's Kappa
    kappa = fleiss_kappa(rating_counts, method='fleiss')
    kappa_values[dim] = kappa

# 打印出每个维度的Fleiss's Kappa值
for dimension, kappa in kappa_values.items():
    print(f"Fleiss's Kappa for dimension {dimension}: {kappa}")


# 读取每个CSV文件，并为每个文件添加一个评分者标识列
data_frames = []
for i, file in enumerate(files):
    df = pd.read_csv(file)
    df['evaluator_id'] = f'evaluator_{i+1}'  # 创建唯一评分者标识
    data_frames.append(df)

# 合并所有CSV文件
combined_data = pd.concat(data_frames)

alpha_values = {}
percent_agreement = {}
icc_values = {}

for dim in combined_data['dimension'].unique():
    dim_data = combined_data[combined_data['dimension'] == dim]
    pivot_data = dim_data.pivot(index='evaluator_id', columns='pairs_id', values='rating')
    ratings = pivot_data.to_numpy()

    # Krippendorff's alpha
    alpha = krippendorff.alpha(ratings, level_of_measurement='nominal')
    alpha_values[dim] = alpha


# Print results
for dimension in alpha_values:
    print(f"Krippendorff's Alpha for dimension {dimension}: {alpha_values[dimension]}")
    

