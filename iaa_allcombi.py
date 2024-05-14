import os
import pandas as pd
import random
from itertools import combinations
import itertools
# 设置两个文件夹路径

import pandas as pd
import statsmodels.api as sm
import krippendorff
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np
import pingouin as pg
import os
import pandas as pd
import itertools

folder1 = r'D:\pythonProject\humeval\amt_exp\humeval\ori_data_after_sopt'
folder2 = r'D:\pythonProject\humeval\amt_exp\humeval\amt_sopt'

  # 获取所有文件路径
files1 = [os.path.join(folder1, f) for f in os.listdir(folder1) if f.endswith('.csv')]
files2 = [os.path.join(folder2, f) for f in os.listdir(folder2) if f.endswith('.csv')]

  # 所有可能的组合
all_combinations = list(itertools.combinations(files1 + files2, 5))


alpha_values_by_dimension = {}  # 存储每个维度的Alpha值
kappa_values_by_dimension = {}  # 存储每个维度的Alpha值

  # 合并并打印每个组合
for combination in all_combinations:
    dfs = []
    files = []

    for file in combination:
        df = pd.read_csv(file)
        dfs.append(df)
        files.append(file)

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



    # 读取每个CSV文件，并为每个文件添加一个评分者标识列
    data_frames = []
    for i, file in enumerate(files):
        df = pd.read_csv(file)
        df['evaluator_id'] = f'evaluator_{i+1}'  # 创建唯一评分者标识
        data_frames.append(df)

    # 合并所有CSV文件
    combined_data = pd.concat(data_frames)

    alpha_values = {}


    for dim in combined_data['dimension'].unique():
        dim_data = combined_data[combined_data['dimension'] == dim]
        pivot_data = dim_data.pivot(index='evaluator_id', columns='pairs_id', values='rating')
        ratings = pivot_data.to_numpy()

        # Krippendorff's alpha
        alpha = krippendorff.alpha(ratings, level_of_measurement='nominal')
        alpha_values[dim] = alpha

    # for dimension, kappa in kappa_values.items():
    #     print(f"Fleiss's Kappa for dimension {dimension}: {kappa}")
    # # Print results
    # for dimension in alpha_values:
    #     print(f"Krippendorff's Alpha for dimension {dimension}: {alpha_values[dimension]}")

        # 将每个维度的Fleiss's Kappa值添加到alpha_values_by_dimension中 kappa_values_by_dimension = {}  # 存储每个维度的Alpha值
    for dim, alpha in alpha_values.items():
        if dim not in alpha_values_by_dimension:
            alpha_values_by_dimension[dim] = []
        alpha_values_by_dimension[dim].append(alpha)

    for dim, kappa in kappa_values.items():
        if dim not in kappa_values_by_dimension:
            kappa_values_by_dimension[dim] = []
        kappa_values_by_dimension[dim].append(kappa)


# 计算每个维度下 Krippendorff's Alpha 的均值
mean_alpha_values = {}
for dim, alpha_list in alpha_values_by_dimension.items():
    mean_alpha_values[dim] = sum(alpha_list) / len(alpha_list)

mean_kappa_values = {}
for dim, kappa_list in kappa_values_by_dimension.items():
    mean_kappa_values[dim] = sum(kappa_list) / len(kappa_list)

# 打印每个维度下 Krippendorff's Alpha 的均值
for dimension, mean_kappa in mean_kappa_values.items():
    print(f"Mean Fleiss's Kappa for dimension {dimension}: {mean_kappa}")


# 打印每个维度下 Krippendorff's Alpha 的均值
for dimension, mean_alpha in mean_alpha_values.items():
    print(f"Mean Krippendorff's Alpha for dimension {dimension}: {mean_alpha}")

