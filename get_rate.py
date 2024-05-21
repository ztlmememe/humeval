import pandas as pd
import numpy as np
from scipy.optimize import minimize
import random
import os
def log_likelihood_bt(params, comparisons, model_names):
    # Log-likelihood for basic Bradley-Terry model
    param_dict = dict(zip(model_names, params))
    # print(param_dict)
    log_likelihood = 0
    epsilon = 1e-8  # Small constant to prevent division by zero


    for _, row in comparisons.iterrows():
        # print(row['model_1'])
        pi = param_dict[row['model_1']]
        pj = param_dict[row['model_2']]
        if row['rating'] == 0:
            log_likelihood += np.log(pi / (pi + pj))
        elif row['rating'] == 1:
            log_likelihood += np.log(pj / (pi + pj))
    return -log_likelihood

def log_likelihood_rk(params, comparisons, model_names):
    # Log-likelihood for the Rao and Kupper model with ties
    param_dict = dict(zip(model_names, params[:-1]))
    tau = params[-1] # Threshold parameter for ties
    theta = np.exp(tau)
    # print(tau)
    epsilon = 1e-8 
    log_likelihood = 0
    for _, row in comparisons.iterrows():
        pi = param_dict[row['model_1']]
        pj = param_dict[row['model_2']]
        if row['rating'] == 0:
            log_likelihood += np.log(pi / (pi + theta * pj))
        elif row['rating'] == 1:
            log_likelihood += np.log( pj / (theta * pi + pj))
        elif row['rating'] == 2:
            log_likelihood += np.log((pi * pj * (theta**2 - 1)) / ((pi + theta * pj) * (pj + theta * pi)))
    return -log_likelihood

def log_likelihood_bs(params, comparisons, model_names):
    # Log-likelihood for the Baker and Scarf model using a discrete distribution
    param_dict = dict(zip(model_names, params))
    log_likelihood = 0
    for _, row in comparisons.iterrows():
        pi = param_dict[row['model_1']]
        pj = param_dict[row['model_2']]
        theta = 1 / (1 + np.exp(-(pi - pj)))  # Probability of tie modified
        if row['rating'] == 0:
            log_likelihood += np.log(theta)
        elif row['rating'] == 1:
            log_likelihood += np.log(1 - theta)
        elif row['rating'] == 2:
            log_likelihood += np.log(0.5)  # Assuming equal probability for a tie
    return -log_likelihood

def fit_models(comparisons):
    # Load data from CSV
    # comparisons = pd.read_csv(csv_file)
    models = pd.unique(comparisons[['model_1', 'model_2']].values.ravel('K'))
    initial_params = np.array([1.0] * len(models) + [0.5])  # tau for Rao-Kupper
    # print(initial_params)
    # 假设 \(\tau\) 的上限是 10,这里设置了每个参数的预估上下限，防止优化过程中出现很大的负值
    bounds = [(0.01, None)] * len(models) + [(1 , 10)]  # 对 \(\tau\) 设置0到10的范围，其他参数保持正常
    # Modified: 对theta设置1到10的范围

    # result_bt = minimize(log_likelihood_bt, x0=initial_params[:-1], args=(comparisons, models),
    # method='L-BFGS-B', bounds=bounds[:-1])
    
    # scores_bt = dict(zip(models, result_bt.x))
    
    # Fit Rao and Kupper model
    result_rk = minimize(log_likelihood_rk, x0=initial_params, args=(comparisons, models),
                         method='L-BFGS-B', bounds=bounds)
                        #  method='BFGS')
    scores_rk = dict(zip(list(models), result_rk.x))
    
    # # Fit Baker and Scarf model
    # result_bs = minimize(log_likelihood_bs, x0=initial_params[:-1], args=(comparisons, models),
    #                      method='L-BFGS-B', bounds=bounds[:-1])
    #                     #  method='BFGS')
    # scores_bs = dict(zip(models, result_bs.x))
    
    return scores_rk

def rank_models(scores):
    # 将得分字典转换为可排序的元组列表
    if 'tau' in scores:
        del scores['tau']

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # 输出排名
    rankings = {model: rank + 1 for rank, (model, score) in enumerate(sorted_scores)}
    return rankings


def calculate_naive_scores(comparisons):
    # 读取CSV文件
    
    # 初始化一个字典来存储每个模型的得分
    scores = {}
    
    # 遍历每行数据，更新得分
    for _, row in comparisons.iterrows():
        model1 = row['model_1']
        model2 = row['model_2']
        result = row['rating']
        
        # 确保模型已经在字典中
        if model1 not in scores:
            scores[model1] = 0
        if model2 not in scores:
            scores[model2] = 0
        
        # 根据比赛结果更新得分
        if result == 0:  # 模型1胜
            scores[model1] += 1
        elif result == 1:  # 模型2胜
            scores[model2] += 1
        elif result == 2:  # 平局
            scores[model1] += 0.5
            scores[model2] += 0.5
    
    return scores




# files = [
# f'/cpfs01/shared/public/ztl/NIPS/humeval/simulate_amt/0_200_10_5_5_0.3.csv',
# f'/cpfs01/shared/public/ztl/NIPS/humeval/simulate_amt/1_200_10_5_5_0.3.csv',
# f'/cpfs01/shared/public/ztl/NIPS/humeval/simulate_amt/2_200_10_5_5_0.3.csv',
# f'/cpfs01/shared/public/ztl/NIPS/humeval/simulate_amt/3_200_10_5_5_0.3.csv',
# f'/cpfs01/shared/public/ztl/NIPS/humeval/simulate_amt/4_200_10_5_5_0.3.csv']

# files = [
# f'/cpfs01/shared/public/ztl/NIPS/humeval/simulate_amt/0_200_10_5_5_0.5.csv',
# f'/cpfs01/shared/public/ztl/NIPS/humeval/simulate_amt/1_200_10_5_5_0.5.csv',
# f'/cpfs01/shared/public/ztl/NIPS/humeval/simulate_amt/2_200_10_5_5_0.5.csv',
# f'/cpfs01/shared/public/ztl/NIPS/humeval/simulate_amt/3_200_10_5_5_0.5.csv',
# f'/cpfs01/shared/public/ztl/NIPS/humeval/simulate_amt/4_200_10_5_5_0.5.csv']


# # 读取所有 CSV 文件并合并为一个 DataFrame
# dfs = [pd.read_csv(file) for file in files]
# df = pd.concat(dfs, ignore_index=True)

# 设置文件夹路径
# folder_path = '/cpfs01/shared/public/ztl/NIPS/humeval/ori_data_before'
# folder_path = r'D:\pythonProject\humeval\amt_exp\humeval\ori_data_after'
# # 读取所有CSV文件并将它们合并成一个DataFrame
# dfs = []
# for file_name in os.listdir(folder_path):
#     if file_name.endswith('.csv'):
#         file_path = os.path.join(folder_path, file_name)
#         df = pd.read_csv(file_path)
#         dfs.append(df)

# # 合并DataFrame
# df = pd.concat(dfs, ignore_index=True)

# # file_path = r'D:\pythonProject\humeval\amt_exp\humeval\ori_data_after\trained_data_4.csv'

# # df = pd.read_csv(file_path)


# # 初始化一个空字典用于存储每个dimension的DataFrame
# dimension_dfs = {}


# # 根据dimension分组并提取所需列
# for dimension, group_df in df.groupby('dimension'):
#     # print(dimension)
#     # if dimension != 5:
#     #     pass
#     # else:


#         comparisons = group_df[['pairs_id', 'model_1', 'model_2', 'rating']].reset_index(drop=True)


#         scores_bt, scores_rk = fit_models(comparisons)


#         # 获取排名
#         # rankings_bt = rank_models(scores_bt)
#         rankings_rk = rank_models(scores_rk)


#         # 打印结果
#         # print(f"Dimension{dimension}: Bradley-Terry Model Rankings:", rankings_bt)
#         print(f"Dimension{dimension}: Rao and Kupper Model scores:", scores_rk)

#         print(f"Dimension{dimension}: Rao and Kupper Model Rankings:", rankings_rk)


    # scores_win = calculate_naive_scores(comparisons)
    # rankings_win = rank_models(scores_win)
    # print(f"Dimension{dimension}: Win ratio ranking:", rankings_win)

