import pandas as pd
import statsmodels.api as sm
import krippendorff
import pandas as pd
from statsmodels.stats.inter_rater import fleiss_kappa
import numpy as np
import pingouin as pg
import os


import pandas as pd
import numpy as np
from scipy.optimize import minimize
import random
from itertools import combinations
import random
import csv
from collections import deque
import math
import argparse

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

def log_likelihood_rk(params, comparisons, model_names):
    # Log-likelihood for the Rao and Kupper model with ties
    param_dict = dict(zip(model_names, params[:-1]))
    tau = params[-1] # Threshold parameter for ties
    theta = np.exp(tau)

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
def fit_models(comparisons):


    models = pd.unique(comparisons[['model_1', 'model_2']].values.ravel('K'))
    initial_params = np.array([1.0] * len(models) + [0.5])  # tau for Rao-Kupper

    # 假设 \(\tau\) 的上限是 10,这里设置了每个参数的预估上下限，防止优化过程中出现很大的负值
    bounds = [(0.01, None)] * len(models) + [(1e-8 , 10)]  # 对 \(\tau\) 设置0到10的范围，其他参数保持正常
    
    # Fit Rao and Kupper model
    result_rk = minimize(log_likelihood_rk, x0=initial_params, args=(comparisons, models),
                         method='L-BFGS-B', bounds=bounds)
                        #  method='BFGS')
    scores_rk = dict(zip(list(models), result_rk.x))

    # result_bt = minimize(log_likelihood_bt, x0=initial_params[:-1], args=(comparisons, models),
    # method='L-BFGS-B', bounds=bounds[:-1])
    
    # scores = dict(zip(models, result_bt.x))

    # scores = calculate_naive_scores(comparisons)
    # return scores
    return scores_rk 
    # return scores_bs

def rank_models(scores):
    # 将得分字典转换为可排序的元组列表
    if 'tau' in scores:
        del scores['tau']

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    # 输出排名
    rankings = {model: rank + 1 for rank, (model, score) in enumerate(sorted_scores)}
    return rankings

def calculate_similarity(v1, v2):
    # Parameter to control the rate of decay, can be adjusted based on specific requirements
    decay_rate = 0.1
    difference = abs(v1['total_score'] - v2['total_score'])
    similarity_index = math.exp(-decay_rate * difference)
    return similarity_index

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


def submit_result(update_count,comparisons_by_dimension, count, rank_per_dimension,eval_status_per_dimension, model_strengths_per_dimension, df_save,final_ranking,final_score,N,pairs_id,video_url_1,video_url_2, model_1,model_2,ratings):


    # 遍历每个评分项，更新对应维度的数据 
    for index, rating in ratings.iterrows():
        dimension = rating['dimension']

        rate_value = rating['rating']
        
        # 创建一个新的结果行
        new_result = pd.DataFrame({
            'pairs_id': [pairs_id],
            'model_1': [model_1],
            'model_2': [model_2],
            'rating': [int(rate_value)]
        }, index=[0])


                # 创建新行数据
        new_row = pd.DataFrame({
            'pairs_id': [pairs_id],
            'model_1': [model_1],
            'model_2': [model_2],
            'video_url_1': [video_url_1],
            'video_url_2': [video_url_2],
            'dimension': [dimension],
            'rating': [int(rate_value)]
        }, index=[0])
        
        
        # 将新行数据添加到DataFrame
        df_save = pd.concat([df_save, new_row], ignore_index=True)


        # 将新结果合并到对应维度的DataFrame中
        comparisons_by_dimension[dimension] = pd.concat([comparisons_by_dimension[dimension], new_result], ignore_index=True)
        count[dimension] += 1
        
        # 检查每个维度是否达到更新阈值
        if count[dimension] % update_count == 0:
            # 调用模型拟合函数更新模型强度
            scores = fit_models(comparisons_by_dimension[dimension])

            # print(f"Updated scores for dimension {dimension}: {scores}")

            model_strengths_per_dimension[dimension].update(scores)

            rankings_per = rank_models(scores)

            # 更新和跟踪模型排名
            rank_per_dimension[dimension].append(rankings_per)

            # 如果排名历史足够（至少五个），则检查最后五个排名是否稳定
            if len(rank_per_dimension[dimension]) >= N:
                # 检查最后五个排名是否一致
                last_five_rankings = list(rank_per_dimension[dimension])[-N:]  # 获取最后五个排名

                if all(rank == last_five_rankings[0] for rank in last_five_rankings):
                    eval_status_per_dimension[dimension] = True  # 标记该维度评测完成

                    # print(f"Dimension {dimension} evaluation completed.")
                    final_ranking[dimension] = last_five_rankings[0]  # 存储最终排名
                    final_score[dimension] = scores
                    # rank_per_dimension[dimension].clear()  # 清空历史数据，为下一轮评测准备
                else:
                    eval_status_per_dimension[dimension] = False

            
            count[dimension] = 0  # 重置计数器
            

    return df_save,all(eval_status_per_dimension.values()) 

def get_videos(update_count,comparisons_by_dimension,all_combinations, count, begain_count,eval_status_per_dimension, model_strengths_per_dimension, current_group_index, groups_per_batch,decay_rate):
 
    n = 10  # 每组10个模型组合
    size_of_group = groups_per_batch * n
    total_groups = len(all_combinations) // size_of_group

    # print(total_groups)
    # print(current_group_index)

    # 检查是否所有维度都已有足够的数据，否则返回所有组合
    if len(comparisons_by_dimension[1]) < begain_count :

        # 在评分数据不足时返回所有可能的视频对组合 
        return all_combinations[:begain_count], current_group_index
    else:
        
        # 评分数据足够后，按8*10*N划分组

        # print(total_groups)

        # 检查当前组索引是否超出范围，如果超出则重置
        if current_group_index >= total_groups:
            return [],current_group_index
        
        # 获取当前批次的视频对
        start_index = current_group_index * size_of_group
        end_index = start_index + size_of_group
        current_combinations = all_combinations[start_index:end_index]
        
        # 更新当前展示组数
        

        # 筛选视频对基于模型强度
        selected_combinations = []
        dimension_to_use = next((d for d in eval_status_per_dimension if not eval_status_per_dimension[d]), None)


        
        for combo_pair in current_combinations:
            model1 = combo_pair[0]['models']
            model2 = combo_pair[1]['models']
            # decay_rate = 0.2  # 提高衰减率会使得保留概率对差异更加敏感。在差异小的时候，保留概率接近 1，而差异大的时候，保留概率迅速接近 0。

            strength_diff = abs(model_strengths_per_dimension[dimension_to_use][model1] - model_strengths_per_dimension[dimension_to_use][model2])
            probability_to_keep = math.exp(-decay_rate * strength_diff)  # 使用指数衰减公式计算保留概率

            if random.random() < probability_to_keep:
                selected_combinations.append(combo_pair)

        return selected_combinations,current_group_index

import csv

def rate(update_count,comparisons_by_dimension, count, rank_per_dimension,eval_status_per_dimension, model_strengths_per_dimension, df_save,final_ranking,final_score,N,pairs_id,video_url_1,video_url_2, model_1,model_2,ratings,count_num):


    # 写入CSV文件
    
    # update_model_strengths(models[0], models[1], ratings)
                
    df_save,down = submit_result(update_count,comparisons_by_dimension, count, rank_per_dimension,eval_status_per_dimension, model_strengths_per_dimension, df_save,final_ranking,final_score,N,pairs_id,video_url_1,video_url_2, model_1,model_2,ratings)

    if down :

        return True,df_save
    else:
        # print('Rating received successfully')
        return False,df_save

def simulate_score(df_input = None,df_videos= None,
                    begain_count = 200 ,groups_per_batch =7 ,M =5 ,N =5 ,decay_rate = 0.3):

    # 创建空的DataFrame
    df_save = pd.DataFrame(columns=['pairs_id', 'model_1', 'model_2', 'video_url_1', 'video_url_2', 'dimension', 'rating'])


    ratings = []

    # 初始化模型强度
    model_strengths = {model: [1]*6 for model in df_videos['models'].unique()}
    all_combinations = []  # 存储所有可能的组合
    # 存储比赛结果的全局变量，针对每个维度
    comparisons_by_dimension = {i: pd.DataFrame(columns=['model_1', 'model_2', 'rating']) for i in range(1, 7)}

    rank_per_dimension = {i: deque(maxlen=6) for i in range(1, 7)}  # 维度从1到6，保留最新的六个排名，方便进行比较
    model_strengths_per_dimension = {i: {model: 1 for model in df_videos['models'].unique()} for i in range(1, 7)}

    count = {i: 0 for i in range(1, 7)}  # 用于跟踪每个维度的数据计数


    begain_count = begain_count # 测评多少开始动态评测
    # 200 # 测评多少开始动态评测

    groups_per_batch =  groups_per_batch  # 每个BATCH的视频对数量，包括每个模型对 10

    update_count = groups_per_batch*M # 过M个BATCH更新一次模型强度

    eval_status_per_dimension = {i: False for i in range(1, 7)}  # 每个维度的评测完成状态

    # 归一化分数
    combinations_type = 'quality' # 'quality' or 'all'
    # combinations_type = 'all'
    if combinations_type == 'quality':
    # 分组
        score_columns = ['subject_consistency', 'temporal_flickering', 'motion_smoothness',
                        'dynamic_degree', 'aesthetic_quality', 'imaging_quality', 'overall_consistency']

        df_videos[score_columns] = df_videos[score_columns].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        df_videos['total_score'] = df_videos[score_columns].sum(axis=1)
        grouped =df_videos.groupby(['prompt', 'image_url'])

        group_combinations = []
        for _, group in grouped:
            videos = group.to_dict('records')

            model_pairs = list(combinations(videos, 2))  # 创建视频对组合
            
            # 计算每对模型组合的相似度指标
            group_similarity = 0
            for v1, v2 in model_pairs:
                similarity_index = calculate_similarity(v1 , v2)
                # similarity_index = abs(v1['total_score'] - v2['total_score'])
                # abs(v1['total_score'] - v2['total_score'])
                group_similarity += similarity_index

            group_combinations.append((group_similarity, model_pairs))

        # 按总相似性指标排序，相似性指标越高表示组内差异越大，排序越靠前
        group_combinations.sort(reverse=True, key=lambda x: x[0])

        # 按总相似性指标排序，相似性指标越高表示组内差异越小，排序越靠前
        # group_combinations.sort(reverse=False, key=lambda x: x[0])

        # print(len(group_combinations[0][1])) # group_combinations 中第一个元素是组合分数，第二个是组合内容


        all_combinations = [] # 每一个prompt有10个组合，每个组合内部获取一个分数，按分数排序
        for _, pairs in group_combinations:

            for pair in pairs:
                all_combinations.append(list(pair))  # 转换元组为列表


    elif combinations_type == 'all':
        grouped = df_videos.groupby(['prompt', 'image_url'])
        for _, group in grouped:
            models_list = group['models'].unique()
            model_combinations = list(combinations(models_list, 2))
            for combo in model_combinations:
                combo_videos = group[group['models'].isin(combo)]
                if len(combo_videos) == 2:
                    all_combinations.append(combo_videos.to_dict('records'))


    # print(len(all_combinations))

    final_ranking = {i: {} for i in range(1, 7)}  # 存储最终排名
    final_score = {i: {} for i in range(1, 7)}  # 存储最终排名
    # 读取结果数据
    result_df = df_input
    current_group_index = 0



    end = False
    count_num = 0
    while True:
        combinations_now,current_group_index = get_videos(update_count,comparisons_by_dimension,all_combinations, count, begain_count,eval_status_per_dimension, model_strengths_per_dimension, current_group_index, groups_per_batch,decay_rate)
        # print(f'combinations_now {len(combinations_now)}') selected_combinations,current_group_index


        # 如果获取的视频对列表为空，结束循环
        if len(combinations_now) == 0:
            # print("No more video pairs to process.")
            break
        current_group_index += 1

        # 初始化计数器

        # 遍历每个视频对
        for pair in combinations_now:
            # 获取当前 pair 在 all_combinations 中的索引位置
            pairs_id = next((i for i, comb in enumerate(all_combinations) if comb == pair), None)
            video_url_1 = pair[0]['video_url']
            video_url_2 = pair[1]['video_url']
            model_1 = pair[0]['models']
            model_2 = pair[1]['models']
            
            # 在 CSV 中查找对应的评分
            ratings = result_df[(result_df['video_url_1'] == video_url_1) & (result_df['video_url_2'] == video_url_2)]

            # 打印查找到的评分信息
            # print(ratings)

            num_batches = len(ratings) // 6

            # 对ratings中的每个六行批次进行迭代
            for j in range(num_batches):
                # 计算当前批次的索引范围
                start_index = j * 6
                end_index = start_index + 6
                
                # 使用iloc获取从start_index到end_index的六行数据
                batch_rating = ratings.iloc[start_index:end_index]
                # print(batch_rating)
                

                count_num += 1
            
            # 调用 rate 函数处理评分数据
                end,df_save = rate(update_count,comparisons_by_dimension, count, rank_per_dimension,eval_status_per_dimension, model_strengths_per_dimension, df_save,final_ranking,final_score,N,pairs_id, video_url_1, video_url_2, model_1, model_2, batch_rating, count_num)
                if end:
                    break

            if end:
                    break
        
        if len(combinations_now) == 2000:
            break
        
        # 打印完成一轮视频对处理的消息
        # print("Completed processing one batch of video pairs.")
        if end:
            break
                
    print('The video evaluation is complete',f'completed_pairs {count_num}')       
    dimension_dfs = {}
    
    # 根据dimension分组并提取所需列
    for dimension, group_df in df_save.groupby('dimension'):
        # print(dimension)

        comparisons = group_df[['pairs_id', 'model_1', 'model_2', 'rating']].reset_index(drop=True)


        scores_rk = fit_models(comparisons)


        # 获取排名
        # rankings_bt = rank_models(scores_bt)
        rankings_rk = rank_models(scores_rk)

        dimension_dfs[dimension] = scores_rk

        # print(f"Dimension{dimension}: Rao and Kupper Model Rankings:", rankings_rk)

    return df_save, dimension_dfs



import time


def simulate_all(df_inputs,df_videos,begain_count = 200 ,groups_per_batch =7 ,M =5 ,N =5 ,decay_rate =0.3):


    i = 0

    dfs = []

    for df_input  in df_inputs:

        df_save, dimension_dfs = simulate_score(df_input,df_videos,begain_count,groups_per_batch,M,N,decay_rate)

        dfs.append(df_save)

        i += 1

    # 合并DataFrame
    df = pd.concat(dfs, ignore_index=True)

    # 初始化一个空字典用于存储每个dimension的DataFrame
    dimension_dfs = {}

    # 根据dimension分组并提取所需列
    for dimension, group_df in df.groupby('dimension'):
        # print(dimension)

        comparisons = group_df[['pairs_id', 'model_1', 'model_2', 'rating']].reset_index(drop=True)

        scores_rk = fit_models(comparisons)

        rankings_rk = rank_models(scores_rk)

        dimension_dfs[dimension] = scores_rk

        print(f"Dimension{dimension}: Rao and Kupper Model Rankings:", rankings_rk)


        print(f"Dimension{dimension}: Rao and Kupper Model Scores:", scores_rk)

    return dimension_dfs

