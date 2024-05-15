# humeval

get_rate.py: 基于一个文件夹下的打分结果，通过RK模型计算每个维度下的模型分数和排名

iaa_allcombi.py：遍历AMT人员和LAB人员的所有组合，共250组，计算平均IAA

iaa_before.py：计算单一来源，一个文件夹下五个人打分结果的IAA

simulate.py：模拟动态评测算法

输入：df的列表
输出：dimension_dfs，一个包含每个维度下模型评分的字典

使用示例：
```python
from simulate_1 import simulate_all
from simulate_1 import simulate_score
import argparse
import pandas as pd
import os
parser = argparse.ArgumentParser(description='Example of a script that accepts hyperparameters.')
parser.add_argument('--begain_count', type=int, default=200)
parser.add_argument('--groups_per_batch', type=int, default=10) # 每一个BETCH过groups_per_batch*10个评分数据
parser.add_argument('--M', type=int, default=5) # 过M个BATCH更新一次模型强度
parser.add_argument('--N', type=int, default=5) # N次检查排名稳定后停止一个维度下的打分
parser.add_argument('--decay_rate', type=float, default=0.1)
parser.add_argument('--num', type=int, default=0) # 模拟第几个人,0-4 100 18:08开始

args = parser.parse_args()

path_csv_per_person = r'/mnt/workspace/ztl/NIPS/humeval/ori_data_after'
csv_videos = r'videos_all_with_result.csv'

df_videos  = pd.read_csv(csv_videos)

df_videos['image_url'].fillna(0, inplace=True)

df_inputs = []

for file in os.listdir(path_csv_per_person):
        
    if file.endswith(".csv"):
        file_path = os.path.join(path_csv_per_person, file)
        
        df_input = pd.read_csv(file_path)

        df_inputs.append(df_input)

simulate_all(df_inputs,df_videos,begain_count = args.begain_count ,groups_per_batch =args.groups_per_batch ,M =args.M ,N =args.N ,decay_rate =args.decay_rate)
```
相关参数：


评测文件夹的目录
path_csv_per_person = r'/mnt/workspace/ztl/NIPS/amt/processed/batch100'


视频文件夹的目录
csv_videos = r'videos_all_with_result.csv'


动态测评结果存放目录
path_csv_save = r'/cpfs01/shared/public/ztl/NIPS/humeval/simulate_amt/'


动态测评相关参数

begain_count = 200 动态评测开始前需要测多少个

groups_per_batch =10 # 每一个BETCH过groups_per_batch*10个评分数据

M =5 # 过M个BATCH更新一次模型强度

N =5 # N 次检查排名稳定后停止该维度下的打分（暂时）

decay_rate =0.3 # 调整drop的概率
