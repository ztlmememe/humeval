# humeval

get_rate.py: 基于一个文件夹下的打分结果，通过RK模型计算每个维度下的模型分数和排名

iaa_allcombi.py：遍历AMT人员和LAB人员的所有组合，共250组，计算平均IAA

iaa_before.py：计算单一来源，一个文件夹下五个人打分结果的IAA

simulate.py：模拟动态评测算法

from simulate_1 import simulate_all

simulate_all()

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
