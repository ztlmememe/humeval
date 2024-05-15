from concurrent.futures import ProcessPoolExecutor
import os
import pandas as pd
import numpy as np
from simulate_2 import simulate_all

# Load data
path_eval_data = 'data/batch100'
df_videos = pd.read_csv('data/videos_all_with_result.csv')
df_videos.fillna(0, inplace=True)

# Load all CSV files into DataFrame list
df_list = [pd.read_csv(os.path.join(path_eval_data, file)) for file in os.listdir(path_eval_data) if file.endswith('.csv')]

def bootstrap_iteration(df_list, df_videos):
    bs_df_list = []
    # Resampling
    for df in df_list:
        indices = np.random.choice(df.index, len(df), replace=True)
        bs_df_list.append(df.iloc[indices])
    # Simulate
    return simulate_all(bs_df_list, df_videos)

# Number of Bootstrap iterations
G = 100

# Execute Bootstrap in parallel
with ProcessPoolExecutor() as executor:
    bs_results = list(executor.map(lambda _: bootstrap_iteration(df_list, df_videos), range(G)))

# Post-processing to save results
result_df_list = [pd.DataFrame() for _ in range(6)]
for result in bs_results:
    for key, value in result.items():
        result_df_list[key-1] = pd.concat([result_df_list[key-1], pd.DataFrame([value])], ignore_index=True)

# Save to CSV
if not os.path.exists('data/bs_result'):
    os.makedirs('data/bs_result')

for i, df in enumerate(result_df_list, 1):
    df.to_csv(f'data/bs_result/bs_result_{i}.csv', index=False)
