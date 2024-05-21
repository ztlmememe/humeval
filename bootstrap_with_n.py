import concurrent.futures
from simulate_2 import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from get_rate import fit_models

# Function to perform bootstrap and fit models
def bootstrap_and_fit(sub_data):
    sub_data = sub_data.sample(frac=1, replace=True)
    result = dict()
    for dim in range(1, 7):
        result[dim] = fit_models(sub_data[sub_data['dimension'] == dim])
    return result


path_eval_data = 'data/batch100'
df_videos = pd.read_csv('data/videos_all_with_result.csv')
df_videos.fillna(0, inplace=True)

df_list = []

for file in os.listdir(path_eval_data):
    
    if file.endswith('.csv'):
        file_path = os.path.join(path_eval_data, file)
        df = pd.read_csv(file_path)
        
        df_list.append(df)

all_data = pd.concat(df_list, ignore_index=True)
models = ['gen2', 'latte', 'pika', 'vgen', 'videocrafter2']
all_data.head()
videos = pd.read_csv('data/videos_all_with_result.csv')
all_data_merged = pd.merge(all_data, videos[['video_url', 'prompt']], 
                           left_on='video_url_1', right_on='video_url', 
                           how='left')

# Constants
G = 10
dimension_names = {
    1: "Video Quality",
    2: "Temporal Quality",
    3: "Motion Quality",
    4: "Text Alignment",
    5: "Ethical Robustness",
    6: "Human Preference"
}
column_names = ['Latte', 'Pika', 'TF-T2V', 'Gen2', 'Videocrafter2']

# Initialize the DataFrame to collect results
bs_result_with_n = pd.DataFrame()

# Sample sizes
nums = list(range(20, 210, 20))

# Unique prompts
all_prompts = all_data_merged['prompt'].unique()

if __name__ == '__main__':

    for n in nums:
        print(f'Bootstrapping for n={n} started.')
        
        prompts = np.random.choice(all_prompts, n, replace=False)
        sub_data = all_data_merged[all_data_merged['prompt'].isin(prompts)]
        
        # Parallel processing
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(bootstrap_and_fit, [sub_data] * G))
        
        print(f'Bootstrapping for n={n} completed.')

        # Initialize a list of empty DataFrames for each dimension
        result_df_list = [pd.DataFrame(columns=column_names) for _ in range(6)]
        
        # print(type(results))
        # print(results)
        
        # Accumulate results
        for result in results:
            for key, value in result.items():
                result_df_list[key-1] = pd.concat([result_df_list[key-1], pd.DataFrame([value])], ignore_index=True)
        
        # Format and append results
        for i, df in enumerate(result_df_list):
            melted_df = df.melt(var_name='model', value_name='estimation')
            melted_df['dimension'] = dimension_names[i+1]
            melted_df['n'] = n
            bs_result_with_n = pd.concat([bs_result_with_n, melted_df], ignore_index=True)

    # Save to CSV
    bs_result_with_n.to_csv('data/bs_result/bs_result_with_n.csv', index=False)
# bs_result_with_n.head()
