import pandas as pd

tsv_file_path = 'sample_data.tsv' 

loaded_df = pd.read_csv(tsv_file_path, sep='\t')

print("読み込んだDataFrame:")
print(loaded_df)