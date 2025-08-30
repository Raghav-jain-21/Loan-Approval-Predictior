import pandas as pd
data = pd.read_pickle('preprocessed_data.pkl')
print(data.columns.tolist())