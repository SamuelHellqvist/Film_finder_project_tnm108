import pandas as pd

csv_path = "test_data/16k_Movies.csv"

df = pd.read_csv(csv_path)
print(df.head())
print(df.shape)

#https://www.kaggle.com/datasets/kashifsahil/16000-movies-1910-2024-metacritic/data

