import pandas as pd
import os

os.chdir(r'd:\\university\\Semester 6\\Minor Project\\loan-approval-app\\Backend')
print("Current working directory:", os.getcwd())
df = pd.read_csv('loan_data.csv')
print(df.head())
print(df.shape)