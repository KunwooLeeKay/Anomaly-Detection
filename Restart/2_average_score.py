
import pandas as pd

df = pd.read_excel("No_Sampling_Result0.xlsx")

print(df['No Sampling'])

for i in range(1,10):
    df_temp = pd.read_excel("No_Sampling_Result"+str(i)+".xlsx")
    df['No Sampling'] += df_temp['No Sampling']

print(df['No Sampling'])
df['No Sampling'] = df['No Sampling'] / 10
df['No Sampling'] = df['No Sampling'].round(3)

df.to_excel("NOSAMPLING.xlsx")