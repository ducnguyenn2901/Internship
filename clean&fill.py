import pandas as pd
df = pd.read_excel("[2024 - TTCM] lms-class-data.xlsx")
course_lines = ['XART', 'ROB', 'C4K', 'C4T']
status = ['FINISHED', 'RUNNING', 'OPEN', 'PRE_OPEN', 'NEW', 'PENDING', 'PREPARING']
df_k12 = df[(df['Course Line'].isin(course_lines)) & (df['Status'].isin(status))]
df_k12_2024 = df_k12[
    (df_k12['Start date'] >= '2024-01-01') &
    (df_k12['Start date'] < '2024-11-01')
]
df_k12_2024_filled = df_k12.copy()
for col in df_k12_2024_filled.columns:
    if pd.api.types.is_numeric_dtype(df_k12_2024_filled[col]):
        df_k12_2024_filled[col] = df_k12_2024_filled[col].fillna(df_k12_2024_filled[col].mean())
    else:
        df_k12_2024_filled[col] = df_k12_2024_filled[col].fillna(method='ffill').fillna(method='bfill')
for col in df_k12_2024_filled.columns:
    if pd.api.types.is_numeric_dtype(df_k12_2024_filled[col]):
        df_k12_2024_filled[col] = df_k12_2024_filled[col].round(0)
df_k12_2024_filled.to_excel('k12_class_data.xlsx', index=False)