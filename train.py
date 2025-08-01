import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def extract_subject_code(class_name):
    try:
        match = re.search(r'-(?P<code>[A-Z]+)', class_name)
        if match:
            return match.group('code')
        elif re.fullmatch(r'[A-Z]{2,}', class_name):
            return class_name
        else:
            return 'Unknown'
    except:
        return 'Unknown'



def detect_class_type(name):
    name = str(name).upper()
    if 'ONLINE' in name or 'ONL' in name:
        return 'Học online'
    elif 'HYBRID' in name or 'HB' in name:
        return 'Học hybrid'
    elif '1:1' in name or '1:2' in name or '1:3' in name:
        return None
    match = re.search(r'\(([^)]+)\)', name)
    if match:
        return match.group(1).strip()
    return 'Học tại trung tâm'

path = "k12_class_data.xlsx"
def load_and_preprocess(path):
    df = pd.read_excel(path)

    df['Student count'] = (
        pd.to_numeric(df['Student count'], errors='coerce')
          .fillna(0).astype(int)
    )
    df['Start date'] = pd.to_datetime(df['Start date'], errors='coerce')
    df['Year']    = df['Start date'].dt.year
    df['Month']   = df['Start date'].dt.month
    df['Quarter'] = df['Start date'].dt.quarter

    # df = df[(df['Year']==2024) & (df['Month']<=11)].copy()
    df['Subject'] = df['Course'].apply(extract_subject_code)
    df['Class Type'] = df['Class name'].apply(detect_class_type)
    df = df[df['Class Type'].notna()]
    return df

def aggregate(df):
    agg = (
        df.groupby(
            ['Center','Course Line','Subject','Class Type','Year','Month','Quarter'],
            as_index=False
        )
        .agg(
            total_students=('Student count','sum'),
            num_classes=('Class name','nunique')
        )
    )
    return agg

def build_pipeline(cat_cols, num_cols):
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
        ('num', 'passthrough', num_cols)
    ])
    model = Pipeline([
        ('preproc', preprocessor),
        ('lr', LinearRegression())
    ])
    return model

def train_evaluate_save(df_agg, out_model='mindx_student_model_2025.pkl'):
    features    = ['Center','Course Line','Subject','Class Type','Year','Month','Quarter','num_classes']
    target      = 'total_students'
    cat_cols    = ['Center','Course Line','Subject','Class Type']
    num_cols    = ['Year','Month','Quarter','num_classes']
    X = df_agg[features]
    y = df_agg[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    # Build & train
    pipeline = build_pipeline(cat_cols, num_cols)
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)
    mae  = mean_absolute_error(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2   = r2_score(y_te, y_pred)

    print(f"MAE : {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R2  : {r2:.3f}")

    # Lưu model và kết quả aggregate
    joblib.dump(pipeline, out_model)
    # df_agg.to_excel("model.xlsx", index=False)
    # print(f"\n– Model đã lưu vào: {out_model}")
    # print("– Bảng aggregate đã lưu vào: model.xlsx")
if __name__ == "__main__":
    df = load_and_preprocess("k12_class_data.xlsx")
    df_agg = aggregate(df)
    train_evaluate_save(df_agg)
