import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("mindx_student_model_2025.pkl")

# Title
st.title("📊 Dự đoán số lượng học sinh cho lớp học MindX")

# Input form
st.header("🔍 Nhập thông tin lớp học")
with st.form("predict_form"):
    center = st.selectbox("Trung tâm", [
        "Hoàng Đạo Thuý (HN)", "Long Biên (HN)", "Trần Phú (HN)",
        "Nguyễn Thị Minh Khai (HCM)", "Phan Xích Long (HCM)", "Hạ Long", 
        "Đà Nẵng", "Online"
    ])

    course_line = st.selectbox("Course Line", [
        "Web", "App", "AI", "Game", "Data", "Art & Design"
    ])

    subject = st.selectbox("Môn học", [
        "WEB", "APP", "AI", "GAME", "DATA", "ART", "JSA", "Unknown"
    ])

    class_type = st.selectbox("Loại hình học", [
        "Học tại trung tâm", "Học online", "Học hybrid"
    ])

    month = st.slider("Tháng", 1, 11, 7)
    quarter = (month - 1) // 3 + 1
    num_classes = st.number_input("Số lớp tương tự", min_value=1, value=1, step=1)

    submitted = st.form_submit_button("Dự đoán")

if submitted:
    input_df = pd.DataFrame([{
        'Center': center,
        'Course Line': course_line,
        'Subject': subject,
        'Class Type': class_type,
        'Year': 2024,
        'Month': month,
        'Quarter': quarter,
        'num_classes': num_classes
    }])

    # Dự đoán
    prediction = model.predict(input_df)[0]
    st.success(f"🎯 Dự đoán số học sinh: **{round(prediction)} học sinh**")
