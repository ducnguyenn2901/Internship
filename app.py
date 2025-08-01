import streamlit as st
import pandas as pd
import pickle
import altair as alt

with open("mindx_student_model_2025.pkl", "rb") as f:
    model = pickle.load(f)
df = pd.read_excel("model.xlsx")
df = df.reset_index(drop=True)

# ==== Mapping ====
center_map = {
    '29T1HDT': 'Hoàng Đạo Thuý (HN)', '25LB': 'Long Biên (HN)', '10TP': 'Trần Phú (HN)',
    '505MK': 'Minh Khai (HN)', '71NCT': 'Nguyễn Chí Thanh (HN)', '06NHT': 'Nguyễn Hữu Thọ (HN)',
    '107NPS': 'Nguyễn Phong Sắc (HN)', '98NVC': 'Nguyễn Văn Cừ (HN)', 'VHHN': 'Vinhomes Hàm Nghi (HN)',
    'KĐT VP': 'KĐT Văn Phú (HN)', '22CTC': 'Thành Công (HN)', '176DC': 'Định Công (HN)',
    '672A28PVT': 'Phan Văn Trị (HCM)', '01TC': 'Trường Chinh (HCM)', '02SH': 'Song Hành (HCM)',
    '223NX': 'Nguyễn Xí (HCM)', '01QT': 'Quang Trung (HCM)', '261-263PXL': 'Phan Xích Long (HCM)',
    '165-167NTT': 'Nguyễn Thị Thập (HCM)', '01TK': 'Tô Ký (HCM)', '6183/2': '3 tháng 2 (HCM)',
    '490 PTB': 'Phạm Thái Bường (HCM)', '120-122PVĐ': 'Phạm Văn Đồng (HCM)', '174TL': 'Tên Lửa (HCM)',
    '322TT': 'Tây Thạnh (HCM)', '39HTLO': 'Hải Thượng Lãn Ông (HCM)', '343PNL': 'Phạm Ngũ Lão (HCM)',
    '99LVV': 'Lê Văn Việt (HCM)', '414LBB': 'Luỹ Bán Bích (HCM)', '624LLQ': 'Lạc Long Quân (HCM)',
    '22-24UVK': 'Ung Văn Khiêm (HCM)', 'HN-ONLINE': 'HN Online', 'HCM-Online': 'HCM Online',
    'DArt': 'Digital Art Online'
}

course_line_map = {
    'XART': 'Art & Design', 'ROB': 'Robotics',
    'C4K': 'Code for Kids', 'C4T': 'Code for Teens'
}

course_code_map = {
    'KAB': 'Kid Arts Basic', 'KAA': 'Kid Arts Advanced', 'KAI': 'Kid Arts Intensive',
    'VAB': 'Visual Arts Basic', 'VAA': 'Visual Arts Advanced', 'VAI': 'Visual Arts Intensive',
    'VCB': 'Visual Creation Basic', 'VCA': 'Visual Creation Advanced', 'VCI': 'Visual Creation Intensive',
    'GDB': 'Graphic Design Basic', 'GDA': 'Graphic Design Advanced', 'GDI': 'Graphic Design Intensive',
    'MDB': 'Multimedia Design Basic', 'MDA': 'Multimedia Design Advanced', 'MDI': 'Multimedia Design Intensive',
    'DAB': 'Digital Animation Basic', 'DAA': 'Digital Animation Advanced', 'DAI': 'Digital Animation Intensive',
    'IDB': 'Interactive Design Basic', 'IDA': 'Interactive Design Advanced', 'IDI': 'Interactive Design Intensive',
    'PREB': 'Robotics Năm 1 Basic', 'PREA': 'Robotics Năm 1 Advanced', 'PREI': 'Robotics Năm 1 Intensive',
    'ARMB': 'Robotics Năm 2 Basic', 'ARMA': 'Robotics Năm 2 Advanced', 'ARMI': 'Robotics Năm 2 Intensive',
    'SEMIB': 'Robotics Năm 3 Basic', 'SEMIA': 'Robotics Năm 3 Advanced', 'SEMII': 'Robotics Năm 3 Intensive',
    'AUTO': 'Robotics Năm 4',
    'SB': 'Scratch Creator Basic', 'SA': 'Scratch Creator Advanced', 'SI': 'Scratch Creator Intensive',
    'GB': 'Game Creator Basic', 'GA': 'Game Creator Advanced', 'GI': 'Game Creator Intensive',
    'PTB': 'App Developer Basic', 'PTA': 'App Developer Advanced', 'PTI': 'App Developer Intensive',
    'JSB': 'Web Developer Basic', 'JSA': 'Web Developer Advanced', 'JSI': 'Web Developer Intensive',
    'CSB': 'Computer Scientist Basic', 'CSA': 'Computer Scientist Advanced', 'CSI': 'Computer Scientist Intensive',
}

df['Center Name'] = df['Center'].map(center_map)
df['Course Line Name'] = df['Course Line'].map(course_line_map)
df['Subject Name'] = df['Subject'].map(course_code_map)

st.title("📊 Dự đoán số học sinh MindX 2025")

center_options = {v: k for k, v in center_map.items()}
selected_center_name = st.selectbox("Chọn trung tâm:", sorted(center_options.keys()))
center = center_options[selected_center_name]

month_options = ['Cả năm'] + sorted(df['Month'].unique())
month = st.selectbox("Chọn tháng:", month_options)

if month == 'Cả năm':
    subset_2024 = (
        df[df['Center'] == center]
        .groupby(['Course Line', 'Subject', 'Class Type'])
        .agg({
            'num_classes': 'sum',
            'total_students': 'sum'
        })
        .reset_index()
    )
    subset_2024['Month'] = 0
else:
    subset_2024 = df[(df['Center'] == center) & (df['Month'] == month)]


if subset_2024.empty:
    st.warning("⚠️ Không tìm thấy dữ liệu lớp học cho trung tâm và tháng này trong năm 2024.")
else:
    st.subheader("Lớp từng mở trong 2024")
    subset_2024['Course Line Name'] = subset_2024['Course Line'].map(course_line_map)
    subset_2024['Subject Name'] = subset_2024['Subject'].map(course_code_map)

    st.dataframe(
        subset_2024[[
            'Course Line Name', 'Subject Name', 'Class Type',
            'num_classes', 'total_students'
        ]].rename(columns={
            'Course Line Name': 'Khoá học',
            'Subject Name': 'Môn học',
            'Class Type': 'Loại lớp',
            'num_classes': 'Số lớp',
            'total_students': 'Tổng học sinh'
        }).reset_index(drop=True),
        use_container_width=True
    )

    # ==== Dự đoán 2025 ====
    st.subheader("Dự đoán số học sinh năm 2025")
    predict_df = subset_2024.copy()
    predict_df['Year'] = 2025
    predict_df['Quarter'] = (predict_df['Month'] - 1) // 3 + 1


    input_cols = ['Center', 'Course Line', 'Subject', 'Class Type', 'Year', 'Month', 'Quarter', 'num_classes']
    predict_df['Center'] = center
    X_pred = predict_df[input_cols]
    y_pred = model.predict(X_pred)
    predict_df['Dự đoán 2025'] = y_pred.round(0).astype(int)
    predict_df['Course Line Name'] = predict_df['Course Line'].map(course_line_map)
    predict_df['Subject Name'] = predict_df['Subject'].map(course_code_map)
    st.dataframe(
        predict_df[[
            'Course Line Name', 'Subject Name', 'Class Type', 'num_classes', 'Dự đoán 2025'
        ]].sort_values('Dự đoán 2025', ascending=False)
        .rename(columns={
            'Course Line Name': 'Khoá học',
            'Subject Name': 'Môn học',
            'Class Type': 'Loại lớp',
            'num_classes': 'Số lớp',
            'Dự đoán 2025': 'Số học sinh'
        }).reset_index(drop=True),
        use_container_width=True
    )
    if month == 'Cả năm':
        st.subheader("📈 Biểu đồ dự đoán số học sinh năm 2025")
        month_chart_df = df[df['Center'] == center].copy()
        month_chart_df['Year'] = 2025
        month_chart_df['Quarter'] = (month_chart_df['Month'] - 1) // 3 + 1
        X_pred_all = month_chart_df[input_cols]
        month_chart_df['Dự đoán 2025'] = model.predict(X_pred_all).round(0).astype(int)
        chart_df = month_chart_df.groupby('Month')['Dự đoán 2025'].sum().reset_index()


        bar_chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X('Month:O', title='Tháng'),
            y=alt.Y('Dự đoán 2025:Q', title='Số học sinh')
        )

        text = bar_chart.mark_text(
            align='center',
            baseline='bottom',
            dy=-5
        ).encode(
            text='Dự đoán 2025:Q'
        )

        st.altair_chart(bar_chart + text, use_container_width=True)

