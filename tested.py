import pandas as pd

df = pd.read_excel('k12_class_data_2024.xlsx')

# n=2 => ta chỉ lấy Center, CourseLine, còn phần còn lại (SubjectCode–ClassType) lưu ở Col3
tmp = df['Class name'].str.split('-', n=2, expand=True)
df['Center']      = tmp[0]
df['Course Line'] = tmp[1]

# tách thêm ClassType từ phần còn lại
df['Class Type'] = tmp[2].str.split('-', n=1).str[-1]

# tiếp phần mapping…

# 3. Định nghĩa mapping cho Center (theo Bảng 2)
center_map = {
    '29T1HDT': 'Hoàng Đạo Thuý (HN)',
    '25LB':    'Long Biên (HN)',
    '10TP':    'Trần Phú (HN)',
    '505MK':   'Minh Khai (HN)',
    '71NCT':   'Nguyễn Chí Thanh (HN)',
    '06NHT': 'Nguyễn Hữu Thọ (HN)',
    '107NPS': 'Nguyễn Phong Sắc (HN)',
    '98NVC': 'Nguyễn Văn Cừ (HN)',
    'KĐT VP': 'KĐT Văn Phú (HN)',
    'VHHN': 'Vinhomes Hàm Nghi (HN)',
    '22CTC': 'Thành Công (HN)',
    '176DC': 'Định Công (HN)',
    '672A28PVT': 'Phan Văn Trị (HCM)',
    '01TC': 'Trường Chinh (HCM)',
    '02SH': 'Song Hành (HCM)',
    '223NX': 'Nguyễn Xí (HCM)',
    '01QT': 'Quang Trung (HCM)',
    '261-263PXL': 'Phan Xích Long (HCM)',
    '165-167NTT': 'Nguyễn Thị Thập (HCM)',
    '01TK': 'Tô Ký (HCM)',
    '6183/2': '3 tháng 2 (HCM)',
    '490 PTB': 'Phạm Thái Bường (HCM)',
    '120-122PVĐ': 'Phạm Văn Đồng (HCM)',
    '174TL': 'Tên Lửa (HCM)',
    '322TT': 'Tây Thạnh (HCM)',
    '39HTLO': 'Hải Thượng Lãn Ông (HCM)',
    '343PNL': 'Phạm Ngũ Lão (HCM)',
    '99LVV': 'Lê Văn Việt (HCM)',
    '414LBB': 'Luỹ Bán Bích (HCM)',
    '624LLQ': 'Lạc Long Quân (HCM)',
    '22-24UVK': 'Ung Văn Khiêm (HCM)',
    '205ALHP': 'Vũng Tàu',
    '01TP': 'Vĩnh Phúc',
    'DLLL': 'Thanh Hóa',
    '04HVT': 'Thái Nguyễn',
    '70NVC': 'Quảng Ninh', 
    '1606AHV': 'Phú Thọ',
    '67ĐLLN': 'Nghệ An',
    '268TNH': 'Hải Phòng',
    '253PVT': 'Đồng Nai',
    '255-257HV': 'Đà Nẵng',
    '153QTHD': 'Cần Thơ',
    '76NAN': 'Dĩ An Bình Dương (BD)',
    '230ĐLBD': '230 Đại Lộ Bình Dương (BD)',
    '299LTT': 'Bắc Ninh',
    'CTPB': 'Các tỉnh phía Bắc',
    'CTPN': 'Các tỉnh phía Nam',
    'HCM-Online': 'HCM Online',
    'HN-ONLINE':  'HN Online',
    'DArt':       'Digital Art Online'
}
course_line_map = {
    'AD':   'Art & Design',
    'ROB':  'Robotics',
    'C4K':  'Code for Kids',
    'C4T':  'Code for Teens',
}

df['Center Name']      = df['Center'].map(center_map).fillna(df['Center'])
df['Course Line Name'] = df['Course Line'].map(course_line_map).fillna(df['Course Line'])

# 5. Chuyển cột số về numeric
df['Student count'] = pd.to_numeric(df['Student count'], errors='coerce').fillna(0).astype(int)
df['Open plan']     = pd.to_numeric(df['Open plan'], errors='coerce').fillna(0).astype(int)

# 6. Tính tỷ lệ hoàn thành plan cho mỗi (Center, Course Line)
group_plan = (
    df
    .groupby(['Center Name', 'Course Line Name'])
    .agg(
        total_students=('Student count', 'sum'),
        total_plan    =('Open plan',      'sum')
    )
    .reset_index()
)
group_plan['completion_rate'] = (
    group_plan['total_students'] / group_plan['total_plan']
).fillna(0)
group_plan = group_plan.sort_values('completion_rate', ascending=False)

print("=== Tỷ lệ hoàn thành kế hoạch ===")
print(group_plan.to_string(index=False))

# 7. Phân tích theo hình thức lớp
group_type = (
    df
    .groupby('Class Type')
    .agg(total_students=('Student count', 'sum'))
    .reset_index()
)
group_type['pct'] = 100 * group_type['total_students'] / group_type['total_students'].sum()
group_type = group_type.sort_values('total_students', ascending=False)

print("\n=== Số học sinh theo hình thức lớp ===")
print(group_type.to_string(index=False))

# 8. Trung tâm dẫn đầu theo từng khóa học
best_centers = (
    df
    .groupby(['Course Line Name', 'Center Name'])
    .agg(total_students=('Student count', 'sum'))
    .reset_index()
    .sort_values(['Course Line Name','total_students'], ascending=[True, False])
    .drop_duplicates('Course Line Name')
    .reset_index(drop=True)
)

print("\n=== Trung tâm dẫn đầu theo từng lộ trình ===")
print(best_centers.to_string(index=False))
best_centers.to_excel('best_centers.xlsx', index=False)
