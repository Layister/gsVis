import pandas as pd

# 读取 truth.txt 文件
file_path = '/Users/wuyang/Documents/MyPaper/3/gsVis/data/BRCA/truth.txt'
df = pd.read_csv(file_path, sep='\t', header=None)
# 假设注释信息在第二列（索引为 1）
annotation_column = df[1]
# 统计不同注释类型的数量
unique_annotation_count = annotation_column.nunique()
# 获取所有不同的注释类型
unique_annotations = annotation_column.unique()
print(f"truth.txt 文件中不同注释类型的数量为: {unique_annotation_count}")
print("不同的注释类型分别为:")
for annotation in unique_annotations:
    print(annotation)
