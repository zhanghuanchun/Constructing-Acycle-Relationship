import pandas as pd

# -------------------------------------------------------------------------#
#                         下面是获取Q矩阵的代码                                #
# -------------------------------------------------------------------------#

# 读取csv文件
item = pd.read_csv("E:/毕业论文/ICD-main/ICD-main/data/HSMath/item.csv")

# 反序列化 knowledge_code 字段
item['knowledge_code'] = item['knowledge_code'].apply(lambda x: eval(x))

# 将数据转换为矩阵形式
matrix = pd.pivot_table(item.explode('knowledge_code'),
                        index='item_id',
                        columns='knowledge_code',
                        aggfunc=len,
                        fill_value=0)

# 显示结果
print(matrix)

# 将矩阵保存为 CSV 文件
matrix.to_csv('E:/毕业论文/ICD-main/ICD-main/Consistency check/Q_matrix.csv', index=True, header=True)

# -------------------------------------------------------------------------#
#                         下面是获取异常数据的代码                              #
# -------------------------------------------------------------------------#

# 读入题目数据
item_ids = pd.read_csv('E:/毕业论文/ICD-main/ICD-main/data/HSMath/item.csv')['item_id'].unique()
# print(item_ids)

# 读入学生做题记录
records = pd.read_csv('E:/毕业论文/ICD-main/ICD-main/data/HSMath/record.csv', header=None, names=['user_id', 'item_id', 'score'], skiprows=1)
# print(records)

# 找出不存在于题目数据中的题目
invalid_records = records[~records['item_id'].isin(item_ids)]
print(invalid_records)
print("不存在于题目数据中的题目数为：",len(invalid_records))

# 打印异常信息所对应的内容
# 先获取总共的行数
num_rows, num_cols = invalid_records.shape

# 获取出问题的行号
row_indices = invalid_records.index.tolist()
for i in range(num_rows):
    row = invalid_records.iloc[i]
    # 打印每个信息，这里的行数，由于数组是0开始下标,并且我去掉了第一行的标题栏，但是表格下标是从1开始，所以加一个2
    print('第{: ^4}行中，\t'.format(row_indices[i] + 2), end='')
    print('学生号：{: ^2}\t'.format(row["user_id"]), end='')
    print(f'题目号：{row["item_id"]}\t', end='')
    print(f'得分：{row["score"]}\t')
