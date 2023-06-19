import pandas as pd
import matplotlib.pyplot as plt

# 读取 CSV 文件
data = pd.read_csv('fs_enn_mnist_results_fail.csv')

# # 绘制防御策略和准确率的关系图
# fig, ax = plt.subplots()
# for attack, group in data.groupby('Attack'):
#     ax.bar(group['Mode'], group['Accuracy'], label=attack)
# plt.title('Accuracy vs. Defense Mode')
# plt.xlabel('Defense Mode')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()

# # 按攻击方法和防御策略分组，计算平均准确率
# grouped_data = data.groupby(['Attack', 'Mode'])['Accuracy'].mean().reset_index()

# # 使用 pivot 函数将防御策略作为列，攻击方法作为行，生成新的 DataFrame
# pivot_data = pd.pivot_table(grouped_data, index='Attack', columns='Mode', values='Accuracy')

# # 显示表格
# print(pivot_data)

# # 显示每一类攻击下的准确率
# for attack in data['Attack'].unique():
#     attack_data = data[data['Attack'] == attack]
#     print('\nAttack:', attack)
#     for mode in data['Mode'].unique():
#         mode_data = attack_data[attack_data['Mode'] == mode]
#         mode_mean_acc = mode_data['Accuracy'].mean()
#         print(mode, 'mode accuracy:', mode_mean_acc)

# 按攻击方法和防御策略分组，计算平均准确率
grouped_data = data.groupby(['Attack'])['Accuracy'].mean().reset_index()

# 使用 pivot 函数将防御策略作为列，攻击方法作为行，生成新的 DataFrame
pivot_data = pd.pivot_table(grouped_data, index='Attack', values='Accuracy')

# 显示表格
print(pivot_data)