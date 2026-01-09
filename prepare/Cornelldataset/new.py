# import numpy as np
#
# def softmax(x):
#     x_exp = np.exp(x)
#     # 如果是列向量，则axis=0
#     x_sum = np.sum(x_exp, axis=0, keepdims=True)
#     s = x_exp / x_sum
#     return s
#
#
# data = np.array([2, 0.8])
# data1 = np.array([4, 1.3])
# soft_v = softmax(data)
# soft_v1 = softmax(data1)
# # index_v = np.argmax(soft_v, axis=0)
#
# # print(index_v, round(soft_v[index_v], 3))
# print(soft_v)
# print(soft_v1)

# import numpy as np
# data = np.array([1, 2,3,4,5,6,7,8,9])
# data = data[2:5]
# print(data)

# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
#
# # 创建一些随机数据
# x = range(1, 21)
# y = [i**2 for i in x]
#
# # 创建主图
# fig, ax = plt.subplots()
#
# # 绘制主图
# ax.plot(x, y, linewidth=2, color='blue')
#
# # 创建内嵌图
# axins = zoomed_inset_axes(ax, 2, loc=2)
#
# # 在内嵌图中绘制局部放大图
# axins.plot(x, y, linewidth=2, color='red')
# axins.set_xlim(5, 10)
# axins.set_ylim(20, 100)
#
# # 将内嵌图放到主图上
# axins.get_xaxis().set_visible(True)
# axins.get_yaxis().set_visible(True)
# mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")
#
# # 美化主图
# ax.set_xlabel('X轴', fontsize=14)
# ax.set_ylabel('Y轴', fontsize=14)
# ax.tick_params(axis='both', which='major', labelsize=12)
#
# # 美化内嵌图
# axins.tick_params(axis='both', which='major', labelsize=10)
# axins.spines['bottom'].set_linewidth(0.5)
# axins.spines['top'].set_linewidth(0.5)
# axins.spines['left'].set_linewidth(0.5)
# axins.spines['right'].set_linewidth(0.5)
# axins.xaxis.set_tick_params(width=0.5)
# axins.yaxis.set_tick_params(width=0.5)
#
# # 显示图形
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10, 0.1)
y = np.sin(x)

fig, ax = plt.subplots()
ax.plot(x, y)

plt.subplots_adjust(left=0.01)
plt.show()




# import matplotlib.pyplot as plt
#
# # 创建一些随机数据
# x = range(1, 21)
# y = [i**2 for i in x]
#
# # 创建图形并绘制主图
# fig, ax = plt.subplots()
# ax.plot(x, y, linewidth=2, color='blue')
#
# # 选定要放大的范围
# x1, x2 = 5, 10
# y1, y2 = 20, 100
#
# # 绘制圆圈
# radius = 0.4
# circle = plt.Circle((x1, y1), radius, color='red', fill=False)
# ax.add_patch(circle)
#
# # 绘制箭头
# arrow_x, arrow_y = x1 + 2 * radius, y1 + 2 * radius
# arrow_dx, arrow_dy = x1 - arrow_x + radius, y1 - arrow_y + radius
# arrow = ax.annotate('',
#                     xy=(arrow_x, arrow_y),
#                     xytext=(arrow_x + arrow_dx, arrow_y + arrow_dy),
#                     arrowprops=dict(facecolor='black', width=1, headwidth=6, headlength=6))
#
# # 绘制放大区域
# axins = ax.inset_axes([0.4, 0.4, 0.5, 0.5])
# axins.plot(x, y, linewidth=2, color='red')
# axins.set_xlim(x1, x2)
# axins.set_ylim(y1, y2)
# axins.spines['right'].set_visible(False)
# axins.spines['top'].set_visible(False)
#
# # 美化主图
# ax.set_xlabel('X轴', fontsize=14)
# ax.set_ylabel('Y轴', fontsize=14)
# ax.tick_params(axis='both', which='major', labelsize=12)
#
# # 显示图形
# plt.show()

