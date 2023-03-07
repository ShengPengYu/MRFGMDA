import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif']='SimHei'#设置中文显示
plt.figure(figsize=(6,6),frameon=False)#将画布设定为正方形，则绘制的饼图是正圆



#定义饼图的标签，标签是列表
values=[363,360,333,276,216,207,165,160,153,147,117,117,114,100,93,90]
labels=['overexpressed',
        'upregulated',
        'downregulated',
        'downregulation',
        'copy number gain',
        'down-regulated',
        'dysregulation',
        'deregulated',
        'dysregulated',
        'overexpression',
        'copy number loss',
        'up-regulated',
        'upregulation',
        'down-regulated',
        'differentially expressed',
        'recurence related']

# values
# labels参数设置每一块的标签；
# labeldistance参数设置标签距离圆心的距离（比例值,只能设置一个浮点小数）
# autopct参数设置比例值的显示格式(%1.1f%%)
# pctdistance参数设置比例值文字距离圆心的距离
# startangle参数设置饼图起始角度

plt.pie(values,labels=labels,
        labeldistance=1,
        autopct='%.2f%%',
        pctdistance=0.85,
        startangle=90)#绘制饼图

# explode参数设置每一块顶点距圆形的长度（比例值,列表）；
# colors参数设置每一块的颜色（列表）；


#plt.title('title')#绘制标题

#  loc 设置摆放的位置
# plt.legend(loc=(1.04,1.5)) loc指的是legend的左下角的那个顶点的坐标
# plt.legend(loc=(1.04,1),fontsize=7,)
# 手动的设置我们的legend box 放置的地方，那就是使用bbox_to_anchor这个参数
#plt.legend(loc="upper left",fontsize=7,bbox_to_anchor=(1.04,1.5))


plt.gca().xaxis.set_major_locator(plt.NullLocator())
plt.gca().yaxis.set_major_locator(plt.NullLocator())
#plt.show()
#name是保存路径

plt.savefig('./pie1.pdf',bbox_inches="tight",pad_inches=0.0)#保存图片
#plt.show()