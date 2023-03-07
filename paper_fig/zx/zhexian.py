# encoding=utf-8
import matplotlib.pyplot as plt
from pylab import *                                 #支持中文
#mpl.rcParams['font.sans-serif'] = ['SimHei']

linewidth = '1' #//定义线条宽度
names = ['1', '100', '200', '300', '400','500','600','700','800','900','1000']
x = range(len(names))
y = [0.70798, 0.152705403268, 0.0751528979093, 0.0727966073155, 0.071515327096,0.0707918071747,0.0702920282632,0.0700353652984,0.0697990836203, 0.0697271420807,0.069601803869]
#plt.plot(x, y, 'ro-')
#plt.plot(x, y1, 'bo-')
plt.xlim(-0.1, len(names))  # 限定横轴的范围
plt.ylim(0, 0.85)  # 限定纵轴的范围
plt.plot(x, y, marker='o', mec='r', mfc='w',label='loss')
#plt.plot(x, y1, marker='*', ms=10,label=u'y=x^3曲线图')
plt.legend()  # 让图例生效
plt.xticks(x, names)
plt.margins(0)
plt.subplots_adjust(bottom=0)
plt.xlabel("Number of iterations") #X轴标签
plt.ylabel("loss") #Y轴标签
#plt.title("A simple plot") #标题
#plt.grid(False,linestyle='-.',linewidth=linewidth,color="grey")

for i in range(len(names)): 
	plt.text(i,y[i],y[i],rotation=90,verticalalignment="bottom",horizontalalignment="left")


plt.savefig('./zhexian.pdf',bbox_inches="tight",pad_inches=0.0)#保存图片
#plt.show()