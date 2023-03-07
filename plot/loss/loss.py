import numpy as np
import matplotlib.pyplot as plt
import os


print("当前路径 -> %s" %os.getcwd())


# 创建模拟数据
t = np.arange(1, 202, 1)

test = np.loadtxt(os.getcwd()+'/loss/data/test.txt')
loss = np.loadtxt(os.getcwd()+'/loss/data/loss.txt')
data1 = np.exp(t)
data2 = np.sin(2 * np.pi * t)

fig, test_ax = plt.subplots()

color = 'tab:blue'
test_ax.set_xlabel('Iterations')
test_ax.set_ylabel('Test accuracy', color=color)
test_ax.plot(t, test, color=color)
test_ax.tick_params(axis='y', labelcolor=color)

test_max = max(test)
print(test_max)
for i in t:
    test_ax.plot(i,test_max, color=color)


#test_max_ax = test_ax.twinx()  # 创建共用x轴的第二个y轴
#test_max_ax.plot((test_max,0),(test_max,201), color=color)

loss_min = min(loss)
print(loss_min)

loss_ax = test_ax.twinx()  # 创建共用x轴的第二个y轴
loss_ax.set_ylim(0,1)
color = 'tab:red'
loss_ax.set_ylabel('Training loss', color=color)
loss_ax.plot(t, loss, color=color)
loss_ax.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
fig.set_size_inches(6, 6)
#plt.show()
plt.savefig("loss.pdf")
