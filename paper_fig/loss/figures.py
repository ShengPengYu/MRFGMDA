import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
data1_loss =np.loadtxt("1.txt")
data2_loss = np.loadtxt("1.txt")

x = data1_loss[:,0]
y = data1_loss[:,1]
fig = plt.figure(figsize = (7,5))       #figsize是图片的大小`

pl.plot(x,y,'r-',label = u'FGGCNMDA')

pl.legend()
#显示图例
#p3 = pl.plot(x2,y2, 'b-', label = u'SCRCA_Net')
pl.legend()
pl.xlabel(u'iters')
pl.ylabel(u'loss')
plt.title('Loss Curver of FGGCNMDA')


#axins.plot(x2,y2 , color='blue', ls='-')

plt.savefig("img_loss.pdf")
pl.show
#pl.show()也可以
