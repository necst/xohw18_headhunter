import numpy as np

data = np.load("fc_full/W.npy")
datab = np.load("fc_full/b.npy")

print('Weights: {shape}'.format(shape=data.shape))
print('Bias: {shape}'.format(shape=datab.shape))

print(data[0][0]);

print(datab[0])


#data.tofile('W.csv',sep=',\n',format='%10.5f')
#datab.tofile('b.csv',sep=',\n',format='%10.5f')
