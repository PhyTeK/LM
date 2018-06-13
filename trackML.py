import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_data():
     data = pd.read_csv('./train_100_events/event000001000-hits_1000.csv',sep=',')
     return data

data = read_data()

print(data.shape,data.head())

a = np.array([[-0.3,0.2,0.04],
             [0.12,-0.11,-0.7],
             [0.03,-0.2,0.33]])

b = np.array([12.3,-1.23,23.23]).transpose()
bt = np.array([[12.3],[-1.23],[23.23]])
t = np.array([0.2,0.3,0.5])

print(bt)
print(b)
print(b.shape, bt.shape)
print(a.dot(b) + t)
print(a@b +t )

x = [1,2,1.5]
y = [3,4,3.5]
z = [5,6,5.5]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

c = ['r','b','g','y']

n_mod = 76
id_mod = data['module_id']

ax.scatter(data['x'],data['y'],data['z'],marker='.',c='r')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


plt.show()

def dfScatter(df, xcol='x', ycol='y', zcol='z', catcol='module_id'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    categories = np.unique(df[catcol])
    colors = np.linspace(0, 1, len(categories))
    colordict = dict(zip(categories, colors))  
    df["Color"] = df[catcol].apply(lambda x: colordict[x])
    ax.scatter(df[xcol], df[ycol], df[zcol], marker='.', c=df.Color)
    return plt


fig = dfScatter(data)
fig.savefig('fig1.png')

plt.show()


print("Done!")


