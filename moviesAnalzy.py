'''
矩阵分解方法
用户矩阵ratings[user_id],电影矩阵ratings[movie_id]
用户电影矩阵ratrings[rating]
拟合一个用户x电影矩阵尽量与ratings[rating]相似
'''
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Embedding,Dropout,Dense,Merge,Reshape

k=128
ratings = pd.read_csv("ratings.dat",sep='::',names=['user_id','movie_id','rating','timestamp'])
n_users = np.max(ratings['user_id'])  # 提取用户id数量
n_movies = np.max(ratings['movie_id'])  # 提取电影数量
#print([n_users,n_movies,len(ratings)])

#plt.hist(ratings['rating']) # 绘制评分柱状图
#plt.show()
#print(np.mean(ratings['rating']))

model1=Sequential() # 创建第一个神经网络模型
model1.add(Embedding(n_users+1,k,input_length=1))  # 128维的用户Embedding层
model1.add(Reshape((k,)))

model2=Sequential() # 创建第二个神经网络模型
model2.add(Embedding(n_movies+1,k,input_length=1))  # 128维的电影Embedding层
model2.add(Reshape((k,)))

# 第一、二个网络的基础上叠加乘积计算
model=Sequential()
model.add(Merge([model1,model2],mode='dot',dot_axes=1))

# 输出层和最后的评分数进行对比，后向传播更新网络参数
model.compile(loss='mse',optimizer='adam')

users = ratings['user_id'].values
movies = ratings['movie_id'].values
X_train = [users,movies]  # 结合用户和电影数据作为训练数据
y_train = ratings['rating'].values  # 提取评分作为训练标签

model.fit(X_train, y_train, batch_size= 100,epochs=50)  # 训练
i = 10
j = 99
pred = model.predict([np.array([users[i]]),np.array([movies[j]])]) # 预测第10个用户对第99部电影的评分
sum=0
for i in range(ratings.shape[0]):
    sum+=(ratings['rating'][i]-model.predict([np.array([ratings['user_id'][i]]),np.array([ratings['movie_id'][i]])]))**2
mse=math.sqrt(sum/ratings.shape[0])
print(mse)