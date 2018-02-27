import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Embedding,Dropout,Dense,Merge

k=128
ratings = pd.read_csv("ratings.dat",sep='::',names=['user_id','movie_id','rating','timestamp'])
n_users = np.max(ratings['user_id'])  # 提取用户id数量
n_movies = np.max(ratings['movie_id'])  # 提取电影数量
print([n_users,n_movies,len(ratings)])

plt.hist(ratings['rating']) # 绘制评分柱状图
plt.show()
print(np.mean(ratings['rating']))