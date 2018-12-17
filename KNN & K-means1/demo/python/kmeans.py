# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 17:03:49 2018

@author: Administrator
"""

import numpy as np
import math
import matplotlib.pyplot as plt
#%matplotlib inline


def load_dataset(file_name):
    data_mat = []
    with open(file_name) as fr:
        lines = fr.readlines()
    for line in lines:
        cur_line = line.strip().split("\t")
        flt_line = list(map(lambda x:float(x), cur_line))
        data_mat.append(flt_line)
    return np.array(data_mat)


data_set = load_dataset(r"C:\Users\Administrator\Desktop\KNN培训\Ch10-Kmeans\testSet.txt")
print(data_set)


point_x = data_set[:,0]
point_y = data_set[:,1]        
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(point_x, point_y, s=30, c="r", marker="o", label="sample point")
ax.legend()
ax.set_xlabel("factor1")
ax.set_ylabel("factor2")

def dist_eclud(vecA, vecB):
    vec_square = []
    for element in vecA - vecB:
        element = element ** 2
        vec_square.append(element)
    return sum(vec_square) ** 0.5

# 构建k个随机质心.首先可以找到数据集每一维的最小和最大值。然后得到每一维的取值范围。
#用0到1之间的随机数和取值范围相乘，再用最小值加上该乘积，就可以得到在每一维取值范围内的随机数。
def rand_cent(data_set, k):
    n = data_set.shape[1]    
    centroids = np.zeros((k, n))
    for j in range(n):
        min_j = float(min(data_set[:,j]))
        range_j = float(max(data_set[:,j])) - min_j
        centroids[:,j] = (min_j + range_j * np.random.rand(k, 1))[:,0]
    return centroids



def Kmeans(data_set, k):    
    m = data_set.shape[0]   
    cluster_assment = np.zeros((m, 2))  
    centroids = rand_cent(data_set, k)  
    cluster_changed = True        
    while cluster_changed:       
        cluster_changed = False   
        for i in range(m):        
            min_dist = np.inf; min_index = -1   
            for j in range(k):     
                dist_ji = dist_eclud(centroids[j,:], data_set[i,:])
                if dist_ji < min_dist:              
                    min_dist = dist_ji; min_index = j   
            if cluster_assment[i,0] != min_index:    
                cluster_changed = True      
            cluster_assment[i,:] = min_index, min_dist**2   
        for cent in range(k):   
            # 这是在求每次分别属于某个类别的数据，然后再以此为基础计算新的均值向量
            pts_inclust = data_set[np.nonzero(list(map(lambda x:x==cent, cluster_assment[:,0])))]
              # 按列求平均值，即新的均值向量
            centroids[cent,:] = np.mean(pts_inclust, axis=0)
    return centroids, cluster_assment          

# 首先初始化样本点的簇分配矩阵（cluster_assment），有80行2列，第一列为该样本点的簇分配索引，第二列为该样本点到该簇质心的欧氏距离。
# 当任意一个点的簇分配发生变化时，迭代执行以下操作：遍历每个样本点，计算样本点i到各个质心的距离，找到最小距离，
# 将该质心所在簇编号分配给该样本点。遍历完所有样本点后，重新计算每个簇的质心。直到所有样本点的簇分配都不再发生变化时迭代停止。
# 最后返回质心和样本点的簇分配矩阵

my_centroids, my_cluster_assment = Kmeans(data_set, 4)
print(my_centroids)
print(my_cluster_assment)

point_x = data_set[:,0]
point_y = data_set[:,1]  
cent_x = my_centroids[:,0]
cent_y = my_centroids[:,1]
fig, ax = plt.subplots(figsize=(10,5))
ax.scatter(point_x, point_y, s=30, c="r", marker="o", label="sample point")
ax.scatter(cent_x, cent_y, s=100, c="black", marker="v", label="centroids")
ax.legend()
ax.set_xlabel("factor1")
ax.set_ylabel("factor2")
