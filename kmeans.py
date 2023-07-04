
import random
import numpy as np

# 距离度量 -- 欧氏距离
def get_distance(x1, x2):
    return np.sqrt(np.sum(np.square(x1-x2)))

# 选择k个随机的聚类中心
def center_init(k, X):
    n_samples, n_features = X.shape
    centers = np.zeros((k, n_features))
    selected_centers_index = []
    for i in range(k):
        # 每一次循环随机选择一个类别中心,判断不让centers重复
        sel_index = random.choice(list(set(range(n_samples))-set(selected_centers_index)))
        centers[i] = X[sel_index]
        selected_centers_index.append(sel_index)
    return centers





