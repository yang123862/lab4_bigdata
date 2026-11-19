import numpy as np
import pandas as pd
import time

# 固定随机种子，保证和Spark版本初始中心点一致
np.random.seed(42)

# 读取本地iris.csv
data = pd.read_csv("iris.csv", header=None).values
k = 3
max_iter = 100
tol = 1e-5

# 随机选初始中心点
n_samples = data.shape[0]
centers = data[np.random.choice(n_samples, k, replace=False)]

start_time = time.time()
for i in range(max_iter):
    # 分配簇
    distances = np.sqrt(((data[:, np.newaxis] - centers)**2).sum(axis=2))
    labels = np.argmin(distances, axis=1)
    # 更新中心点
    new_centers = np.zeros_like(centers)
    for c in range(k):
        points = data[labels == c]
        if len(points) > 0:
            new_centers[c] = points.mean(axis=0)
    # 判断收敛
    if np.linalg.norm(new_centers - centers) < tol:
        break
    centers = new_centers
end_time = time.time()

# 输出结果
print("===== 单机K-Means结果 =====")
print(f"迭代次数: {i+1}")
print(f"运行时间: {end_time - start_time:.4f} s")
print("最终聚类中心：")
print(centers)
print("前10个样本的簇标签：")
print(labels[:10])

# 保存结果到文件
with open("standalone_result.txt", "w", encoding="utf-8") as f:
    f.write("单机K-Means结果\n")
    f.write(f"迭代次数: {i+1}\n")
    f.write(f"运行时间: {end_time - start_time:.4f} s\n")
    f.write("聚类中心:\n")
    f.write(str(centers))
    f.write("\n所有样本标签:\n")
    f.write(str(labels))
