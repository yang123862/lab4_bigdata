from pyspark.sql import SparkSession
import numpy as np
import time

# 固定随机种子，和单机保持一致
np.random.seed(42)

# 构建spark会话
spark = SparkSession.builder \
    .appName("KMeansSpark") \
    .master("local[*]") \
    .getOrCreate()

# 读取HDFS上的iris.csv
rdd = spark.sparkContext.textFile("hdfs://localhost:9000/input/iris.csv")
# 转成浮点数组
data_rdd = rdd.map(lambda line: np.array([float(x) for x in line.split(',')]))
data = data_rdd.collect()
data = np.array(data)

k = 3
max_iter = 100
tol = 1e-5

# 和单机完全一样的初始中心点
n_samples = data.shape[0]
centers = data[np.random.choice(n_samples, k, replace=False)]

start_time = time.time()
for i in range(max_iter):
    centers_broad = spark.sparkContext.broadcast(centers)
    # 分布式计算：每个样本算距离、分配簇
    assign_rdd = data_rdd.map(lambda vec: (
        np.argmin(np.sqrt(((vec - centers_broad.value)**2).sum(axis=1))),
        (vec, 1)
    ))
    # 按簇求和、计数
    sum_cnt_rdd = assign_rdd.reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1]))
    new_centers = np.zeros_like(centers)
    for c, (vec_sum, cnt) in sum_cnt_rdd.collect():
        new_centers[c] = vec_sum / cnt
    # 收敛判断
    if np.linalg.norm(new_centers - centers) < tol:
        break
    centers = new_centers
end_time = time.time()

# 拿到全部标签，用于对比单机
labels = assign_rdd.sortByKey().map(lambda x:x[0]).collect()

print("===== Spark K-Means结果 =====")
print(f"迭代次数: {i+1}")
print(f"运行时间: {end_time - start_time:.4f} s")
print("最终聚类中心：")
print(centers)
print("前10个样本的簇标签：")
print(labels[:10])

# 保存spark结果
with open("spark_result.txt", "w", encoding="utf-8") as f:
    f.write("Spark K-Means结果\n")
    f.write(f"迭代次数: {i+1}\n")
    f.write(f"运行时间: {end_time - start_time:.4f} s\n")
    f.write("聚类中心:\n")
    f.write(str(centers))
    f.write("\n所有样本标签:\n")
    f.write(str(labels))

spark.stop()
