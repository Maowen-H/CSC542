import tensorflow as tf

# 如果你想用 TF1.x 风格，就加上：
tf.compat.v1.disable_eager_execution()

# 显示设备放置信息（可选）
tf.debugging.set_log_device_placement(True)

# 建立一个较大的矩阵乘法图，并循环多次
with tf.device("/GPU:0"):
    a = tf.random.normal([10000, 10000])
    b = tf.random.normal([10000, 10000])
    c = tf.matmul(a, b)

session = tf.compat.v1.Session()  # TF1.x 会话
for i in range(50):
    print(f"Iteration {i+1} ...")
    _ = session.run(c)

print("All done!")
