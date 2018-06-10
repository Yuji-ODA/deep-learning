import numpy as np
import tensorflow as tf

'''
データの生成
'''
# ORゲート
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [1]])

'''
モデル設定
'''
tf.set_random_seed(0)

w = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))

x = tf.placeholder(tf.float32, shape=[None, 2])
t = tf.placeholder(tf.float32, shape=[None, 1])
y = tf.nn.sigmoid(tf.matmul(x, w) + b) #活性化関数

cross_entropy = - tf.reduce_sum(t * tf.log(y) + (1 - t) * tf.log(1 - y)) #コスト関数
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy) #勾配降下法の定式化

correct_prediction = tf.equal(tf.to_float(tf.greater(y, 0.5)), t) #予測結果の確認（正解数）

'''
モデル学習
'''
# 初期化
init = tf.global_variables_initializer() #変数の一括初期化
sess = tf.Session() #セッション作成
sess.run(init)

# 学習
for epoch in range(200):
    sess.run(train_step, feed_dict={
        x: X,
        t: Y
    })

'''
学習結果の確認
'''
classified = correct_prediction.eval(session=sess, feed_dict={
    x: X,
    t: Y
})
prob = y.eval(session=sess, feed_dict={
    x: X
})

print('classified:')
print(classified)
print()
print('output probability:')
print(prob)
