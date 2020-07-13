import sys
sys.path.append('D:/repo')

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import light as F
from light import backend as np
import light.functions as graph

x1 = graph.placeholder()
x2 = graph.placeholder()
x3 = graph.placeholder()

y = graph.sum(graph.arccosh(graph.add(graph.matmul(x1, x2), graph.matmul(x1, x3))))
sess = graph.executor(*[y])
engine = graph.gradient(y, *[y, x1, x2, x3])

t_x1 = tf.placeholder(tf.float64)
t_x2 = tf.placeholder(tf.float64)
t_x3 = tf.placeholder(tf.float64)
t = tf.matmul(t_x1, t_x2) + tf.matmul(t_x1, t_x3)

t_y = tf.reduce_sum(tf.log(t + tf.sqrt(tf.square(t) - 1.0)))
t_engine = tf.gradients(t_y, [t_y, t_x1, t_x2, t_x3])


with tf.Session() as t_sess:
    epochs = 100
    x1_shape = (5, 3)
    x2_shape = (3, 4)
    x3_shape = (3, 4)
    for i in range(epochs):
        np.random.seed(i)
        np_x1 = np.random.uniform(1.0, 2.0, x1_shape)
        np_x2 = np.random.uniform(1.5, 2.6, x2_shape)
        np_x3 = np.random.uniform(1.9, 3.2, x3_shape)
        sess_y = sess.run(feed_dict={x1:np_x1, x2:np_x2, x3:np_x3})
        grad_y, grad_x1, grad_x2, grad_x3 = engine.get()

        tf_y = t_sess.run(t_y, feed_dict={t_x1:np_x1, t_x2:np_x2, t_x3:np_x3})
        t_grad_y, t_grad_x1, t_grad_x2, t_grad_x3 = t_sess.run(t_engine, feed_dict={t_x1:np_x1, t_x2:np_x2, t_x3:np_x3})

        flag = True
        print('-------------------')
        print(tf_y)
        r = np.max(np.abs(tf_y - sess_y))
        boundary = 1e-13
        if r > boundary:
            flag = False
        print(r)
        r = np.max(np.abs(grad_y-t_grad_y))
        if r > boundary:
            flag = False
        print(r)
        r = np.max(np.abs(grad_x1-t_grad_x1))
        if r > boundary:
            flag = False
        print(r)
        r = np.max(np.abs(grad_x2-t_grad_x2))
        if r > boundary:
            flag = False
        print(r)
        r = np.max(np.abs(grad_x3-t_grad_x3))
        if r > boundary:
            flag = False
        print(r)
        if flag == False:
            print("warning!!!")
            input()
 