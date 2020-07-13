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
t1 = graph.matmul(x1, x2)#(5,4)
t2 = graph.matmul(x1, x3)#(5,4)

t9 = graph.add(graph.transpose(t1), graph.transpose(t2))
t9 = graph.expand_dims(t9, -1)
y = graph.mean(t9)
sess = graph.executor(*[y])
engine = graph.gradient(y, *[y, x1, x2, x3])

t_x1 = tf.placeholder(tf.float64)
t_x2 = tf.placeholder(tf.float64)
t_x3 = tf.placeholder(tf.float64)
t_t1 = tf.matmul(t_x1, t_x2)
t_t2 = tf.matmul(t_x1, t_x3)
t_t9 = tf.transpose(t_t1) + tf.transpose(t_t2)
t_t9 = tf.expand_dims(t_t9, -1)
t_y = tf.reduce_mean(t_t9)
t_engine = tf.gradients(t_y, [t_y, t_x1, t_x2, t_x3])


with tf.Session() as t_sess:
    epochs = 100
    x1_shape = (5, 3)
    x2_shape = (3, 4)
    x3_shape = (3, 4)
    for i in range(epochs):
        np.random.seed(i)
        np_x1 = np.random.uniform(1.0, 1.5, x1_shape)
        np_x2 = np.random.uniform(1.2, 2.1, x2_shape)
        np_x3 = np.random.uniform(1.8, 2.2, x3_shape)
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
