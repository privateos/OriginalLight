import sys
sys.path.append('D:/repo')

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import light as F
from light import backend as np
import light.functions as graph

x1 = graph.placeholder()

t1 = graph.mean_pooling1d(x1, 3, 1)
t1 = graph.sum(t1, axis=0)
y = graph.mean(t1)

sess = graph.executor(*[y])
engine = graph.gradient(y, *[y, x1])



t_x1 = tf.placeholder(tf.float64)

#N, W,C
t_t1 = tf.nn.avg_pool1d(t_x1, ksize=[1, 3, 1], strides=[1, 1, 1], padding='VALID')
t_t1 = tf.reduce_sum(t_t1, 0)
t_y = tf.reduce_mean(t_t1)
t_engine = tf.gradients(t_y, [t_y, t_x1])


with tf.Session() as t_sess:
    epochs = 100
    x1_shape = (10, 8, 3)#(10, 3, 5, 9, 3)
    x2_shape = (10, 3, 5, 9, 3)
    x3_shape = (10, 3, 5, 9, 3)
    for i in range(epochs):
        np.random.seed(i)
        np_x1 = np.random.uniform(-3.5, 3.5, x1_shape)
        #np_x1 = np_x1.astype(np.float32)
        sess_y, = sess.run(feed_dict={x1:np_x1})
        grad_y, grad_x1 = engine.get()

        tf_y = t_sess.run(t_y, feed_dict={t_x1:np_x1})
        t_grad_y, t_grad_x1 = t_sess.run(t_engine, feed_dict={t_x1:np_x1})

        flag = True
        print('-------------------')
        print(tf_y)
        r = np.max(np.abs(tf_y - sess_y))
        #a = np.asarray(tf_y)
        #print(a.shape)
        #print(sess_y.shape)
        #print(a.dtype)
        #print(sess_y.dtype)
        boundary = 1e-13
        if r > boundary:
            flag = False
        print(r)
        r = np.max(np.abs(grad_y-t_grad_y))
        if r > boundary:
            flag = False
        print(r)
        t = grad_x1-t_grad_x1
        t_v = np.max(np.abs(grad_x1))
        t = np.abs(t)
        r = np.max(t)
        if r > boundary:
            flag = False
        print(r)
        print("t_v=" + str(t_v))
        if flag == False:
            print("warning!!!")
            input()
