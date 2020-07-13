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

t1 = graph.max_pooling1d(x1, 2, 1)
t2 = graph.add(t1, x2)
t3 = graph.tanh(t2)
t4 = graph.add(t3, x3)
y = graph.mean(t4)

sess = graph.executor(*[y])
engine = graph.gradient(y, *[y, x1, x2, x3])



t_x1 = tf.placeholder(tf.float64)
t_x2 = tf.placeholder(tf.float64)
t_x3 = tf.placeholder(tf.float64)

#t_t1 = tf.nn.max_pool(t_x1, (1, 2, 2, 1), (1, 1, 1, 1), padding='VALID')#N, W, C
t_t1 = tf.nn.max_pool1d(t_x1, ksize=[1, 2, 1], strides=[1, 1, 1], padding='VALID')
t_t2 = tf.add(t_t1, t_x2)
t_t3 = tf.nn.tanh(t_t2)
t_t4 = tf.add(t_t3, t_x3)
t_y = tf.reduce_mean(t_t4)
t_engine = tf.gradients(t_y, [t_y, t_x1, t_x2, t_x3])


with tf.Session() as t_sess:
    epochs = 100
    x1_shape = (10, 28, 3)#(10, 27, 3)
    x2_shape = (27, 3)
    x3_shape = (27, 3)
    for i in range(epochs):
        np.random.seed(i)
        np_x1 = np.random.uniform(-1.5, 1.5, x1_shape)
        np_x2 = np.random.uniform(-1.2, 1.2, x2_shape)
        np_x3 = np.random.uniform(-1.6, 1.9, x3_shape)
        sess_y = sess.run(feed_dict={x1:np_x1, x2:np_x2, x3:np_x3})
        grad_y, grad_x1, grad_x2, grad_x3 = engine.get()
        #print(type(grad_y), type(grad_x1), type(grad_x2), type(grad_x3))

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
        #print(type(grad_x1))
        #print(type(t_grad_x1))
        #exit()
        t = grad_x1-t_grad_x1
        #print(grad_x1)
        #print(t_grad_x1)
        t = np.abs(t)
        r = np.max(t)
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
 