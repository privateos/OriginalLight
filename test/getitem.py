import sys
sys.path.append('D:/repo')

import light as F
from light import backend as np
import light.functions as graph
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

x = graph.placeholder()#(2,3,4)
key = (slice(0, 1, None), slice(1, 2, None))
get = graph.getitem(x, key)
y = graph.sum(get, axis=0)
y = graph.mean(y, axis=-1)
y = graph.sum(y, axis=0)
sess = graph.executor(*[y])
grad = graph.gradient(y, *[y, get, x])

t_x = tf.placeholder(tf.float64)
t_get = t_x[key]
t_y = tf.reduce_sum(t_get, axis=0)
t_y = tf.reduce_mean(t_y, axis=-1)
t_y = tf.reduce_sum(t_y, axis=0)
t_grad = tf.gradients(t_y, [t_y, t_get, t_x])




with tf.Session() as t_sess:
    epochs = 100
    for i in range(epochs):
        x_v = np.random.random((2,3,4))
        f_y, = sess.run(feed_dict={x:x_v})
        grad_y, grad_get, grad_x = grad.get()

        t_y_ = t_sess.run(t_y, feed_dict={t_x:x_v})
        t_grad_y, t_grad_get, t_grad_x = t_sess.run(t_grad, feed_dict={t_x:x_v})

        boundary = 1e-13
        flag = True
        print("--------------------------------")
        #print(f_y.shape, t_y_.shape)
        print(f_y)
        print(t_y_)
        r = np.max(np.abs(f_y - t_y_))
        print(r)
        if r > boundary:
            flag = False
        
        r = np.max(np.abs(grad_y - t_grad_y))
        print(r)
        if r > boundary:
            flag = False

        r = np.max(np.abs(grad_get - t_grad_get))
        print(r)
        if r > boundary:
            flag = False
        
        r = np.max(np.abs(grad_x - t_grad_x))
        print(r)
        if r > boundary:
            flag = False
        
        if flag == False:
            print("warning!!!")
            input()


