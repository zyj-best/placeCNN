#coding=utf-8

import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import numpy as np 
import pdb
from datetime import datetime
from VGG16 import *
import cv2
import os
import matplotlib.pyplot as plt

def test(path):

    x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='input')
    keep_prob = tf.placeholder(tf.float32)
    output = VGG16(x, keep_prob, 3)
    score = tf.nn.softmax(output)
    f_cls = tf.argmax(score, 1)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, './model/model.ckpt-499')
    for i in os.listdir(path):
        imgpath = os.path.join(path, i)
        im = cv2.imread(imgpath)
        im = cv2.resize(im, (224 , 224))# * (1. / 255)
        im = np.expand_dims(im, axis=0)
        #pred = sess.run(f_cls, feed_dict={x:im, keep_prob:1.0})
        pred, _score = sess.run([f_cls, score], feed_dict={x:im, keep_prob:1.0})
        prob = round(np.max(_score), 4)
        #print "{} flowers class is: {}".format(i, pred)
        print("{}  class is: {}, score: {}".format(i, int(pred), prob))

        # plt.imshow(im)
        # # plt.imshow(im1)
        # plt.title(u'预测值:%i' % pred)
        # plt.show()
    sess.close()


if __name__ == '__main__':
    path = './test'
    test(path)


    