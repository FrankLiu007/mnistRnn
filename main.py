# -*- coding: utf-8 -*- 
# @Time : 2018/9/4 9:05 
# @Author : Allen 
# @Site :
import tensorflow as tf
import data_helper
from MnistRnn import MnistRnn
import os
import numpy as np
import configparser


class RNN:
    def __init__(self):
        cf = configparser.ConfigParser()
        cf.read('conf.ini', encoding="utf-8-sig")
        self.mnist = data_helper.get_data()
        self.num_class = int(cf.get('parm', 'num_class'))
        self.height = int(cf.get('parm', 'height'))
        self.dim_hidden = int(cf.get('parm', 'dim_hidden'))
        self.width = int(cf.get('parm', 'width'))
        self.learning_rate = float(cf.get('parm', 'learning_rate'))
        self.epochs = int(cf.get('parm', 'epochs'))
        self.batch_size = int(cf.get('parm', 'batch_size'))
        self.display_step = int(cf.get('parm', 'display_step'))
        self.model_path = os.path.join(os.getcwd(), cf.get('parm', 'model_path'))
        self.summaries_path = os.path.join(os.getcwd(), cf.get('parm', 'summaries_path'))

    def train(self):
        input_x = tf.placeholder(tf.float32, [None, self.height, self.width])
        out_y = tf.placeholder(tf.float32, [None, self.num_class])
        rnn = MnistRnn(input_x, self.height, self.width, self.dim_hidden, self.num_class,
                       self.learning_rate)
        # 最优化
        optimizer = rnn.optimizer_graph(out_y)
        # 偏差
        accuracy = rnn.accuracy_graph(out_y)
        # 日志
        summary_op = tf.summary.merge_all()

        # print(self.model_path)
        # print(self.summaries_path)

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.summaries_path):
            os.makedirs(self.summaries_path)
        saver = tf.train.Saver(max_to_keep=1)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(self.summaries_path, sess.graph)
            for epoch in range(self.epochs):
                avg_accuracy = 0
                total_batch = int(self.mnist.train.num_examples / self.batch_size)
                for i in range(total_batch):
                    batch_train_x, batch_train_y = self.mnist.train.next_batch(self.batch_size)
                    batch_train_x = batch_train_x.reshape([self.batch_size, self.height, self.width])
                    feed_dict = {
                        input_x: batch_train_x,
                        out_y: batch_train_y,
                    }
                    _, summary = sess.run([optimizer, summary_op], feed_dict=feed_dict)
                    summary_writer.add_summary(summary, epoch)
                    avg_accuracy += sess.run(accuracy, feed_dict=feed_dict) / total_batch
                if epoch % 1 == 0:
                    batch_test_x, batch_test_y = self.mnist.test.next_batch(self.batch_size)
                    batch_test_x = batch_test_x.reshape([self.batch_size, self.height, self.width])
                    test_acc = sess.run(accuracy, feed_dict={
                        input_x: batch_test_x,
                        out_y: batch_test_y,
                    })
                    saver.save(sess, self.model_path + 'model.ckpt')
                    print("epoch:{}/{},train_accuracy:{},test_accuracy:{}".format(epoch + 1, self.epochs,
                                                                                  avg_accuracy,
                                                                                  test_acc))

    def predict(self, batch_x_test):
        input_x = tf.placeholder(tf.float32, [None, self.height, self.width])
        rnn = MnistRnn(input_x, self.height, self.width, self.dim_hidden, self.num_class,
                       self.learning_rate)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, self.model_path + 'model.ckpt')
            predict = tf.argmax(rnn.pre, 1)
            vector_list = sess.run(predict, feed_dict={input_x: batch_x_test})
            vector_list = vector_list.tolist()
            return vector_list


if __name__ == '__main__':
    rnn = RNN()
    if os.path.exists(os.path.join(os.getcwd(), 'ckpt', 'checkpoints')):
        if not os.path.getsize(os.path.join(os.getcwd(), 'ckpt', 'checkpoints')) > 0:
            print("正在训练模型.....")
            rnn.train()
    else:
        print("正在训练模型.....")
        rnn.train()
    mnist = data_helper.get_data()
    batch_x_test = mnist.test.images
    batch_x_test = batch_x_test.reshape([-1, rnn.height, rnn.width])
    batch_y_test = mnist.test.labels
    batch_y_test = list(np.argmax(batch_y_test, 1))
    pre_y = rnn.predict(batch_x_test)
    for text in batch_y_test:
        print('Label:', text, ' Predict:', pre_y[batch_y_test.index(text)])
