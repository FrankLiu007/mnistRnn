# -*- coding: utf-8 -*- 
# @Time : 2018/9/4 9:19 
# @Author : Allen 
# @Site :
import tensorflow as tf


class MnistRnn:
    def __init__(self, input_x, num_steps, dim_input, dim_hidden, num_class, learning_rate):
        self.input_x = input_x
        self.learning_rate = learning_rate
        with tf.name_scope('input'):
            input_x = tf.transpose(self.input_x, [1, 0, 2])
            input_x = tf.reshape(input_x, [-1, dim_input], name='input')
        with tf.name_scope('hidden'):
            hidden = tf.matmul(input_x, self.get_weight([dim_input, dim_hidden])) + self.get_bias([dim_hidden])
            hidden_split = tf.split(value=hidden, num_or_size_splits=num_steps, axis=0, name='hidden')
        with tf.variable_scope(name_or_scope='BasicLSTMCell'):
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(dim_hidden, forget_bias=1.0)
            lstm_o, lstm_s = tf.nn.static_rnn(lstm_cell, hidden_split, dtype=tf.float32)
        with tf.name_scope('output'):
            self.pre = tf.matmul(lstm_o[-1], self.get_weight([dim_hidden, num_class])) + self.get_bias([num_class])

    def get_weight(self, shape):
        return tf.Variable(tf.random_normal(shape))

    def get_bias(self, shape):
        return tf.Variable(tf.random_normal(shape))

    def optimizer_graph(self, y):
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=self.pre, labels=y), name='Loss')
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
        tf.summary.scalar('Loss', loss)
        return optimizer

    def accuracy_graph(self, y):
        with tf.name_scope('Accuracy'):
            acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.pre, 1), tf.argmax(y, 1)), tf.float32),
                                 name='Accuracy')
        tf.summary.scalar('Accuracy', acc)
        return acc
