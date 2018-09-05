# -*- coding: utf-8 -*- 
# @Time : 2018/9/4 9:05 
# @Author : Allen 
# @Site :
from tensorflow.examples.tutorials.mnist import input_data


def get_data():
    mnist = input_data.read_data_sets('./data', one_hot=True)
    return mnist


if __name__ == '__main__':
    import configparser

    cf = configparser.ConfigParser()
    cf.read('conf.ini', encoding="utf-8-sig")
    print(float(cf.get('parm', 'learning_rate')))
