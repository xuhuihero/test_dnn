# -*- coding: UTF-8 _*_
from __future__ import print_function
import datetime
from time import time
import os
import socket
import re
import sys
#import cbor
import numpy
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from tensorflow.contrib import layers
import random

HOST = socket.gethostname()
print(HOST)
os.system('rm -rf ./my_graph/' + HOST)

# Launch the graph
configure = tf.ConfigProto()
configure.gpu_options.allow_growth=True
#configure.gpu_options.per_process_gpu_memory_fraction = 0.45
configure.log_device_placement = False

buckets = [50, 100, 150, 200, 250, 300, 350, 400, 500, 9999999]  # 按link_num 分桶


method="train"
print("#########################")


file_dir = "./data/"

vacab_size = 10001
epochs = 3
embedding_size = 64
wide_learning_rate = 0.1
wide_learning_beta = 0.5
deep_learning_rate = 0.0003
l2 = 0.0001
dnn_layer=10
nHidden = 256
batch_size = 512
max_link_num = 5000
filter_num=256
clip_norm = 50000

print( "\nwide_learning_rate: " + str(
    wide_learning_rate) + "\twide_learning_beta: " + str(wide_learning_beta) + "\nembedding_size(DNN, FM): " + str(
    embedding_size) + "\nnLays: " + str(dnn_layer) + "\tnHidden: " + str(nHidden) + "\tdeep_learning_rate: " + str(
    deep_learning_rate) + "\tbatch_size: " + str(batch_size))
print("l2: " + str(l2))

class DataSetBucket(object):
    def __init__(self, file_name):
        self.instances = []
        self.batch_id = 0
        self.batchs = []
        self.total_batch_size = 0

        self.load_data(file_name)
            #print ("Load Finish " + str(datetime.datetime.now()))
        self.buckets = buckets
        self.num = len(self.instances)

    def load_data(self, file_name):
        for line in open(file_name):
            case_id, label, features = line.strip().split("\t")
            label=int(label)
            l_link = features.split(",")
            linkidfeat = []
            for link in l_link:
                linkidfeat.append(int(link))
            inst = (case_id, label, linkidfeat)
            self.instances.append(inst)

    def shuffle(self):
        begin = time()
        self.reset()

        def gen_key_list_pair(buckets):
            return [(i, []) for i in buckets]

        def drop_in(instance, buckets):
            for i in buckets:
                if len(instance[2]) < i[0]:
                    i[1].append(instance)
                    return

        # sort instances before drop in buckets
        self.instances = sorted(self.instances, key=lambda x: len(x[2]))
        #print("sorted time cost: ", time() - begin)

        # pre-drop
        key_list = gen_key_list_pair(self.buckets)

        # drop instances into buckets
        for i in self.instances:
            drop_in(i, key_list)
        #print("drop_in time cost:", time() - begin)
        self.instances = []

        self.batchs = []
        total_case_num = 0
        for index in range(0, len(key_list)):

            (x, arr) = key_list[index]
            itr = int(len(arr) / batch_size)
            last_batch_size = len(arr) % batch_size
            # print(str(itr) + " " + str(last_batch_size))
            begin_itr_id = 0
            end_itr_id = batch_size
            for idx in range(0, itr):
                tmp_arr = arr[begin_itr_id:end_itr_id]
                self.batchs.append(tmp_arr)
                begin_itr_id += batch_size
                end_itr_id += batch_size
                total_case_num += batch_size
            if last_batch_size != 0:
                tmp_arr = arr[begin_itr_id:end_itr_id]
                # self.batchs.append(tmp_arr)
                #print("remove last batch in bucket " + str(x) + " has case num: " + str(last_batch_size))

            key_list[index] = []
        self.total_batch_size = len(self.batchs)

        #print("cut time cost:", time() - begin)
        #print("total case num : ", total_case_num)
        numpy.random.shuffle(self.batchs)

    def next(self):
        if self.batch_id == self.total_batch_size:
            self.batch_id = 0

        batch_id = self.batch_id
        batch_oids = [[i[0]] for i in self.batchs[batch_id]]
        batch_labels_data = [[i[1]] for i in self.batchs[batch_id]]
        batch_linkid_data = [i[2] for i in self.batchs[batch_id]]
        batch_link_seq_len_data = [len(i[2]) for i in self.batchs[batch_id]]
        max_link_seq_len = max(batch_link_seq_len_data)

        for i in range(0, len(self.batchs[batch_id])):
            instlen = batch_link_seq_len_data[i]
            batch_linkid_data[i] = batch_linkid_data[i][0:instlen] + [0] * (max_link_seq_len - instlen)

        self.batch_id += 1
        return batch_oids, batch_labels_data, batch_linkid_data, batch_link_seq_len_data

    def has_next(self):
        return self.batch_id < self.total_batch_size

    def reset(self):
        self.batch_id = 0

    def clear(self):
        self.instances = []
        self.batch_id = 0
        self.batchs = []
        self.total_batch_size = 0


# tf Graph input
oid = tf.placeholder(tf.string, [batch_size, 1])
label = tf.placeholder(tf.float32, [batch_size, 1])
linkidFeat = tf.placeholder(tf.int32, [batch_size, None])
linkseqlens = tf.placeholder(tf.int32, [None])

# A placeholder for indicating each sequence length
lr_div = tf.placeholder("float")
train_phase = tf.placeholder(tf.bool)

loss_sum = tf.Variable(0.0, name="loss_sum")
count = tf.Variable(0.0, name="count")
reset_op = tf.group(loss_sum.assign(0.0), count.assign(0))
mean_loss = tf.div(loss_sum, count)

def DeepNetwork(train_phase):
    sm_link = tf.sequence_mask(linkseqlens, tf.reduce_max(linkseqlens), tf.float32)

    m_embeddings = tf.get_variable("m_embeddings", [vacab_size, embedding_size],
                                    initializer=tf.random_uniform_initializer(-0.05, 0.05))
    with tf.variable_scope("deep"):
        if embedding_size != 0 and dnn_layer != 0:
            deep_embeddings = m_embeddings
            sm_link_expand = tf.expand_dims(sm_link, -1)

            origin_link_embedding = tf.nn.embedding_lookup(deep_embeddings, linkidFeat) #
            link_embedding = tf.multiply(origin_link_embedding, sm_link_expand)
            dnninput = tf.layers.dense(link_embedding, embedding_size)
            dnninput = tf.reduce_sum(dnninput, axis=1)

            dnninput = tf.reshape(dnninput, [batch_size, embedding_size])

            layer_out=dnninput
            for idx in range(dnn_layer):
                layer_out = tf.layers.dense(layer_out, nHidden, activation=tf.nn.relu)

            local_prediction = tf.layers.dense(layer_out, 1)

    local_cost = tf.reduce_sum(tf.divide(tf.abs(local_prediction - label), label))
    loss = tf.reduce_sum(tf.divide(tf.abs(local_prediction - label), label))

    with tf.control_dependencies([local_cost]):
        update_op = tf.group(count.assign(tf.add(count, tf.cast(tf.size(label), dtype=tf.float32))),
                             loss_sum.assign(tf.add(loss_sum, loss)))

    return oid, label, local_prediction, local_cost, update_op

with tf.name_scope("training"):
    oid_train, _, _, cost, update_op_train = DeepNetwork(train_phase)
    train_op = []
    if embedding_size != 0 and dnn_layer != 0:
        deep_weights = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "deep")
        deep_gradients, global_norm = tf.clip_by_global_norm(tf.gradients(cost, deep_weights), clip_norm)
        deep_optimizer = tf.train.AdamOptimizer(learning_rate=deep_learning_rate / lr_div)
        deep_train_op = deep_optimizer.apply_gradients(zip(deep_gradients, deep_weights))
        train_op.append(deep_train_op)

init = tf.global_variables_initializer()

train_set_path = [file_dir + "train_data.txt"]

with tf.Session(config=configure) as sess:
    sess.run(init)
    if method == "train":
        for epoch in range(0, epochs):
            start = time()
            reset_op.run()
            print("Epoch " + str(epoch))
            for d_train_path in train_set_path:
                data_set = DataSetBucket(d_train_path)
                data_set.shuffle()

                while data_set.has_next():
                    batch_oids, batch_labels_data, batch_linkid_data, batch_link_seq_len_data = data_set.next()

                    _, _ = sess.run([train_op, update_op_train],
                                       feed_dict={oid: batch_oids, label: batch_labels_data,linkidFeat: batch_linkid_data,
                                       linkseqlens: batch_link_seq_len_data, train_phase: True, lr_div: (float(epoch)+1)})

                data_set.clear()
                train_mape = mean_loss.eval()
                print("CurEpoch " + " ".join([str(x) for x in [epoch, train_mape]]))

            train_mape = mean_loss.eval()

            elapsed = int(time() - start)
            print("Epoch " + " ".join([str(x) for x in [epoch, elapsed, train_mape]]))

    else:
        print("Unknown order!!!")
        
