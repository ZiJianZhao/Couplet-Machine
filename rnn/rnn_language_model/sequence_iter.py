import re, os, sys, argparse, logging, collections
import codecs, random
import bisect
import mxnet as mx
import numpy as np
from sklearn.cluster import KMeans

def build_dict(data_dir, data_name, dict_name):
    data_path = data_dir + data_name
    dict_path = data_dir + dict_name
    with codecs.open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read().decode("utf-8").replace("\n", "<eos>").split()
        counter = collections.Counter(content)
        count_pairs = sorted(counter.items(), key = lambda x: (-x[1], x[0]))
        words, _ = list(zip(*count_pairs))
    with codecs.open(dict_path, 'w', encoding='utf-8', errors='ignore') as g:
        for ch in words:
            g.write(ch+'\n')

def read_dict(path):
    word2idx = {'<pad>' : 0, '<eos>' : 1, '<unk>' : 2}
    idx = 3
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        for line in fid:
            line = line.strip('\n')
            if len(line) == 0:
                continue
            if word2idx.get(line) == None:
                word2idx[line] = idx
                idx += 1
    return word2idx

def get_text_id(path, word2idx):
    data = []
    label = []
    index = 0
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        for line in fid:
            line = line.strip('\n').split()
            if len(line) == 0:
                continue
            tmp = [word2idx.get(word) if word2idx.get(word) is not None else word2idx.get('<unk>') for word in line]
            tmp.insert(0, word2idx.get('<eos>'))
            data.append(tmp[:])
            tmp.append(word2idx.get('<eos>'))
            label.append(tmp[1:])
            if index == 0:
                print 'Text2id Example:'
                print line
                print tmp[:-1]
            index += 1
    return data, label


def generate_buckets(sequence_length, num_buckets):
    sequence_length = np.array(sequence_length).reshape(-1,1)
    kmeans = KMeans(n_clusters = num_buckets, random_state = 1) # use clustering to decide the buckets
    assignments = kmeans.fit_predict(sequence_length) # get the assignments
    # get the max of every cluster
    buckets = np.array([np.max( sequence_length[assignments==i], axis=0 ) for i in range(num_buckets)])
    # get # of sequences in each bucket... then assign the batch size as the minimum(minimum(bucketsize), batchsize)
    buckets_count = np.array([sequence_length[assignments==i].shape[0] for i in range(num_buckets) ] )
    # define attributes future use
    buckets = [bucket[0] for bucket in buckets]
    return buckets

class SequenceIter(mx.io.DataIter):
    ''' This the dataiter for language model
    
    Args:
        data: preprocessed input sequence
        label: preprocessed output sequence
        batch_size: batch_size of data
        num_buckets: number of buckets. 
        pad: the pad label, always be 0, consistent with vocab.
    '''
    def __init__(self, data, label, batch_size, buckets, pad=0):
        # Initialization
        super(SequenceIter, self).__init__() 

        self.data = data
        self.label = label 
        self.data_num = len(data)
        self.data_name = 'data'
        self.label_name = 'label'
        self.batch_size = batch_size
        self.pad = pad

        # Assign id for buckets
        self.buckets = sorted(buckets)
        self.default_bucket_key = max(self.buckets)
        self.assignments = []
        for sent in data:
            idx = bisect.bisect_left(self.buckets, len(sent))
            self.assignments.append(idx)
        buckets_count = [0 for i in range(len(self.buckets))]
        for idx in self.assignments:
            buckets_count[idx] += 1
        print 'buckets: ', self.buckets
        print 'buckets_count: ', buckets_count
        print 'default_bucket_key: ', self.default_bucket_key

        # define attributes
        self.provide_data = [(self.data_name , (self.batch_size, self.default_bucket_key))]
        self.provide_label = [(self.label_name, (self.batch_size, self.default_bucket_key))]
        
        # generate the data, label numpy array
        self.data, self.label = self.make_numpy_array()

        # make a random data iteration plan in each epoch
        self.plan = []
        for (i, buck) in enumerate(self.data):
            self.plan.extend([(i,j) for j in range(0, buck.shape[0] - batch_size + 1, batch_size)])
        self.idx = [np.random.permutation(x.shape[0]) for x in self.data]
        self.curr_plan = 0
        self.reset()

    def reset(self):
        self.curr_plan = 0
        random.shuffle(self.plan)
        for idx in self.idx:
            np.random.shuffle(idx)

    
    def next(self):
        if self.curr_plan == len(self.plan):
            raise StopIteration
        i, j = self.plan[self.curr_plan]
        self.curr_plan += 1
        index = self.idx[i][j:j+self.batch_size]
        
        data = mx.nd.array(self.data[i][index])
        label = mx.nd.array(self.label[i][index])
        
        return mx.io.DataBatch(
            data = [data], 
            label = [label],
            bucket_key = self.buckets[i],
            provide_data = [(self.data_name, data.shape)],
            provide_label = [(self.label_name, label.shape)]
        )

    def generate_buckets(self):
        sequence_length = []
        for i in xrange(self.data_num):
            sequence_length.append(len(self.data[i]))
        sequence_length = np.array(sequence_length).reshape(-1,1)
        kmeans = KMeans(n_clusters = self.num_buckets, random_state = 1) # use clustering to decide the buckets
        assignments = kmeans.fit_predict(sequence_length) # get the assignments
        # get the max of every cluster
        buckets = np.array([np.max( sequence_length[assignments==i], axis=0 ) for i in range(self.num_buckets)])
        # get # of sequences in each bucket... then assign the batch size as the minimum(minimum(bucketsize), batchsize)
        buckets_count = np.array([sequence_length[assignments==i].shape[0] for i in range(self.num_buckets) ] )
        # define attributes future use
        buckets = [bucket[0] for bucket in buckets]
        default_bucket_key = max(buckets)
        assignments = assignments
        return buckets, default_bucket_key, assignments

    def make_numpy_array(self):
        data = [[] for _ in self.buckets]
        label  = [[] for _ in self.buckets]
        for i in xrange(self.data_num):
            bkt_idx = self.assignments[i]
            bkt_len = self.buckets[bkt_idx]
            i_data = np.full(bkt_len, self.pad, dtype = float)
            i_label = np.full(bkt_len, self.pad, dtype = float)
            i_data[:len(self.data[i])] = self.data[i]
            i_label[:len(self.label[i])] = self.label[i]
            data[bkt_idx].append(i_data)
            label[bkt_idx].append(i_label)
        data = [np.asarray(i, dtype='float32') for i in data]
        label = [np.asarray(i, dtype='float32') for i in label]
        return data, label
