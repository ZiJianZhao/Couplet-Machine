#-*- coding:utf-8 -*-
import logging
import codecs
import re
import random
from collections import namedtuple

import mxnet as mx
import numpy as np
from sklearn.cluster import KMeans
from pyltp import Postagger
import pypinyin
from pypinyin import pinyin

def generate_buckets(enc_dec_data, num_buckets):
    enc_dec_data = np.array(enc_dec_data)
    kmeans = KMeans(n_clusters = num_buckets, random_state = 1) # use clustering to decide the buckets
    assignments = kmeans.fit_predict(enc_dec_data) # get the assignments
    # get the max of every cluster
    clusters = np.array([np.max( enc_dec_data[assignments==i], axis=0 ) for i in range(num_buckets)])

    buckets = []
    for i in xrange(num_buckets):
        buckets.append((clusters[i][0], clusters[i][1]))
    return buckets 

def read_dict(path):
    word2idx = {'<pad>' : 0, '<eos>' : 1, '<unk>' : 2}
    idx = 3
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        for line in fid:
            line = line.strip(' ').strip('\n')
            if len(line) == 0:
                continue
            if word2idx.get(line) == None:
                word2idx[line] = idx
            idx += 1
    return word2idx

def get_pos_dict(path, word2idx):
    postagger = Postagger()
    postagger.load('/slfs1/users/zjz17/tools/ltp_data/pos.model') 
    word2pos = {'<pad>' : 0, '<eos>' : 0, '<unk>' : 0}
    index = 1
    pos2idx = {}
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        for line in fid:
            pos2idx[line.strip()] = index
            index += 1
    for key in word2idx:
        if key not in word2pos:
            lis = [key.encode('utf-8')]
            postags = postagger.postag(lis)
            pos = ' '.join(postags)[0]
            if pos2idx.get(pos) is None:
                raise  Exception("Invalid level!")
            word2pos[key] = pos2idx.get(pos)
    return word2pos, pos2idx

import re
def get_rhyme(string):
    string = ''.join(string.split())
    rhyme = pinyin(string, style=pypinyin.TONE2)
    result = []
    for i in range(len(rhyme)):
        if string[i] == u'。' or string[i] == u'，':
            result.append(3)
            continue
        word = rhyme[i][0]
        lis = re.findall(r'\d+', word)
        if len(lis) == 0:
            tone = 1
        else:
            tone = int(lis[0])
        if tone == 1 or tone == 2:
            result.append(1)
        elif tone == 3 or tone == 4:
            result.append(2)
        else:
            result.append(0)
    if len(result) != len(string):
        print string
        raw_input()
    return result

def get_rhyme_dict(path, word2idx):
    '''
    pad, eos, unk: 0,
    ping sheng: 1,
    ze sheng: 2,
    biao dian: 3,
    rhyme2idx is no use here, just for correspondance to pos2idx
    ''' 
    word2rhyme = {'<pad>' : 0, '<eos>' : 0, '<unk>' : 0}
    index = 1
    rhyme2idx = {}
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        for line in fid:
            rhyme2idx[line.strip()] = index
            index += 1
    rhyme2idx['b'] = index

    for key in word2idx:
        if key not in word2rhyme:
            result = get_rhyme(key)
            if result[0] > 3:
                raise  Exception("Invalid level!")
            word2rhyme[key] = result[0]
    return word2rhyme, rhyme2idx

def get_enc_dec_text_id(path, enc_word2idx, dec_word2idx):
    enc_data = []
    dec_data = []
    white_spaces = re.compile(r'[ \n\r\t]+')
    index = 0
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        for line in fid:
            line = line.strip()
            line_list = line.split('\t=>\t')
            length = len(line_list)
            for i in xrange(1, length):
                enc_list = line_list[0].strip().split()
                dec_list = line_list[i].strip().split()
                enc = [enc_word2idx.get(word) if enc_word2idx.get(word) is not None else enc_word2idx.get('<unk>') for word in enc_list]
                dec = [dec_word2idx.get(word) if dec_word2idx.get(word) is not None else  dec_word2idx.get('<unk>') for word in dec_list]
                enc_data.append(enc)
                dec_data.append(dec)
                if index == 0:
                    print 'Text2digit Preprocess Example:'
                    print line_list[0].strip().encode('utf-8'), '\t=>\t', line_list[1].strip().encode('utf-8')
                    print enc, '\t=>\t', dec
                index += 1
    return enc_data, dec_data

def get_pos(string, pos2idx):
    lis = string.strip().split()
    lis = [w.encode('utf-8') for w in lis]
    postags = postagger.postag(lis)
    pos = [l[0] for l in list(postags)]
    res = [pos2idx.get(word) if pos2idx.get(word) is not None else 0 for word in pos]
    return res

def get_enc_dec_pos_id(path, enc_pos2idx, dec_pos2idx):
    postagger = Postagger()
    postagger.load('/slfs1/users/zjz17/tools/ltp_data/pos.model') 
    enc_data = []
    dec_data = []
    white_spaces = re.compile(r'[ \n\r\t]+')
    index = 0
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        for line in fid:
            line = line.strip()
            line_list = line.split('\t=>\t')
            length = len(line_list)
            for i in xrange(1, length):
                enc_list = line_list[0].strip().split()
                enc_pos = [w.encode('utf-8') for w in enc_list]
                postags = postagger.postag(enc_pos)
                enc_pos = [l[0] for l in list(postags)]
                dec_list = line_list[i].strip().split()
                dec_pos = [w.encode('utf-8') for w in dec_list]
                postags = postagger.postag(dec_pos)
                dec_pos = [l[0] for l in list(postags)]
                enc = [enc_pos2idx.get(word) if enc_pos2idx.get(word) is not None else 0 for word in enc_pos]
                dec = [dec_pos2idx.get(word) if dec_pos2idx.get(word) is not None else 0 for word in dec_pos]
                enc_data.append(enc)
                dec_data.append(dec)
                if index == 0:
                    print 'Text2pos Preprocess Example:'
                    print line_list[0].strip().encode('utf-8'), '\t=>\t', line_list[1].strip().encode('utf-8')
                    print enc, '\t=>\t', dec
                index += 1
    return enc_data, dec_data

def get_enc_dec_rhyme_id(path, enc_pos2idx, dec_pos2idx):
    enc_data = []
    dec_data = []
    white_spaces = re.compile(r'[ \n\r\t]+')
    index = 0
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        for line in fid:
            line = line.strip()
            line_list = line.split('\t=>\t')
            length = len(line_list)
            for i in xrange(1, length):
                enc_string = line_list[0].strip()
                enc = get_rhyme(enc_string)
                dec_string = line_list[i].strip()
                dec = get_rhyme(dec_string)
                enc_data.append(enc)
                dec_data.append(dec)
                if index == 0:
                    print 'Text2rhyme Preprocess Example:'
                    print line_list[0].strip().encode('utf-8'), '\t=>\t', line_list[1].strip().encode('utf-8')
                    print enc, '\t=>\t', dec
                index += 1
    return enc_data, dec_data

class EncoderDecoderIter(mx.io.DataIter):
    """This iterator is specially defined for the Couplet Generation
    
    """
    def __init__(self, enc_data, dec_data, enc_pos, dec_pos, enc_rhyme, dec_rhyme, batch_size,
        shuffle, knowledge, buckets, pad=0, eos=1):
        
        super(EncoderDecoderIter, self).__init__()
        # initilization

        self.enc_data = enc_data
        self.dec_data = dec_data
        self.enc_pos = enc_pos 
        self.dec_pos = dec_pos
        self.enc_rhyme = enc_rhyme
        self.dec_rhyme = dec_rhyme
        self.shuffle = shuffle
        self.knowledge = knowledge
        self.data_len = len(self.enc_data)
        self.pad = pad
        self.eos = eos
        self.batch_size = batch_size
        if self.knowledge == 'view':
            self.data_names = ['enc_data', 'enc_mask', 'enc_pos', 'enc_rhyme', 'dec_data', 'dec_mask', 'dec_pos', 'dec_rhyme']
            self.label_names = ['label']
        else:
            self.data_names = ['enc_data', 'enc_mask', 'dec_data', 'dec_mask']
            self.label_names = ['label', 'enc_pos', 'dec_pos', 'enc_rhyme', 'dec_rhyme']            
        # process buckets
        self.buckets = sorted(buckets)
        enc_len = max([bucket[0] for bucket in self.buckets])
        dec_len = max([bucket[1] for bucket in self.buckets])
        self.default_bucket_key = (enc_len, dec_len)
        self.assignments = []
        for idx in range(self.data_len):
            for bkt in range(len(self.buckets)):
                if len(self.enc_data[idx]) <= self.buckets[bkt][0] and len(self.dec_data[idx]) <= self.buckets[bkt][1]:
                    break
            self.assignments.append(bkt)
        buckets_count = [0 for i in range(len(self.buckets))]
        for idx in self.assignments:
            buckets_count[idx] += 1
        print 'buckets: ', self.buckets
        print 'buckets_count: ', buckets_count
        print 'default_bucket_key: ', self.default_bucket_key
        
        # generate the data , mask, label numpy array
        self.enc_data, self.enc_mask, self.enc_pos, self.enc_rhyme, self.dec_data, self.dec_mask, self.dec_pos, self.dec_rhyme, self.label = self.make_numpy_array()


        # make a random data iteration plan
        self.plan = []
        for (i, buck) in enumerate(self.enc_data):
            self.plan.extend([(i,j) for j in range(0, buck.shape[0] - batch_size + 1, batch_size)])
        if self.shuffle:
            self.idx = [np.random.permutation(x.shape[0]) for x in self.enc_data]
        else:
            self.idx = [np.arange(x.shape[0]) for x in self.enc_data]
        self.curr_plan = 0
        self.reset()
    
    def reset(self):
        self.curr_plan = 0
        if self.shuffle:
            random.shuffle(self.plan)
            for idx in self.idx:
                np.random.shuffle(idx)    

    def next(self):
        if self.curr_plan == len(self.plan):
            raise StopIteration
        i, j = self.plan[self.curr_plan]
        self.curr_plan += 1
        index = self.idx[i][j:j+self.batch_size] 

        enc_data = mx.nd.array(self.enc_data[i][index])
        enc_mask = mx.nd.array(self.enc_mask[i][index])
        enc_pos = mx.nd.array(self.enc_pos[i][index])
        enc_rhyme = mx.nd.array(self.enc_rhyme[i][index])
        dec_data = mx.nd.array(self.dec_data[i][index])
        dec_mask = mx.nd.array(self.dec_mask[i][index])
        dec_pos = mx.nd.array(self.dec_pos[i][index])   
        dec_rhyme = mx.nd.array(self.dec_rhyme[i][index])   
        label = mx.nd.array(self.label[i][index])
        if self.knowledge == 'view':
            data_all = [enc_data, enc_mask, enc_pos, enc_rhyme, dec_data, dec_mask, dec_pos, dec_rhyme]
            label_all = [label]
        else:
            data_all = [enc_data, enc_mask, dec_data, dec_mask]
            label_all = [label, enc_pos, dec_pos, enc_rhyme, dec_rhyme]            
        data_names = self.data_names
        label_names = self.label_names

        return mx.io.DataBatch(
            data = data_all,
            label = label_all,
            bucket_key = self.buckets[i],
            provide_data = zip(data_names, [data.shape for data in data_all]),
            provide_label = zip(label_names, [label.shape for label in label_all])
        )

    @property
    def provide_data(self):
        if self.knowledge == 'view':
            return [('enc_data' , (self.batch_size, self.default_bucket_key[0])),
                  ('enc_mask' , (self.batch_size, self.default_bucket_key[0])),
                  ('enc_pos' , (self.batch_size, self.default_bucket_key[0])),
                  ('enc_rhyme' , (self.batch_size, self.default_bucket_key[0])),
                  ('dec_data' , (self.batch_size, self.default_bucket_key[1])),
                  ('dec_mask' , (self.batch_size, self.default_bucket_key[1])),
                  ('dec_pos' , (self.batch_size, self.default_bucket_key[1])),
                  ('dec_rhyme' , (self.batch_size, self.default_bucket_key[1]))]
        else:
            return [('enc_data' , (self.batch_size, self.default_bucket_key[0])),
                  ('enc_mask' , (self.batch_size, self.default_bucket_key[0])),
                  ('dec_data' , (self.batch_size, self.default_bucket_key[1])),
                  ('dec_mask' , (self.batch_size, self.default_bucket_key[1]))]         

    @property
    def provide_label(self):
        if self.knowledge == 'view':
            return [('label', (self.batch_size, self.default_bucket_key[1]))]
        else:
            return [('label', (self.batch_size, self.default_bucket_key[1])),
                    ('enc_pos' , (self.batch_size, self.default_bucket_key[0])),
                    ('dec_pos' , (self.batch_size, self.default_bucket_key[1])),
                    ('enc_rhyme' , (self.batch_size, self.default_bucket_key[0])),
                    ('dec_rhyme' , (self.batch_size, self.default_bucket_key[1]))]            
            
        
    def make_data_line(self, i, bucket):
        data = self.enc_data[i]
        label = self.dec_data[i]
        enc_len = bucket[0]
        dec_len = bucket[1]
        ed = np.full(enc_len, self.pad, dtype = float)
        dd = np.full(dec_len, self.pad, dtype = float)
        ep = np.full(enc_len, 0, dtype = float)
        dp = np.full(dec_len, 0, dtype = float)
        er = np.full(enc_len, 0, dtype = float)
        dr = np.full(dec_len, 0, dtype = float)
        em = np.zeros(enc_len, dtype = float)
        dm = np.zeros(dec_len, dtype = float)      
        l  = np.full(dec_len, self.pad, dtype = float)
        
        #ed[enc_len-len(data):enc_len] = data
        #em[enc_len-len(data):enc_len] = 1.0
        ed[0:len(data)] = data
        em[0:len(data)] = 1.0
        if self.knowledge == 'view':
            ep[0:len(data)] = self.enc_pos[i]
            dp[0] = 0
            dp[1:len(label)+1] = self.dec_pos[i]
            er[0:len(data)] = self.enc_rhyme[i]
            dr[0] = 0 
            dr[1:len(label)+1] = self.dec_rhyme[i]
        else:
            # predict the next word property
            '''
            ep[0:len(data)-1] = self.enc_pos[i][1:]
            ep[len(data)-1] = 0
            dp[0:len(label)] = self.dec_pos[i]
            dp[len(label)] = 0 
            '''
            # predict the current word 
            ep[0:len(data)] = self.enc_pos[i]
            dp[0] = 0
            dp[1:len(label)+1] = self.dec_pos[i] 
            er[0:len(data)] = self.enc_rhyme[i]
            dr[0] = 0 
            dr[1:len(label)+1] = self.dec_rhyme[i]
                      
        dd[0] = self.eos
        dd[1:len(label)+1] = label 
        dm[0:len(label)+1] = 1.0
        l[0:len(label)] = label
        l[len(label)] = self.eos
        
        return ed, em, ep, er, dd, dm, dp, dr, l

    def make_numpy_array(self):
        enc_data = [[] for _ in self.buckets]
        enc_mask = [[] for _ in self.buckets]
        enc_pos = [[] for _ in self.buckets]
        enc_rhyme = [[]for _ in self.buckets]
        dec_data = [[] for _ in self.buckets]
        dec_mask = [[] for _ in self.buckets]
        dec_pos = [[] for _ in self.buckets]
        dec_rhyme = [[] for _ in self.buckets]
        label  = [[] for _ in self.buckets]

        for i in xrange(self.data_len):
            bkt_idx = self.assignments[i]
            ed, em, ep, er, dd, dm, dp, dr, l = self.make_data_line(i, self.buckets[bkt_idx])
            enc_data[bkt_idx].append(ed)
            enc_mask[bkt_idx].append(em)
            enc_pos[bkt_idx].append(ep)
            enc_rhyme[bkt_idx].append(er)
            dec_data[bkt_idx].append(dd)
            dec_mask[bkt_idx].append(dm)
            dec_pos[bkt_idx].append(dp)
            dec_rhyme[bkt_idx].append(dr)
            label[bkt_idx].append(l)
        enc_data = [np.asarray(i, dtype='float32') for i in enc_data]
        enc_mask  = [np.asarray(i, dtype='float32') for i in enc_mask]
        enc_pos  = [np.asarray(i, dtype='float32') for i in enc_pos]
        enc_rhyme  = [np.asarray(i, dtype='float32') for i in enc_rhyme]
        dec_data = [np.asarray(i, dtype='float32') for i in dec_data]
        dec_mask  = [np.asarray(i, dtype='float32') for i in dec_mask]
        dec_pos  = [np.asarray(i, dtype='float32') for i in dec_pos]
        dec_rhyme  = [np.asarray(i, dtype='float32') for i in dec_rhyme]
        label = [np.asarray(i, dtype='float32') for i in label]
        return enc_data, enc_mask, enc_pos, enc_rhyme, dec_data, dec_mask, dec_pos, dec_rhyme, label

