#-*- coding:utf-8 -*-

import re, os, sys, argparse, logging, collections
import codecs
import math, time
import random
from collections import defaultdict
import mxnet as mx
import pickle
import numpy as np

from focus_attention import FocusSeq2Seq
from filter import rescore

def select_random_sentences(filename):
    with codecs.open(filename, 'r', encoding = 'utf-8') as fid:
        lines = fid.readlines()
    random.seed(time.time())
    random.shuffle(lines)
    shanglian = []
    num = 0
    for i in range(len(lines)):
        line = lines[i]
        line_list = line.strip().split('\t=>\t')
        shang = ''.join(line_list[0].strip()[:-1].split())
        if len(shang) <= 10:
            shanglian.append(shang)
            num += 1
        if num == 10:
            break
    shanglian.sort(key=lambda string: len(string)) 
    return shanglian

def normal_line(line):
    lis = []
    tmp = u''
    for s in line:
        if len(s.strip()) != 0:
            tmp += s
    line = tmp
    for s in line:
        if u'\u4e00' <= s <= u'\u9fff':
            pass
        else:
            s = u'，'
        lis.append(s)
    if lis[-1] == u'，':
        lis = lis[:-1]
        lis.append(u'。')
    string = ' '.join(lis)
    return string



def generate(string):
    # ========= process input string ======================
    if not isinstance(string, unicode):
        string = unicode(string, 'utf-8')
    string = normal_line(string)

    # ========= choose model and choose parameter =========
    knowledge = 'view' # or 'label'
    epoch = 3

    Model = FocusSeq2Seq
    params_dir = knowledge + '_params/'
    params_prefix = 'couplet'
    sym, arg_params, aux_params = mx.model.load_checkpoint('%s%s' % (params_dir, params_prefix), epoch)

    word2idx = pickle.load(open('dicts/word2idx.vocab', 'r'))
    pos2idx = pickle.load(open('dicts/pos2idx.vocab', 'r'))
    rhyme2idx = pickle.load(open('dicts/rhyme2idx.vocab', 'r'))
    word2pos = pickle.load(open('dicts/word2pos.vocab', 'r'))
    word2rhyme = pickle.load(open('dicts/word2rhyme.vocab', 'r'))
    enc_word2idx = word2idx
    dec_word2idx = word2idx
    enc_word2rhyme = word2rhyme 
    dec_word2rhyme = word2rhyme 
    enc_word2pos = word2pos
    dec_word2pos = word2pos
    enc_pos2idx = pos2idx
    dec_pos2idx = pos2idx 
    enc_rhyme2idx = rhyme2idx 
    dec_rhyme2idx = rhyme2idx 


    dec_idx2word = {}
    for k, v in dec_word2idx.items():
        dec_idx2word[v] = k
    enc_idx2word = {}
    for k, v in enc_word2idx.items():
        enc_idx2word[v] = k

    # ------------------- get input ---------------------
    enc_string = string
    string_list =  enc_string.strip().split()
    enc_len = len(enc_string.strip().split())
    data = []
    for item in string_list:
        if enc_word2idx.get(item) is None:
            data.append(enc_word2idx.get('<unk>'))
        else:
            data.append(enc_word2idx.get(item))
    enc_data = mx.nd.array(np.array(data).reshape(1, enc_len))            
    # --------------------- beam seqrch ------------------          
    seq2seq = Model(
        enc_input_size = len(enc_word2idx), 
        enc_pos_size = len(enc_pos2idx) + 1,
        enc_rhyme_size = len(enc_rhyme2idx) + 1,
        dec_input_size = len(dec_word2idx),
        dec_pos_size = len(dec_pos2idx) + 1,
        dec_rhyme_size = len(dec_rhyme2idx) + 1,
        enc_len = enc_len,
        dec_len = 1,
        num_label = len(dec_word2idx),
        share_embed_weight = True,
        is_train = False
    )
    input_str = ""
    enc_list = enc_data.asnumpy().reshape(-1,).tolist()
    for i in enc_list:
        input_str += " " +  enc_idx2word[int(i)]

    #results = seq2seq.couplet_predict(enc_string, enc_word2idx, enc_word2pos, enc_word2rhyme, arg_params, knowledge=knowledge)
    results = seq2seq.couplet_knowledge_predict(enc_string, enc_word2idx, enc_word2pos, enc_word2rhyme, 
        arg_params, knowledge=knowledge)
    res = []
    for pair in results:
        sent = pair[1]
        mystr = ""
        for idx in sent:
            if dec_idx2word[idx] == '<eos>':
                continue
            mystr += " " +  dec_idx2word[idx]
        res.append((pair[0], mystr.strip()))       
    results = rescore(input_str, res)
    minmum = min(10, len(results))
    final = []
    for pair in results[0:minmum]:
        final.append(pair[1])
    return final
