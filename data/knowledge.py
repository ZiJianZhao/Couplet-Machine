#-*- coding:utf-8 -*-
import codecs
import pickle
import numpy as np
import re
import random
import jieba.posseg as pseg
from pyltp import Postagger
#postagger = Postagger()
#postagger.load('/slfs1/users/zjz17/tools/ltp_data/pos.model') 

def pos(filename = 'valid.txt'):
    with codecs.open(filename, 'r', encoding = 'utf-8') as fid:
        lines = fid.readlines()
    for line in lines:
        line_list = line.strip().split('\t=>\t')
        for sent in line_list:
            print sent
            words = pseg.cut(sent)
            lis = ['%s' %  flag for (word, flag) in words]
            print 'jieba: ', ' '.join([l for l in lis if l != 'x'])
            lis = [w.encode('utf-8') for w in sent.split()]
            postags = postagger.postag(lis)
            print 'ltp: ',  ' '.join(postags)
        print '=============================='
        raw_input()

def pos_vocab(filename = 'alllist.txt'):
    """Pos the word in the vocab file
    Note: http://www.voidcn.com/blog/huludan/article/p-6224283.html
        形容词：a,
        副词：c, d, 
        名词：b, 
        动词：
        数词：m, 
        其他词：e, 
    Args:
        filename (TYPE): vocab file
    
    Returns:
        TYPE: Description
    """
    with codecs.open(filename, 'r', encoding = 'utf-8') as fid:
        lines = fid.readlines()
    g = open('vocab.txt', 'w')
    res = []
    for line in lines:
        line = line.strip().encode('utf-8')
        lis = [line]
        postags = postagger.postag(lis)
        flag = ' '.join(postags)
        res.append((line, flag[0]))
        '''
        words = pseg.cut(line)
        for (word, flag) in words:
            res.append((word, flag[0]))
        '''
    dic = {}
    for tup in res:
        if dic.get(tup[1]) is None:
            dic[tup[1]] = [tup[0]]
        else:
            dic[tup[1]].append(tup[0])
    print len(dic)
    for key in dic:
        print key, len(dic[key]), 
        for word in dic[key][:30]:
            print word, 
        print '\n',

def rhyme(filename = u'syhb.txt', vocabfile = 'alllist.txt'):
    with codecs.open(filename, 'r', encoding = 'utf-8') as fid:
        lines = fid.readlines()
    lis = list(''.join(lines))
    print len(lis)
    dic = {w:i for (i,w) in enumerate(lis)}
    with codecs.open(vocabfile, 'r', encoding = 'utf-8') as fid:
        lines = fid.readlines()
    num = 0
    for word in lines:
        if dic.get(word.strip()) is None:
            num += 1
    print num
#rhyme()

def get_statistics_dict(filename = 'train.txt', gram = 1):
    with codecs.open(filename, 'r', encoding = 'utf-8') as fid:
        lines = fid.readlines()
    dic = {}
    for line in lines:
        line_list = line.split('\t=>\t')
        assert len(line_list) == 2
        shang = line_list[0].strip().split()
        xia = line_list[1].strip().split()
        assert len(shang) == len(xia)
        length = len(shang)
        for i in range(length - gram + 1):
            sword = ''.join(shang[i:i+gram])
            xword = ''.join(xia[i:i+gram])
            if dic.get(sword) is None:
                dic[sword] = {}
                dic[sword][xword] = 1
            else:
                if dic[sword].get(xword) is None:
                    dic[sword][xword] = 1
                else:
                    dic[sword][xword] += 1
    return dic