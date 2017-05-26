#-*- coding:utf-8 -*-
import re, os, sys, argparse, logging, collections
import codecs
import math
import random
from collections import defaultdict

import pypinyin
from pypinyin import pinyin

def read_chaizi(mode = 0):
    '''mode: 0, 检索；1, 生成 '''
    if mode == 0:
        filename=u'/home/zjz17/couplet/检索拆字联.txt'
    else:
        filename=u'/home/zjz17/couplet/生成拆字联.txt'
    with codecs.open(filename,encoding='utf-8') as f:
        lines = f.readlines()
    dic = {}
    for line in lines:
        line = line.strip()
        if len(line) == 0:
            continue
        line_list = line.split()
        assert len(line_list) == 3
        dic[line_list[2]] = [line_list[0], line_list[1]]
    return dic

def get_chaizi_dict(dic, word2idx):
    he2chai_n = {} # 不重复
    chai2he_n = defaultdict(list) # 不重复
    he2chai_c = {} # 重复
    chai2he_c = defaultdict(list) # 重复        
    for key in dic:
        try:
            index =  word2idx.get(key)
            index0 = word2idx.get(dic[key][0])
            index1 = word2idx.get(dic[key][1])
            if index0 == index1:
                he2chai_c[index] = [(index0, index1)]
                chai2he_c[index0].append((index1, index))
            else:
                he2chai_n[index] = [(index0, index1)]
                chai2he_n[index0].append((index1, index))                    
        except:
            continue
    return he2chai_c, chai2he_c, he2chai_n, chai2he_n

def get_chaizi(string_list, dic):
    '''string_list is of type list '''
    reverse_string_list = string_list[len(string_list)-1:None:-1]
    chaizi = [None for i in range(len(string_list))]
    for i, word in enumerate(string_list):
        if chaizi[i] is None and word in dic:
            try:
                if dic[word][0] != dic[word][1]:
                    index0 = string_list.index(dic[word][0])
                    index1 = string_list.index(dic[word][1])
                else:
                    index0 = string_list.index(dic[word][0])
                    index1 = len(string_list) - reverse_string_list.index(dic[word][1]) - 1                
            except:
                continue
            if index0 != -1 and index1 != -1 and index0 != index1:
                begin = min([index0, index1, i])
                end = max([index0, index1, i])
                [mid] = list(set([index0, index1, i]) - set([begin, end]))
                for index in [index0, index1, i]:
                    if index == begin:
                        chaizi[index] = ['B', begin, mid, end] # this word is the first occur word
                    elif index == end:
                        chaizi[index] = ['E', begin, mid, end] # this word is the last occur word
                    else:
                        chaizi[index] = ['I', begin, mid, end] # this word is the inside occur word
                if string_list[index0] == string_list[index1]:
                    chaizi[index0].append('CC') # first c represents chai fen zi, last c represents chong fu
                    chaizi[index1].append('CC')
                    chaizi[i].append('HC') # this word is the he bing zi
                else:
                    chaizi[index0].append('CN') 
                    chaizi[index1].append('CN')
                    chaizi[i].append('HN') 
    return chaizi


def get_repetations(string_list):
    '''string_list is of type list '''
    repetations = []
    dic = {}
    for i in range(len(string_list)-1, -1, -1):
        dic[string_list[i]] = i 
    for i in range(len(string_list)):
        repetations.append(dic[string_list[i]])
    return repetations

def get_rhyme(string_list):
    ''' string_list is of type list'''
    string_list = ''.join(string_list)
    rhyme = pinyin(string_list, style=pypinyin.TONE2)
    result = []
    for i in range(len(rhyme)):
        if string_list[i] == u'。' or string_list[i] == u'，':
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
    if len(result) != len(string_list):
        print string_list
        raw_input()
    return result

def rescore(string_list, results):
    res = []
    string_list = string_list.strip().split()
    # remove the error of length
    new_results = []
    for line in results:
        sent = line[1].strip().split()
        if len(sent) != len(string_list):
            continue
        new_results.append((line[0], sent))
    results = new_results
    # make the punctuations same
    new_results = []
    for line in results:
        sent = line[1]
        flag = True
        for i in range(len(string_list)):
            if string_list[i] == u'，' or string_list[i] == u'。':
                if sent[i] != string_list[i]:
                    flag = False
                    break
            else:
                if sent[i] == u'，' or sent[i] == u'。':
                    flag = False
                    break
        if flag:
            new_results.append((line[0], sent))
    results = new_results
    # make sure the repetations same
    new_results = []
    string_list_repetation = get_repetations(string_list)
    for line in results:
        sent = line[1]
        sent_repetation = get_repetations(sent)
        if string_list_repetation == sent_repetation:
            new_results.append((line[0], sent))
    results = new_results
    # make sure the rhyme suitable
    final = [(score, ''.join(sent)) for (score, sent) in results]

    # return the result
    return final

