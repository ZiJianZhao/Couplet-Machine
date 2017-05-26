#-*- coding:utf-8 -*-
import codecs
import pickle
import numpy as np
import re
import random
import bs4
import codecs
from collections import defaultdict


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
    if u'\u4e00' <= lis[-1] <= u'\u9fff':
        lis.append(u'。')
    string = ' '.join(lis)
    return string

def get_repetations(string_list):
    '''string_list is of type list '''
    repetations = []
    dic = {}
    for i in range(len(string_list)-1, -1, -1):
        dic[string_list[i]] = i 
    for i in range(len(string_list)):
        repetations.append(dic[string_list[i]])
    return repetations

def judge_two_lines(line0, line1):
    if len(line0) != len(line1):
        return False
    length = len(line0)
    sline = line0.split()
    xline = line1.split()
    s = get_repetations(sline)
    x = get_repetations(xline)
    if s != x:
        return False
    for i in range(length):
        flag0 = (u'\u4e00' <= line0[i] <= u'\u9fff')
        flag1 = (u'\u4e00' <= line1[i] <= u'\u9fff')
        flag = (flag0 is flag1)
        if not flag:
            return False
    return True

def normal_text(file = 'qts_for_embedding.txt', write_file = 'final.txt', length=100):
    g = open(write_file, 'w')
    white_spaces = re.compile(r'[ \n\r\t]+')
    uniq_dict = {}
    num = 0
    with codecs.open(file, 'r', encoding = 'utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) > length: # 34 the length should be carefully defined here
                continue
            line_list = line.strip().split('\t=>\t')
            #line_list = white_spaces.split(line.strip())
            if len(line_list) != 2:
                continue
            string0 = normal_line(line_list[0])
            string1 = normal_line(line_list[1])
            if not judge_two_lines(string0, string1):
                num += 1
                continue
            string = string0.encode('utf-8')+'\t=>\t'+string1.encode('utf-8')
            if uniq_dict.get(string) is None:
                uniq_dict[string] = 0
                g.write(string+'\n')
    print 'num: %d' % num
    g.close()
#normal_text()

def count_pairs(file = 'all.txt'):
    dic = defaultdict(list)
    g = open('repeat.txt', 'w')
    with codecs.open(file, 'r', encoding = 'utf-8') as f:
        lines =  f.readlines()
    for line in lines:
        line_list = line.split('\t=>\t')
        dic[line_list[0]].append(line_list[1])
    lis = [0 for i in range(100)]
    for key in dic:
        count = len(dic[key])
        for i in range(count):
            lis[i] += 1
    print lis 
    for key in dic:
        if len(dic[key]) > 3:
            for cmnt in dic[key]:
                g.write(key.strip().encode('utf-8')+'\t=>\t'+cmnt.strip().encode('utf-8')+'\n')
    g.close()

def split_train_valid(file='final.txt', valid_ratio=0.05):
    dic = defaultdict(list)
    with codecs.open(file, 'r', encoding = 'utf-8') as f:
        lines =  f.readlines()
    g = open('valid.txt', 'w')
    length = len(lines)
    valid_length = int(valid_ratio*length)
    for i in range(valid_length):
        g.write(lines[i].encode('utf-8'))
    g.close()
    g = open('train.txt', 'w')
    length = len(lines)
    valid_length = int(valid_ratio*length)
    for i in range(valid_length, length):
        g.write(lines[i].encode('utf-8'))   
    g.close()

#split_train_valid()

def BuildDict(path):
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        sdict = defaultdict(int)
        xdict = defaultdict(int)
        adict = {}
        for line in fid:
            line = line.strip().split('\t=>\t')
            assert len(line) == 2
            line1 = line[0].strip().split(' ')
            line2 = line[1].strip().split(' ')
            for i in xrange(len(line1)):
                if len(line1[i]) == 0:
                    print line[0]
                    return
                sdict[line1[i]] += 1
                xdict[line2[i]] += 1    
        slist = sdict.keys()
        xlist = xdict.keys()
        alist = list(set(slist+xlist))
        for word in alist:
            count = 0
            if sdict.get(word) is not None:
                count += sdict.get(word)
            if xdict.get(word) is not None:
                count += xdict.get(word)
            adict[word] = count
        sortedsList= sorted(sdict.iteritems(), key=lambda d:d[1], reverse = True)
        sortedxList= sorted(xdict.iteritems(), key=lambda d:d[1], reverse = True)
        sortedaList= sorted(adict.iteritems(), key=lambda d:d[1], reverse = True)
        with codecs.open("shangfreq.txt", 'w', encoding='utf-8') as g:
            for i in xrange(len(sortedsList)):
                mstr = sortedsList[i][0] + "\t" + str(sortedsList[i][1])
                g.write(mstr + "\n")
        with codecs.open("xiafreq.txt", 'w', encoding='utf-8') as g:
            for i in xrange(len(sortedxList)):
                mstr = sortedxList[i][0] + "\t" + str(sortedxList[i][1])
                g.write(mstr + "\n")
        with codecs.open("allfreq.txt", 'w', encoding='utf-8') as g:
            for i in xrange(len(sortedaList)):
                mstr = sortedaList[i][0] + "\t" + str(sortedaList[i][1])
                g.write(mstr + "\n")
        slist.sort()
        xlist.sort()
        alist.sort()
        with codecs.open("shanglist.txt", 'w', encoding='utf-8') as h:
            for i in slist:
                h.write(i + "\n")
        with codecs.open("xialist.txt", 'w', encoding='utf-8') as h:
            for i in xlist:
                h.write(i + "\n")
        with codecs.open("alllist.txt", 'w', encoding='utf-8') as h:
            for i in alist:
                h.write(i + "\n")

#BuildDict('train.txt')

def count_repeat(file1='alllist.txt', file2='../final/alllist.txt'):
    with codecs.open(file1, 'r', encoding = 'utf-8') as f:
        lines1 =  f.readlines()
    with codecs.open(file2, 'r', encoding = 'utf-8') as f:
        lines2 =  f.readlines()
    dic = defaultdict(int)
    for line in lines1:
        dic[line.strip()] += 1 
    for line in lines2:
        dic[line.strip()] += 1
    count = len(lines2)
    num = 0
    for key in dic:
        if dic[key] == 2:
            num += 1 
    print 'num: %d, count: %d' % (num, count)

#count_repeat()