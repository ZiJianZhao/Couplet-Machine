#-*- coding:utf-8 -*-
import codecs
import pickle
import numpy as np
import re
import random
import bs4
import codecs
from collections import defaultdict
from bs4 import BeautifulSoup
import string, re
import urllib2

def normal_line(line):
    lis = []
    tmp = u''
    for s in line:
        if len(s.strip()) != 0:
            tmp += s
    line = tmp
    index = 0
    #print line
    while not (u'\u4e00' <= line[index] <= u'\u9fff'):
        index += 1
    line = line[index:]
    index = 0
    while index < len(line):
        s = line[index]
        if u'\u4e00' <= s <= u'\u9fff' or s == u'，' or s == u'；':
            pass
        else:
            break
        lis.append(s)
        index += 1
    if lis[-1] == u'，' or lis[-1] == u'；':
        lis = lis[:-1]
        lis.append(u'。')
    if u'\u4e00' <= lis[-1] <= u'\u9fff':
        lis.append(u'。')
    string = ' '.join(lis)
    return string

def process_couplet(filename='couplet.txt'):
    f = codecs.open('pro_couplet.txt', 'w', encoding = 'utf-8')
    with codecs.open(filename, 'r', encoding='utf-8', errors='ignore') as fid:
        lines = fid.readlines()
    index = 0
    length = len(lines)
    result = []
    num = 0
    while index < length - 1:
        sline = lines[index].strip()
        if sline[0:2] == u'上联':
            index += 1
            xline = lines[index].strip()
            if xline[0:2] == u'下联':
                sline = sline[3:]
                xline = xline[3:]
                if len(sline) != len(xline):
                    num += 1
                else:
                    sline = normal_line(sline)
                    xline = normal_line(xline)
                    result.append((sline, xline))
        index += 1
    print 'processed lines: %d, skipped lines: %d' % (len(result), num)
    for pair in result:
        f.write(pair[0]+'\t=>\t'+pair[1]+'\n')
    f.close()
process_couplet()



def judge_two_lines(line0, line1):
    if len(line0) != len(line1):
        return False
    length = len(line0)
    for i in range(length):
        flag0 = (u'\u4e00' <= line0[i] <= u'\u9fff')
        flag1 = (u'\u4e00' <= line1[i] <= u'\u9fff')
        flag = (flag0 is flag1)
        if not flag:
            return False
    return True

def process_couplet_2(filename='couplet_2.txt'):
    f = codecs.open('pro_couplet_2.txt', 'w', encoding = 'utf-8')
    with codecs.open(filename, 'r', encoding='utf-8', errors='ignore') as fid:
        lines = fid.readlines()
    index = 0
    length = len(lines)
    result = []
    num = 0
    while index < length - 1:
        sline = lines[index].strip()
        if len(sline) > 0:
            index += 1
            xline = lines[index].strip()
            if len(xline) > 0:
                sline = normal_line(sline)
                xline = normal_line(xline)
                if len(sline) != len(xline):
                    #print sline, '\t=>\t', xline
                    #raw_input()
                    num += 1
                else:
                    result.append((sline, xline))
        index += 1
    print 'processed lines: %d, skipped lines: %d' % (len(result), num)
    for pair in result:
        f.write(pair[0]+'\t=>\t'+pair[1]+'\n')
    f.close()

#process_couplet_2()

def process_couplet_3(filename='couplet_3.txt'):
    f = codecs.open('pro_couplet_3.txt', 'w', encoding = 'utf-8')
    with codecs.open(filename, 'r', encoding='utf-8', errors='ignore') as fid:
        lines = fid.readlines()
    index = 0
    length = len(lines)
    result = []
    num = 0
    while index < length:
        line = lines[index].strip()
        if len(line) > 0:
            idx = 0
            while not (u'\u4e00' <= line[idx] <= u'\u9fff'):
                idx += 1
            line = line[idx:]
            line_list = line.split(u'/')
            flag = True
            if len(line_list) >= 2 and len(line_list[0].strip()) > 0 and len(line_list[1].strip()) > 0:
                pass
            else:
                line_list = line.split()
                if len(line_list) >= 2 and len(line_list[0].strip()) > 0 and len(line_list[1].strip()) > 0:
                    pass
                else:
                    line_list = line.split(u'；')
                    if len(line_list) >= 2 and len(line_list[0].strip()) > 0 and len(line_list[1].strip()) > 0:
                        pass
                    else:
                        line_list = line.split(u'，')
                        if len(line_list) >= 2 and len(line_list[0].strip()) > 0 and len(line_list[1].strip()) > 0:
                            pass
                        else:
                            flag = False
                            #num += 1
        if flag:
            sline = line_list[0]
            xline = line_list[1]
            if sline[0:2] == u'上联':
                sline = sline[3:]
                xline = xline[3:]
            sline = normal_line(sline)
            xline = normal_line(xline)
            if len(sline) != len(xline):
                num += 1
                print index, line, num, sline, xline
            else:
                result.append((sline, xline))
        index += 1
    print 'processed lines: %d, skipped lines: %d' % (len(result), num)
    for pair in result:
        f.write(pair[0]+'\t=>\t'+pair[1]+'\n')
    f.close()
#process_couplet_3()

def judge_two_lines(line0, line1):
    if len(line0) != len(line1):
        return False
    length = len(line0)
    for i in range(length):
        flag0 = (u'\u4e00' <= line0[i] <= u'\u9fff')
        flag1 = (u'\u4e00' <= line1[i] <= u'\u9fff')
        flag = (flag0 is flag1)
        if not flag:
            return False
    return True

def normal_text(file = 'duilian.ori.txt', write_file = 'final.txt'):
    g = open(write_file, 'w')
    white_spaces = re.compile(r'[ \n\r\t]+')
    uniq_dict = {}
    with codecs.open(file, 'r', encoding = 'utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) > 34: # the length should be carefully defined here
                continue
            line_list = line.strip().split('\t=>\t')
            #line_list = white_spaces.split(line.strip())
            if len(line_list) != 2:
                continue
            string0 = normal_line(line_list[0])
            string1 = normal_line(line_list[1])
            if not judge_two_lines(string0, string1):
                continue
            string = string0.encode('utf-8')+'\t=>\t'+string1.encode('utf-8')
            if uniq_dict.get(string) is None:
                uniq_dict[string] = 0
                g.write(string+'\n')
    g.close()


