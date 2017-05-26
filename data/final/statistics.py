#-*- coding:utf-8 -*-
import codecs
import numpy as np 
import re
from collections import defaultdict
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

def plot_bar(X, Y, tag):
    #fig = plt.figure()
    plt.bar(left = X, height =  Y, width = 0.4, color = 'green')
    #for x,y in zip(X,Y):
        #plt.text(x+0.3, y+0.05, y, ha='center', va= 'bottom')
    plt.xlabel('Couplet-Length')
    maximum = int(len(Y) / 10)
    x_labels = [10 * i for i in range(maximum+1)]
    plt.xticks(x_labels, [str(i) for i in x_labels])
    plt.ylabel('Couplet-Count')
    plt.yscale('log')
    plt.title('Couplet-Length-Count-Statistics')
    #plt.show()
    plt.savefig(tag+'.jpg')
    plt.close()

def count_length(file = 'all.txt'):
    lis = []
    with codecs.open(file, 'r', encoding = 'utf-8') as f:
        lines =  f.readlines()
    for line in lines:
        line_list = line.split('\t=>\t')
        lis.append(len(line_list[0].split()))
    maximum =  max(lis)
    x = range(maximum)
    y = [0 for i in range(maximum)]
    for i in range(maximum):
        y[i] = lis.count(i)
    plot_bar(x, y, 'All')

#count_length('all.txt')

def count_words(file = 'xiafreq.txt'):
    with codecs.open(file, 'r', encoding = 'utf-8') as f:
        lines =  f.readlines()
    num = len(lines) / 5
    for i in range(12):
        l,h = lines[i].strip().split('\t')
        print '(', l, ',', h, ')', ', ',
    labels = ['1-2', '3-10', '11-50', '51-100', '>100']
    sizes = [0 for i in range(5)]
    for line in lines:
        count = int(line.strip().split('\t')[1])
        if count > 100:
            sizes[4] += 1
        elif count > 50:
            sizes[3] += 1
        elif count > 10:
            sizes[2] += 1 
        elif count > 2:
            sizes[1] += 1
        else:
            sizes[0] += 1
    colors = []
    print sizes

    plt.pie(sizes, labels=labels,autopct='%1.1f%%')
    plt.title('All Lines Word Frequency Statistics')
    plt.savefig('xword.jpg')
    plt.close()

count_words('allfreq.txt')

def count_pairs(file = 'all.txt'):
    dic = defaultdict(int)
    with codecs.open(file, 'r', encoding = 'utf-8') as f:
        lines =  f.readlines()
    for line in lines:
        line_list = line.split('\t=>\t')
        dic[line_list[0]] += 1
    lis = [0 for i in range(100)]
    for key in dic:
        count = dic[key]
        for i in range(count):
            lis[i] += 1 
    print lis 

