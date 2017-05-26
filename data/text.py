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
class DuiduilianCralwer(object):
    '''This the cralwer for the web www.duiduilian.com
    '''
    def __init__(self, url = 'http://www.duiduilian.com/'):
        self.url = 'http://www.duiduilian.com/'
        self.dic = self.home_page()
        self.one_kinds = [u'春联', u'名胜古迹联', u'祝寿对联', u'节日对联', 
        u'结婚对联', u'行业对联', u'名人对联']
        self.two_kind = [u'庆贺对联', u'格言对联', u'居室对联', u'名著对联',
        u'集句对联', u'佛教对联']
        self.removed = [u'挽联', u'经典对联', u'其他对联']
        self.cralw(file = 'couplet.txt')
        self.process(file = 'couplet.txt', write_file = 'zgdlw.txt')

    def process(self, file = 'couplet.txt', write_file = 'process.txt'):
        with codecs.open(file, 'r', encoding = 'utf-8') as fid:
            lines = fid.readlines()
        g = open(write_file, 'w')
        for line in lines:
            line_list = line.split('<br/>')
            if len(line_list) < 2 or len(line_list) > 3:
                pass
            elif len(line_list) == 2:
                length0 = line_list[0][::-1].strip().find('>')
                string0 = line_list[0][-length0:].strip()
                length1 = line_list[1].strip().find('<')
                string1 = line_list[1].strip()[:length1].strip()
                if len(string0) != len(string1):
                    continue
                    print line
                    print string0, '\t=>\t', string1
                    raw_input()
                g.write('%s\t=>\t%s\n' % (string0.encode('utf8'), string1.encode('utf8')))
            elif len(line_list) == 3:
                length0 = line_list[0][::-1].strip().find('>')
                string0 = line_list[0][-length0:].strip()
                string1 = line_list[1].strip()
                if len(string0) != len(string1):
                    continue
                    print line
                    print string0, '\t=>\t', string1
                    raw_input()
                g.write('%s\t=>\t%s\n' % (string0.encode('utf8'), string1.encode('utf8')))
        g.close()

    def cralw(self, file = 'couplet.txt'):
        g = open(file, 'w')
        for key in self.dic:
            print 'Cralwing %s, %s' % (key, self.dic[key])
            if key in self.one_kinds:
                results = self.get_page_one(self.dic[key])
            else:
                results = self.get_page_two(self.dic[key])
            for line in results:
                g.write('%s\n' % line)
            print 'Total %d lines' % len(results)
            print '============================='
        g.close()

    def get_couplets_from_urls(self, lis):
        results = []
        for l in lis:
            index = 2
            url = l
            while True:
                try:
                    request = urllib2.Request(url)
                    response = urllib2.urlopen(request)
                    content = response.read()
                    soup = BeautifulSoup(content, 'lxml')
                    for tag in soup.descendants:
                        if isinstance(tag, bs4.element.Tag):
                            if tag.name == 'div' and tag.get('class') is not None:
                                if tag.get('class')[0] == 'content_zw':
                                    for p in tag.find_all('p'):
                                        line = ' '.join(s.strip() for s in str(p).split('\n'))
                                        results.append(line)
                    url = url.replace('.html', '_%d.html' % index)
                    index += 1
                except:
                    break
        return results         

    def get_url_content_soup(self, url):
        request = urllib2.Request(url)
        response = urllib2.urlopen(request)
        content = response.read()
        soup = BeautifulSoup(content, 'lxml')
        return soup

    def home_page(self):
        soup = self.get_url_content_soup(self.url)
        dic = {}
        for tag in soup.descendants:
            if isinstance(tag, bs4.element.Tag):
                if tag.name == 'div' and tag.get('class') is not None:
                    if tag.get('class')[0] == "main_box_right" and tag.h2 is not None:
                        if tag.h2.string == u'对联分类':
                            for a in tag.find_all('a'):
                                if a.string is not None and a.get('href') is not None:
                                    if a.get('href')[0] == '/':
                                        dic[a.string] = self.url + a.get('href')[1:]
                                    else:
                                        dic[a.string] = self.url + a.get('href')
        return dic

    def get_page_two(self, url):
        if url is None:
            print 'Error in get_page'
        lis = []
        index = 2
        while True:
            try:
                soup = self.get_url_content_soup(url)
                for tag in soup.descendants:
                    if isinstance(tag, bs4.element.Tag):
                        if tag.name == 'div' and tag.get('class') is not None:
                            if (' '.join(tag.get('class')) == u"pd_list_2 l_dot"):
                                for li in tag.ul.find_all('li'):
                                    if li.a is not None and li.a.get('href') is not None:
                                        if li.a.get('href')[0] == '/':
                                            lis.append(self.url + li.a.get('href')[1:])
                                        else:
                                            lis.append(self.url + li.a.get('href'))
                url = url + '%d.html' % index
                index += 1 
            except:
                break
        return self.get_couplets_from_urls(lis)


    def get_page_one(self, url):
        if url is None:
            print 'Error in get_page'
        lis = []
        soup = self.get_url_content_soup(url)
        for tag in soup.descendants:
            if isinstance(tag, bs4.element.Tag):
                if tag.name == 'div' and tag.get('class') is not None:
                    if (' '.join(tag.get('class')) == u"pd_list_3 l5 l_dot" or 
                    ' '.join(tag.get('class')) == u"pd_list_3 l3 l_dot"):
                        for a in tag.find_all('a'):
                            if a.string is not None and a.get('href') is not None:
                                if a.get('href')[0] == '/':
                                    lis.append(self.url + a.get('href')[1:])
                                else:
                                    lis.append(self.url + a.get('href'))
        return self.get_couplets_from_urls(lis)       

class ZhongHuaShiKuQuanSongCi(object):
    '''http://www.shitan.org/shiku/gs/songci/index.htm'''
    def __init__(self):
        self.prefix = 'http://www.shitan.org/shiku/gs/songci/'
        self.url = 'http://www.shitan.org/shiku/gs/songci/index.htm'
        #self.lis = self.home_page()
        #self.get_contents_from_urls(self.lis)
        

    def get_url_content_soup(self, url):
        request = urllib2.Request(url)
        response = urllib2.urlopen(request)
        content = response.read()
        soup = BeautifulSoup(content, 'lxml')
        return soup

    def home_page(self):
        soup = self.get_url_content_soup(self.url)
        lis = []
        '''
        for tag in soup.descendants:
            if isinstance(tag, bs4.element.Tag):
                if tag.name == 'li':
                    lis.append(self.prefix + tag.a.get('href'))     
        '''
        for i in range(1, 1484):
            string = '000' + str(i)
            string = string[-4:]
            lis.append(self.prefix + string + '.htm')
        return lis

    def ensure_unicode(self, v):
        if isinstance(v, str):
            v = v.decode('utf8')
        return unicode(v)

    def get_contents_from_urls(self, lis):
        results = []
        n = 0
        for url in lis:
            print url, len(results)
            try:
                request = urllib2.Request(url)
                response = urllib2.urlopen(request)
                content = response.read()
                #print chardet.detect(content)
                try:
                    tmp = content.decode('gb2312').encode('utf-8')
                except:
                    try:
                        tmp = content.decode('gb18030').encode('utf-8')
                    except:
                        try:
                            tmp = content.decode('gbk').encode('utf-8')
                        except:
                            try:
                                tmp = content.decode('utf-8').encode('utf-8')
                            except:
                                continue
                n += 1
                content_lis = tmp.split('\n')
                index = 0
                string = ''
                while index < len(content_lis):
                    if content_lis[index].strip() == '<ul>':
                        string = ''
                        index += 1
                        while index < len(content_lis) and content_lis[index].strip() != '</ul>':
                            string += content_lis[index].strip()
                            index += 1
                        results.append(string)
                    index += 1
            except:
                continue
        print 'Pages: ', n
        f = open('songci.txt', 'w')
        num = 0
        for line in results:
            try:
                f.write(line+'\n')
            except:
                num += 1
                continue
        print 'miss lines: ', num
        f.close()

#a = ZhongHuaShiKuQuanSongCi()

def Xmpj():
    path = u"训蒙骈句.txt"
    f = codecs.open('xmpj.txt', 'w', encoding = 'utf-8')
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        for line in fid:
            line = line.strip()
            line_list = line.split(u"。")
            for str in line_list:
                if str.find(u"；") > 0:
                    str = str.replace(u"；", " ")
                else:
                    str = str.replace(u"，", " ")
                f.write(str+'\n')

def Lwdy():
    path = u"笠翁对韵.txt"
    f = codecs.open('lwdy.txt', 'w', encoding = 'utf-8')
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        index = 0
        for line in fid:
            line = line.strip()
            if len(line) < 10:
                continue
            index += 1
            if index < 4:
                line = line.split(" ")
                if len(line) == 1:
                    line = line[0].split(u"　")
                for st in line:
                    st = st.replace(u"对", " ")
                    st = st.replace(u"、", " ")
                    f.write(st+'\n')
            if index >= 4 and index < 6:
                line = line.replace(u"、", " ")
                f.write(line + '\n')
            if index == 6:
                line = line.replace(u"　", u"，")
                f.write(line + " ")
            if index == 7:
                line = line.replace(u"　", u"，")
                f.write(line + "\n")
                index = 0

def Slqm():
    path = u"声律启蒙.txt"
    f = codecs.open('slqm.txt', 'w', encoding = 'utf-8')
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        for line in fid:
            line = line.strip()
            line = line.split(u"。")

            list1 = line[0].split(u"，")
            for l in list1:
                st = l.replace(u"对", " ")
                f.write(st+'\n')

            list2 = line[1].split(u"，")
            for l in list2:
                st = l.replace(u"对", " ")
                f.write(st+'\n')

            list3 = line[2].split(u"，")
            if len(list3) == 2:
                print line[0]
            f.write(list3[0] + " " + list3[1] + '\n')
            st = list3[2].replace(u"对", " ")
            f.write(st + '\n')

            st = line[3].replace(u"，", " ")
            f.write(st + '\n')

            st = line[4].replace(u"，", " ")
            f.write(st + '\n')

            st = line[5].replace(u"；", " ")
            f.write(st + '\n')        

def Cldldq():
    path = u"春联对联大大全.txt"
    f = codecs.open('cldldq.txt', 'w', encoding = 'utf-8')
    result = 0
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        lines = fid.readlines()
        index = 1
        last_length = 0
        for i in xrange(len(lines)):
            line = lines[i].strip()
            if len(line) == 0:
                continue

            line_list = line.split(" ")
            length = len(line_list)
            
            if index == 0:
                if (length != last_length):
                    print "line i: ", i, last_length, length
                    print "last sent: ", last_sent 
                    print "this sent: ", line
                    break
                else:
                    result += length
                    for j in xrange(length):
                        f.write(last_sent[j] + "\t" + line_list[j]+'\n')
            index = (index + 1) % 2
            last_length = length
            last_sent = line_list
        print result
    f.close()

def Qts_for_word_embedding():
    path = u"origin/全唐诗.txt"
    f = codecs.open('qts_for_embedding.txt', 'w', encoding = 'utf-8')
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        lines = fid.readlines()
        result = u''
        for line in lines:
            line = line.strip()
            if line.find(u'--') >= 0:
                index = line.find(u'--')
                line = line[:index].strip()
            if line.find(u'（') >= 0:
                index = line.find(u'（')
                line = line[:index].strip()
            if len(line) == 0:
                continue
            #if (line.find(u'。') < 0 and line.find(u'，') < 0) or line[0:2] == u"◎卷":
            if line[0] == u'第' or line[1] == u'第' or line[0:2] == u"◎卷":
                if len(result.strip()) != 0:
                    if result.count(u'，') == result.count(u'。'):
                        lis = result.split(u'。')
                        for tmp_line in lis:
                            tmp_lis = tmp_line.split(u'，')
                            if len(tmp_lis) == 2:
                                if len(tmp_lis[0]) == len(tmp_lis[1]):
                                    f.write(tmp_lis[0].strip()+'\t=>\t'+tmp_lis[1].strip()+'\n')
                    '''lis = result.split(u'。')
                    for l in lis:
                        if l.find(u'（') >= 0 or l.find(u'）') >= 0:
                            continue
                        if len(l.strip()) != 0:
                            f.write(l+u'。\n')'''
                result = u''
                continue
            result += line
    f.close()

Qts_for_word_embedding()

def Qsc_for_word_embedding():
    path = u"origin/全宋词.txt"
    punctuations = [u'，', u'、', u'。', u'；']
    f = codecs.open('qsc_for_embedding.txt', 'w', encoding = 'utf-8')
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        lines = fid.readlines()
        for line in lines:
            string =  ''.join([s.strip() for s in line.strip().split('<p>')])
            flag = True
            for char in string:
                if (u'\u4e00' <= char <= u'\u9fff') or (char in punctuations):
                    pass
                else:
                    flag = False
                    break
            if flag:
                ''' Optional choice '''
                string_list = string.split(u'。')
                for s in string_list:
                    if len(s.strip()) == 0:
                        continue
                    f.write(s.strip()+u'。\n')
    f.close()

#Qsc_for_word_embedding()

def Qts_for_duilian():
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

    def judge_four_sents_five_words_tangshi(sent):
        raw_input()
        flag = True
        line_list = sent.strip().split(u"。")
        if len(line_list) != 3:
            return None
        sents = []
        for i in xrange(0,2):
            sent = line_list[i].strip().split(u"，")
            if len(sent) != 2 or len(sent[0]) != len(sent[1]):
                return None
            sents.append(sent[0])
            sents.append(sent[1])
        sent_len = len(sents[0])
        for i in xrange(4):
            if (len(sents[i]) != sent_len):
                return None
        return sents[2].strip()+'\t=>\t'+sents[3].strip()

    def judge_eight_sents_seven_words_tangshi(sent):
        flag = True
        line_list = sent.strip().split(u"。")
        if len(line_list) != 5:
            return None
        sents = []
        for i in xrange(0,4):
            sent = line_list[i].strip().split(u"，")
            if len(sent) != 2 or len(sent[0]) != len(sent[1]):
                return None
            sents.append(sent[0])
            sents.append(sent[1])
        sent_len = len(sents[0])
        for i in xrange(8):
            if (len(sents[i]) != sent_len):
                return None
        result = []
        if judge_two_lines(sents[2], sents[3]): result.append(sents[2].strip()+'\t=>\t'+sents[3].strip())
        if judge_two_lines(sents[4], sents[5]): result.append(sents[4].strip()+'\t=>\t'+sents[5].strip())
        return result
    dic = {u'一':1, u'二':2, u'三':3, u'四': 4, u'五': 5, u'六':6, u'七': 7, u'八': 8, u'九':9, u'十': 10}
    path = u"origin/全唐诗.txt"
    f = codecs.open('qts_for_duilian.txt', 'w', encoding = 'utf-8')
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as fid:
        lines = fid.readlines()
        index = 0
        length = len(lines)
        num = 0
        while index < length:
            result = ""
            line = lines[index].strip()
            '''
            tag = None
            if line[0:2] == u'◎卷':
                line_list = line.split(u'】')
                if line_list[0][-1] == u'首':
                    tag = dic.get(line_list[0][-2])
                    #num += 1
            '''
            while len(line) > 0 and line[0:2] != u"◎卷" and line[1] != u'第' and index < length - 1:
                result = result + line
                index += 1
                line = lines[index].strip()
            if result.count(u"。") == 2 and result.count(u"，") == 2:
                # sent = judge_four_sents_five_words_tangshi(result)
                # this is not determined
                sent = None
                if sent is not None:
                    f.write(sent+'\n')
            elif result.count(u"。") == 4 and result.count(u"，") == 4:
                sents = judge_eight_sents_seven_words_tangshi(result)
                if sents is not None:
                    for sent in sents:
                        num += 1
                        f.write(sent+'\n')
            else:
                pass
            index += 1
        print num
    f.close()

#Qts_for_duilian()

def Zgdl(path = 'origin/中国对联.txt', write_file = 'origin/zgdl.txt'):
    left_signal = u'“'
    right_signal = u'”'
    split_signal = u'；'
    g = open(write_file, 'w')
    lis = []
    with codecs.open(path, 'r', encoding = 'utf-8', errors = 'ignore') as f:
        for line in f.readlines():
            line = line.strip()
            index = 0
            while index < len(line):
                if line[index] == left_signal:
                    string = ''
                    index += 1
                    while index < len(line) and line[index] != right_signal:
                        string += line[index]
                        index += 1
                    string_list = string.split(split_signal)
                    if len(string_list) == 2 and len(string_list[0])+1 == len(string_list[1]):
                        g.write(string.encode('utf-8')+'\n')
                else:
                    index += 1
    g.close()

def Zgdl2(file = 'origin/zgdl.txt', write_file = 'origin/zgdl2.txt'):
    g = open(write_file, 'w')
    split_signal = u'；'
    with codecs.open(file, 'r', encoding = 'utf-8') as f:
        for line in f.readlines(): 
            line_list = line.strip().split(split_signal)
            if len(line_list) != 2:
                continue
            str1 = ' '.join(list(line_list[0]))
            str2 = ' '.join(list(line_list[1][:-1]))
            g.write(str1.encode('utf-8')+'\t=>\t'+str2.encode('utf-8')+'\n')
    g.close()



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

#BuildDict('all.txt')

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

def normal_text(file = 'duilian.ori.txt', write_file = 'final.txt', length=100):
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

#normal_text('cldldq+zgdl+zgdldq+zgdlw.txt')
'''
def normal_text2(file = 'duilian.ori.txt', write_file = 'final.txt'):
    strange_symbols = [u'/', u'-', u' ', u'—', u'…', u'□', u'　', u'《', u'》', u'○', u'“', u'”',
    u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8', u'9', u'Ｑ', u'（', u'）']
    change_symbols = {u',': u'，', u';' : u'；'}
    punctuations = [u'？', u'；', u'．', u'：', u'、', u',', u';', u'。']
    g = open(write_file, 'w')
    white_spaces = re.compile(r'[ \n\r\t]+')
    with codecs.open(file, 'r', encoding = 'utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if len(line) >= 80: # the length should be carefully defined here
                continue
            for punc in punctuations:
                line = line.replace(punc, u'，')
            flag = False
            for symbol in strange_symbols:
                if line.find(symbol) >= 0:
                    flag = True
                    break
            if flag:
                continue
            line_list = line.strip().split('\t=>\t')

            #line_list = white_spaces.split(line.strip())
            if len(line_list) != 2:
                continue
            if len(line_list[0]) != len(line_list[1]):
                if line_list[1][-1] == u'。':
                    line_list[1] = line_list[1][:-1]
                if len(line_list[0]) != len(line_list[1]):
                    continue
            line1 = []
            for s in line_list[0]:
                if len(s.strip()) != 0:
                    line1.append(s)
            line2 = []
            for s in line_list[1]:
                if len(s.strip()) != 0:
                    line2.append(s)
            str1 = ' '.join(line1)
            str2 = ' '.join(line2)
            g.write(str1.encode('utf-8')+'\t=>\t'+str2.encode('utf-8')+'\n')
    g.close()
'''

#normal_text(file = 'all.txt', write_file = 'final.txt')
#BuildDict('final.txt')

def split_train_valid_test(file = 'duilian.txt', percent = 0.05):
    with codecs.open(file, 'r', encoding = 'utf-8') as f:
        lines = f.readlines()
    random.seed(1)
    random.shuffle(lines)
    index0 = int(len(lines) * 0.05)
    index1 = int(len(lines) * 0.10)
    g = open('test.txt', 'w')
    for i in range(0, index0):
        line = lines[i]
        g.write(line.encode('utf-8'))
    g.close()
    t = open('valid.txt', 'w')
    for i in range(index0, index1):
        line = lines[i]
        t.write(line.encode('utf-8'))
    t.close()
    h = open('train.txt', 'w')
    for i in range(index1, len(lines)):
        line = lines[i]
        h.write(line.encode('utf-8'))
    h.close()
    BuildDict('train.txt')


def split_valid_and_test_from_common(filename = 'cldldq+zgdl+zgdldq+zgdlw.txt'):
    normal_text(file = filename, write_file = 'duilian.txt')
    split_train_valid_test(file = 'duilian.txt', percent = 0.05)

#normal_text(file = 'cldldq+zgdl+zgdldq+zgdlw.txt', write_file = 'common.txt')

#split_valid_and_test_from_common()

#BuildDict('train.txt')