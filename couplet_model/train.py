#-*- coding:utf-8 -*-

import mxnet as mx
import numpy as np

import re, os, sys, argparse, logging, collections
import codecs
import math
import random
from collections import defaultdict

from enc_dec_iter import EncoderDecoderIter, read_dict, get_rhyme_dict, get_enc_dec_rhyme_id
from enc_dec_iter import get_enc_dec_text_id, generate_buckets, get_pos_dict, get_enc_dec_pos_id
from seq2seq import Seq2Seq
from focus_attention import FocusSeq2Seq
from global_attention import GlobalSeq2Seq
from eval_and_visual import read_file
from metric import PerplexityWithoutExp
from nltk.translate.bleu_score import corpus_bleu
sys.path.append('..')

from filter import rescore

ctx = mx.gpu(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Encoder-Decoder Model Inference")
    parser.add_argument('--mode', default = 'train', type = str, 
        help='you want to train or test or generate')
    parser.add_argument('--pos', default = 'vocab', type = str, 
        help='the pos type you want to add: vocab, sent')  
    parser.add_argument('--knowledge', default = 'view', type = str, 
        help='the multi-learning type you want to use: view, label')  

    args = parser.parse_args()
    print args
    mode = args.mode
    pos = args.pos
    knowledge = args.knowledge
    Model = FocusSeq2Seq

    # ----------------- 0. Process the data  ---------------------------------------
    # ----------------- 0.1 stc data -----------------------------------------------
    task = 'couplet'
    task_dir = '..'
    data_dir = task_dir + 'data/final'
    params_dir = knowledge + '_params/'
    #params_dir = 'knowledge' + pos + '_params/'
    params_prefix = 'couplet'
    share_embed_weight = True
    if share_embed_weight:
        enc_vocab_file = 'alllist.txt'
        dec_vocab_file = 'alllist.txt'
    else:
        enc_vocab_file = 'shanglist.txt'
        dec_vocab_file = 'xialist.txt'
    pos_vocab_file = 'pos.vocab'
    rhyme_vocab_file = 'rhyme.vocab'
    train_file = 'train.txt'
    valid_file = 'valid.txt'
    test_file = 'test.txt'
     

    enc_word2idx = read_dict(os.path.join(data_dir, enc_vocab_file))
    dec_word2idx = read_dict(os.path.join(data_dir, dec_vocab_file))
    enc_word2pos, enc_pos2idx = get_pos_dict(os.path.join(data_dir, pos_vocab_file), enc_word2idx)
    dec_word2pos, dec_pos2idx = get_pos_dict(os.path.join(data_dir, pos_vocab_file), dec_word2idx)
    enc_word2rhyme, enc_rhyme2idx = get_rhyme_dict(os.path.join(data_dir, rhyme_vocab_file), enc_word2idx)
    dec_word2rhyme, dec_rhyme2idx = get_rhyme_dict(os.path.join(data_dir, rhyme_vocab_file), dec_word2idx)
    ignore_label = dec_word2idx.get('<pad>')
    
    # ----------------- 1. Configure logging module  ---------------------------------------
    # This is needed only in train mode
    if mode == 'train':
        logging.basicConfig(
            level = logging.DEBUG,
            format = '%(asctime)s %(message)s', 
            datefmt = '%m-%d %H:%M:%S %p',  
            filename = knowledge+ '_Log',
            filemode = 'w'
        )
        logger = logging.getLogger()
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        logger.addHandler(console)
    
    # -----------------2. Params Defination ----------------------------------------
    num_buckets = 3
    batch_size = 32

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)

    if mode == 'train':
        enc_train, dec_train = get_enc_dec_text_id(os.path.join(data_dir, train_file), enc_word2idx, dec_word2idx)
        enc_valid, dec_valid = get_enc_dec_text_id(os.path.join(data_dir, valid_file), enc_word2idx, dec_word2idx)
        if pos == 'vocab':
            enc_train_pos, dec_train_pos = get_enc_dec_text_id(os.path.join(data_dir, train_file), enc_word2pos, dec_word2pos)
            enc_valid_pos, dec_valid_pos = get_enc_dec_text_id(os.path.join(data_dir, valid_file), enc_word2pos, dec_word2pos)
            enc_train_rhyme, dec_train_rhyme = get_enc_dec_text_id(os.path.join(data_dir, train_file), enc_word2rhyme, dec_word2rhyme)
            enc_valid_rhyme, dec_valid_rhyme = get_enc_dec_text_id(os.path.join(data_dir, valid_file), enc_word2rhyme, dec_word2rhyme)
        else:
            enc_train_pos, dec_train_pos = get_enc_dec_pos_id(os.path.join(data_dir, train_file), enc_pos2idx, dec_pos2idx)
            enc_valid_pos, dec_valid_pos = get_enc_dec_pos_id(os.path.join(data_dir, valid_file), enc_pos2idx, dec_pos2idx)            
            enc_train_rhyme, dec_train_rhyme = get_enc_dec_rhyme_id(os.path.join(data_dir, train_file), enc_rhyme2idx, dec_rhyme2idx)
            enc_valid_rhyme, dec_valid_rhyme = get_enc_dec_rhyme_id(os.path.join(data_dir, valid_file), enc_rhyme2idx, dec_rhyme2idx) 

        # ----------------------3. Data Iterator Defination ---------------------
        sequence_length = []
        for i in range(len(enc_train)):
            sequence_length.append((len(enc_train[i]), len(dec_train[i])+1))
        for i in range(len(enc_valid)):
            sequence_length.append((len(enc_valid[i]), len(dec_valid[i])+1))
        buckets = generate_buckets(sequence_length, num_buckets)

        train_iter = EncoderDecoderIter(
            enc_data = enc_train, 
            dec_data = dec_train, 
            enc_pos = enc_train_pos,
            dec_pos = dec_train_pos,
            enc_rhyme = enc_train_rhyme,
            dec_rhyme = dec_train_rhyme,
            batch_size = batch_size,
            shuffle = True,
            knowledge = knowledge, 
            buckets = buckets, 
            pad = enc_word2idx.get('<pad>'), 
            eos = enc_word2idx.get('<eos>'),
        )
        valid_iter = EncoderDecoderIter(
            enc_data = enc_valid, 
            dec_data = dec_valid, 
            enc_pos = enc_valid_pos,
            dec_pos = dec_valid_pos,
            enc_rhyme = enc_valid_rhyme,
            dec_rhyme = dec_valid_rhyme,
            batch_size = batch_size, 
            shuffle = False,
            knowledge = knowledge,
            buckets = buckets, 
            pad = enc_word2idx.get('<pad>'), 
            eos = enc_word2idx.get('<eos>'),
        )
        frequent = train_iter.data_len / batch_size / 10 # log frequency

        def sym_gen(bucketkey):
            seq2seq = Model(
                enc_input_size = len(enc_word2idx), 
                enc_pos_size = len(enc_pos2idx) + 1,
                enc_rhyme_size = len(enc_rhyme2idx) + 1,
                dec_input_size = len(dec_word2idx),
                dec_pos_size = len(dec_pos2idx) + 1,
                dec_rhyme_size = len(dec_rhyme2idx) + 1,
                enc_len = bucketkey[0],
                dec_len = bucketkey[1],
                num_label = len(dec_word2idx),
                share_embed_weight = share_embed_weight,
                is_train = True
            )
            softmax_symbol = seq2seq.symbol_define(knowledge = knowledge)
            data_names = train_iter.data_names
            label_names = train_iter.label_names
            return (softmax_symbol, data_names, label_names)

        # ------------------4. Load paramters if exists ------------------------------

        model_args = {}

        if os.path.isfile('%s/%s-symbol.json' % (params_dir, params_prefix)):
            filelist = os.listdir(params_dir)
            paramfilelist = []
            for f in filelist:
                if f.startswith('%s-' % params_prefix) and f.endswith('.params'):
                    paramfilelist.append( int(re.split(r'[-.]', f)[1]) )
            last_iteration = max(paramfilelist)
            print('laoding pretrained model %s%s at epoch %d' % (params_dir, params_prefix, last_iteration))
            sym, arg_params, aux_params = mx.model.load_checkpoint('%s/%s' % (params_dir, params_prefix), last_iteration)
            model_args.update({
                'arg_params' : arg_params,
                'aux_params' : aux_params,
                'begin_epoch' : last_iteration
                })


        # -----------------------5. Training ------------------------------------
        if not os.path.exists(params_dir):
            os.makedirs(params_dir)

        if num_buckets == 1:
            mod = mx.mod.Module(*sym_gen(train_iter.default_bucket_key), context = [ctx])
        else:
            mod = mx.mod.BucketingModule(
                sym_gen = sym_gen, 
                default_bucket_key = train_iter.default_bucket_key, 
                context = [ctx]
            )
        mod.fit(
            train_data = train_iter, 
            eval_data = valid_iter, 
            num_epoch = 5,
            eval_metric = PerplexityWithoutExp(ignore_label),
            epoch_end_callback = [mx.callback.do_checkpoint('%s%s' % (params_dir, params_prefix), 1)],
            batch_end_callback = [mx.callback.Speedometer(batch_size, frequent = frequent)],
            initializer = mx.init.Uniform(0.05),
            optimizer = 'adam',
            optimizer_params = {'wd': 0.0000, 'clip_gradient': 0.1},
            **model_args
            #optimizer_params = {'learning_rate':0.01, 'momentum': 0.9, 'wd': 0.0000}
        )
    elif mode == 'test':
        sym, arg_params, aux_params = mx.model.load_checkpoint('%s%s' % (params_dir, params_prefix), 3)
        dec_idx2word = {}
        for k, v in dec_word2idx.items():
            dec_idx2word[v] = k
        enc_idx2word = {}
        for k, v in enc_word2idx.items():
            enc_idx2word[v] = k
        while True:
            # ------------------- get input ---------------------
            enc_string = raw_input('Enter the encode sentence:\n')
            if not isinstance(enc_string, unicode):
                enc_string = unicode(enc_string, 'utf-8')
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
                share_embed_weight = share_embed_weight,
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

            print "Encode Sentence: ", input_str
            print 'Beam Search Results: '
            minmum = min(10, len(results))
            for pair in results[0:minmum]:
                print "score : %f, %s" % (pair[0], pair[1])           
    else:
        enc_test, dec_test = get_enc_dec_text_id(os.path.join(data_dir, test_file), enc_word2idx, dec_word2idx)
        if pos == 'vocab':
            enc_test_pos, dec_test_pos = get_enc_dec_text_id(os.path.join(data_dir, test_file), enc_word2pos, dec_word2pos)
            enc_test_rhyme, dec_test_rhyme = get_enc_dec_text_id(os.path.join(data_dir, test_file), enc_word2rhyme, dec_word2rhyme)
        else:
            enc_test_pos, dec_test_pos = get_enc_dec_pos_id(os.path.join(data_dir, test_file), enc_pos2idx, dec_pos2idx)   
            enc_test_rhyme, dec_test_rhyme = get_enc_dec_rhyme_id(os.path.join(data_dir, test_file), enc_rhyme2idx, dec_rhyme2idx)        
        sequence_length = []
        for i in range(len(enc_test)):
            sequence_length.append((len(enc_test[i]), len(dec_test[i])+1))
        buckets = generate_buckets(sequence_length, num_buckets)
        test_iter = EncoderDecoderIter(
            enc_data = enc_test, 
            dec_data = dec_test, 
            enc_pos = enc_test_pos,
            dec_pos = dec_test_pos,
            enc_rhyme = enc_test_rhyme,
            dec_rhyme = dec_test_rhyme,
            batch_size = batch_size,
            shuffle = False, 
            knowledge = knowledge, 
            buckets = buckets, 
            pad = enc_word2idx.get('<pad>'), 
            eos = enc_word2idx.get('<eos>')
        )
        def sym_gen(bucketkey):
            seq2seq = Model(
                enc_input_size = len(enc_word2idx), 
                enc_pos_size = len(enc_pos2idx) + 1,
                enc_rhyme_size = len(enc_rhyme2idx) + 1,
                dec_input_size = len(dec_word2idx),
                dec_pos_size = len(dec_pos2idx) + 1,
                dec_rhyme_size = len(dec_rhyme2idx) + 1,
                enc_len = bucketkey[0],
                dec_len = bucketkey[1],
                num_label = len(dec_word2idx),
                share_embed_weight = share_embed_weight,
                is_train = True
            )
            softmax_symbol = seq2seq.symbol_define(knowledge = knowledge)
            data_names = test_iter.data_names
            label_names = test_iter.label_names
            return (softmax_symbol, data_names, label_names)
        if num_buckets == 1:
            mod = mx.mod.Module(*sym_gen(test_iter.default_bucket_key), context = [ctx])
        else:
            mod = mx.mod.BucketingModule(
                sym_gen = sym_gen, 
                default_bucket_key = test_iter.default_bucket_key, 
                context = [ctx]
            )
        epoch = 2
        sym, arg_params, aux_params = mx.model.load_checkpoint('%s%s' % (params_dir, params_prefix), epoch)
        mod.bind(data_shapes=test_iter.provide_data, label_shapes = test_iter.provide_label)
        mod.set_params(arg_params=arg_params, aux_params=aux_params)
        res  = mod.score(test_iter, PerplexityWithoutExp(ignore_label))
        for name, val in res:
            print 'Test-%s=%f' %  (name, val)

        dec_idx2word = {}
        for k, v in dec_word2idx.items():
            dec_idx2word[v] = k
        enc_idx2word = {}
        for k, v in enc_word2idx.items():
            enc_idx2word[v] = k
        g = open(knowledge+'_generate_epoch_%d.txt' % epoch, 'w')
        
        path = os.path.join(data_dir, test_file)
        with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        dic = defaultdict(str)
        for line in lines:
            line_list = line.strip().split('\t=>\t')
            if len(line_list) != 2 or len(line_list[0].strip()) != len(line_list[1].strip()) :
                continue
            dic[line_list[0].strip()] += line_list[1].strip() + '\t'
        index = 0
        for key in dic:
            enc_string = key
            dec_string = dic[key]
            string_list = enc_string.strip().split()
            enc_len = len(string_list)
            data = []
            for item in string_list:
                if enc_word2idx.get(item) is None:
                    data.append(enc_word2idx.get('<unk>'))
                else:
                    data.append(enc_word2idx.get(item))
            enc_data = mx.nd.array(np.array(data).reshape(1, enc_len))
            g.write('post: ' + enc_string.strip().encode('utf8') + '\n')
            g.write('cmnt: ' + dec_string.strip().encode('utf8') + '\n')
            g.write('beam search results:\n')
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
                share_embed_weight = share_embed_weight,
                is_train = False
            )
            print enc_string
            # ---------------------- print result ------------------
            results = seq2seq.couplet_predict(enc_string, enc_word2idx, enc_word2pos, enc_word2rhyme, arg_params, knowledge=knowledge)

            res = []
            for pair in results:
                sent = pair[1]
                mystr = ""
                for idx in sent:
                    if dec_idx2word[idx] == '<eos>':
                        continue
                    mystr += " " +  dec_idx2word[idx]
                res.append((pair[0], mystr.strip()))
            results = rescore(enc_string, res)
            minmum = min(10, len(results))
            for pair in results[0:minmum]:
                g.write("score : %f, sentence: %s\n" % (pair[0], pair[1].encode('utf8')))
            g.write('==============================================\n')
            index += 1
            if index > 500:
                break
        g.close()
        
        list_of_hypothesis, list_of_references = read_file(knowledge+'_generate_epoch_%d.txt' % epoch)
        '''Use blue0 since there are some  sentences with only one word and the lexicon is built on top of one-word segmentation'''
        blue = corpus_bleu(list_of_references, list_of_hypothesis, 
            weights=(1,),
            smoothing_function=None
        )
        print 'Test-Blue: %f' % blue
