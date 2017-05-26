#coding=utf-8

import sys
import copy
import math
import mxnet as mx
import numpy as np
from collections import defaultdict
import distance

sys.path.append('..')
from rnn.rnn import GRU
from filter import get_repetations, get_chaizi, read_chaizi, get_chaizi_dict

class FocusSeq2Seq(object):
    '''Sequence to sequence learning with neural networks
    The basic sequence to sequence learning network

    Note: you can't use gru as encoder and lstm as decoder
    because so makes the lstm cell has no initilization. 
    '''

    def __init__(self, enc_input_size,enc_pos_size,enc_rhyme_size, dec_input_size, dec_pos_size, dec_rhyme_size, 
        enc_len, dec_len, num_label, share_embed_weight = False, is_train = True, ignore_label = 0):
        super(FocusSeq2Seq, self).__init__()
        # ------------------- Parameter definition -------------------------
        self.enc_input_size = enc_input_size
        self.dec_input_size = dec_input_size
        self.enc_pos_size = enc_pos_size
        self.enc_rhyme_size = enc_rhyme_size
        self.dec_pos_size = dec_pos_size
        self.dec_rhyme_size = dec_rhyme_size
        self.enc_len = enc_len
        self.dec_len = dec_len
        self.num_label = num_label
        self.share_embed_weight = share_embed_weight
        self.is_train = is_train
        self.enc_num_embed = 400
        self.enc_pos_embed = 100
        self.enc_rhyme_embed = 100
        self.enc_num_hidden = 800
        self.enc_name = 'enc'
        self.dec_num_embed = 400
        self.dec_pos_embed = 100
        self.dec_rhyme_embed = 100
        self.dec_num_hidden = 800
        self.dec_name = 'dec'
        self.output_dropout = 0.
        self.ignore_label = ignore_label

        if self.share_embed_weight:  # (for same language task, for example, dialog)
            self.embed_weight = mx.sym.Variable('embed_weight')
            self.enc_embed_weight = self.embed_weight
            self.dec_embed_weight = self.embed_weight
            self.embed_pos_weight = mx.sym.Variable('embed_pos_weight')
            self.enc_pos_weight = self.embed_pos_weight
            self.dec_pos_weight = self.embed_pos_weight  
            self.embed_rhyme_weight = mx.sym.Variable('embed_rhyme_weight')
            self.enc_rhyme_weight = self.embed_rhyme_weight
            self.dec_rhyme_weight = self.embed_rhyme_weight              
        else:  # (for multi languages task, for example, translation)
            self.enc_embed_weight = mx.sym.Variable('%s_embed_weight' % self.enc_name)
            self.dec_embed_weight = mx.sym.Variable('%s_embed_weight' % self.dec_name)
            self.enc_pos_weight = mx.sym.Variable('%s_embed_pos_weight' % self.enc_name)
            self.dec_pos_weight = mx.sym.Variable('%s_embed_pos_weight' % self.dec_name)
            self.enc_rhyme_weight = mx.sym.Variable('%s_embed_rhyme_weight' % self.enc_name)
            self.dec_rhyme_weight = mx.sym.Variable('%s_embed_rhyme_weight' % self.dec_name)
    
    def symbol_define(self, knowledge='view'):
        self.knowledge = knowledge
        enc_data = mx.sym.Variable('%s_data' % self.enc_name)
        enc_data_embed = mx.sym.Embedding(
            data = enc_data, 
            input_dim = self.enc_input_size, 
            weight = self.enc_embed_weight, 
            output_dim = self.enc_num_embed, 
            name = '%s_embed' % self.enc_name
        )
        if self.is_train:
            enc_mask = mx.sym.Variable('%s_mask' % self.enc_name)
        else:
            enc_mask = None
        if self.knowledge == 'view':
            enc_pos = mx.sym.Variable('%s_pos' % self.enc_name)
            enc_rhyme = mx.sym.Variable('%s_rhyme' % self.enc_name)
        if self.knowledge == 'view':
            enc_pos_embed = mx.sym.Embedding(
                data = enc_pos, 
                input_dim = self.enc_pos_size, 
                weight = self.enc_pos_weight, 
                output_dim = self.enc_pos_embed, 
                name = '%s_pos_embed' % self.enc_name
            )
            enc_rhyme_embed = mx.sym.Embedding(
                data = enc_rhyme, 
                input_dim = self.enc_rhyme_size, 
                weight = self.enc_rhyme_weight, 
                output_dim = self.enc_rhyme_embed, 
                name = '%s_rhyme_embed' % self.enc_name
            )
            enc_embed = mx.sym.Concat(*[enc_data_embed, enc_pos_embed, enc_rhyme_embed], dim = 2) 
        elif self.knowledge == 'label':
            enc_embed = enc_data_embed
        else:           
            raise NameError, 'No knowledge type named %s' % self.knowledge
        forward_gru = GRU(num_hidden = self.enc_num_hidden, name = '%s_forward' % self.enc_name)
        forward_enc_output, [forward_enc_last_h] = forward_gru.unroll(
            data = enc_embed, 
            seq_len = self.enc_len, 
            mask = enc_mask, 
            forward = True,
            merge_outputs = True
        )
        backwrd_gru = GRU(num_hidden = self.enc_num_hidden, name = '%s_backward' % self.enc_name)
        backward_enc_output, [backward_enc_last_h] = backwrd_gru.unroll(
            data = enc_embed, 
            seq_len = self.enc_len, 
            mask = enc_mask, 
            forward = False,
            merge_outputs = True
        )
        zero_for_last_eos = mx.sym.zeros(shape = (0, 1, 2*self.enc_num_hidden), name = 'zeros_for_last_eos')
        enc_output = mx.sym.Concat(*[forward_enc_output, backward_enc_output], dim = 2)
        if self.knowledge == 'label':
            pred = mx.sym.Reshape(enc_output, shape=(-1, 2 * self.enc_num_hidden))
            pos_pred = mx.sym.FullyConnected(
                data = pred, 
                num_hidden = self.enc_pos_size,
                name = '%s_pos_pred' % self.enc_name
            )
            enc_pos = mx.sym.Variable('%s_pos' % self.enc_name)
            pos_label = mx.sym.Reshape(data = enc_pos, shape = (-1, ))
            enc_pos_sm = mx.sym.SoftmaxOutput(
                data = pos_pred, 
                label = pos_label, 
                name = 'enc_pos_softmax',
                use_ignore = True, 
                ignore_label = 0
            )
            rhyme_pred = mx.sym.FullyConnected(
                data = pred, 
                num_hidden = self.enc_rhyme_size,
                name = '%s_rhyme_pred' % self.enc_name
            )
            enc_rhyme = mx.sym.Variable('%s_rhyme' % self.enc_name)
            rhyme_label = mx.sym.Reshape(data = enc_rhyme, shape = (-1, ))
            enc_rhyme_sm = mx.sym.SoftmaxOutput(
                data = rhyme_pred, 
                label = rhyme_label, 
                name = 'enc_rhyme_softmax',
                use_ignore = True, 
                ignore_label = 0
            )
        enc_output = mx.sym.Concat(*[enc_output, zero_for_last_eos], dim = 1)

        dec_trans_h_temp = mx.sym.FullyConnected(
            data = forward_enc_last_h, 
            num_hidden = self.dec_num_hidden,
            name = 'encode_to_decode_transform_weight'
        )
        dec_trans_h = mx.sym.Activation(dec_trans_h_temp, act_type = "tanh")
        dec_data = mx.sym.Variable('%s_data' % self.dec_name)
        if self.is_train:
            dec_mask = mx.sym.Variable('%s_mask' % self.dec_name)
        else:
            dec_mask = None
        dec_data_embed = mx.sym.Embedding(
            data = dec_data, 
            input_dim = self.dec_input_size, 
            weight = self.dec_embed_weight, 
            output_dim = self.dec_num_embed, 
            name = '%s_embed' % self.dec_name
        )
        if self.knowledge == 'view':
            dec_pos = mx.sym.Variable('%s_pos' % self.dec_name)
            dec_rhyme = mx.sym.Variable('%s_rhyme' % self.dec_name)
        if self.knowledge == 'view':
            dec_pos_embed = mx.sym.Embedding(
                data = dec_pos, 
                input_dim = self.dec_pos_size, 
                weight = self.dec_pos_weight, 
                output_dim = self.dec_pos_embed, 
                name = '%s_pos_embed' % self.dec_name
            )
            dec_rhyme_embed = mx.sym.Embedding(
                data = dec_rhyme, 
                input_dim = self.dec_rhyme_size, 
                weight = self.dec_rhyme_weight, 
                output_dim = self.dec_rhyme_embed, 
                name = '%s_rhyme_embed' % self.dec_name
            )
            dec_embed = mx.sym.Concat(*[dec_data_embed, dec_pos_embed, dec_rhyme_embed], dim = 2)
        elif self.knowledge == 'label':
            dec_embed = dec_data_embed
        else:           
            raise NameError, 'No knowledge type named %s' % self.knowledge
        if self.is_train:
            dec_embed_with_context = mx.sym.Concat(*[enc_output, dec_embed], dim = 2)
        else:
            enc_hidden = mx.sym.Variable('enc_hidden')
            dec_embed_with_context = mx.sym.Concat(*[enc_hidden, dec_embed], dim = 2)

        gru = GRU(num_hidden = self.dec_num_hidden, name = self.dec_name)
        if not self.is_train:
            dec_init_h = mx.sym.Variable('%s_l0_init_h' % self.dec_name)
        else:
            dec_init_h = dec_trans_h
        dec_output, [dec_last_h] = gru.unroll(
            data = dec_embed_with_context, 
            seq_len = self.dec_len, 
            mask = dec_mask,
            begin_state = [dec_init_h],
            merge_outputs = True
        )
        hidden_concat = mx.sym.Reshape(dec_output, shape=(-1, self.dec_num_hidden))
        #hidden_concat = mx.sym.Dropout(data = hidden_concat, p = self.output_dropout)
        pred = mx.sym.FullyConnected(
            data = hidden_concat, 
            num_hidden = self.num_label,
            name = '%s_pred' % self.dec_name
        )
        label = mx.sym.Variable('label')
        label = mx.sym.Reshape(data = label, shape = (-1, ))

        if self.knowledge == 'label':
            dec_pos_pred = mx.sym.Reshape(hidden_concat, shape=(-1, self.dec_num_hidden))
            dec_pos_pred = mx.sym.FullyConnected(
                data = dec_pos_pred, 
                num_hidden = self.dec_pos_size,
                name = '%s_pos_pred' % self.dec_name
            )
            dec_pos = mx.sym.Variable('%s_pos' % self.dec_name)
            dec_pos_label = mx.sym.Reshape(data = dec_pos, shape = (-1, ))
            dec_pos_sm = mx.sym.SoftmaxOutput(
                data = dec_pos_pred, 
                label = dec_pos_label, 
                name = 'dec_pos_softmax',
                use_ignore = True, 
                ignore_label = 0
            )

            dec_rhyme_pred = mx.sym.Reshape(hidden_concat, shape=(-1, self.dec_num_hidden))
            dec_rhyme_pred = mx.sym.FullyConnected(
                data = dec_rhyme_pred, 
                num_hidden = self.dec_rhyme_size,
                name = '%s_rhyme_pred' % self.dec_name
            )
            dec_rhyme = mx.sym.Variable('%s_rhyme' % self.dec_name)
            dec_rhyme_label = mx.sym.Reshape(data = dec_rhyme, shape = (-1, ))
            dec_rhyme_sm = mx.sym.SoftmaxOutput(
                data = dec_rhyme_pred, 
                label = dec_rhyme_label, 
                name = 'dec_rhyme_softmax',
                use_ignore = True, 
                ignore_label = 0
            )
        if self.is_train:
            sm = mx.sym.SoftmaxOutput(
                data = pred, 
                label = label, 
                name = 'softmax',
                use_ignore = True, 
                ignore_label = self.ignore_label
            )
            if self.knowledge == 'label':
                return mx.sym.Group([sm, enc_pos_sm, dec_pos_sm, enc_rhyme_sm, dec_rhyme_sm])
            else:
                return sm
        else:
            sm = mx.sym.SoftmaxOutput(data = pred, name = 'softmax')      
            return mx.sym.Group([dec_trans_h, enc_output]), mx.sym.Group([sm, dec_last_h])


    def couplet_knowledge_predict(self, enc_string, word2idx, word2pos, word2rhyme, arg_params, 
        knowledge ='view', pad = 0, eos = 1, unk = 2):
        # ------------------------- bind data to symbol ---------------------------
        idx2word = {}
        for key in word2idx:
            idx2word[word2idx[key]] = key
        string_list = enc_string.strip().split()
        enc_len = len(string_list)
        data = []
        for item in string_list:
            if word2idx.get(item) is None:
                data.append(word2idx.get('<unk>'))
            else:
                data.append(word2idx.get(item))
        enc_list = data
        enc_data = mx.nd.array(np.array(data).reshape(1, enc_len))

        pos_data = []
        for item in string_list:
            if word2pos.get(item) is None:
                pos_data.append(word2pos.get('<unk>'))
            else:
                pos_data.append(word2pos.get(item))
        rhyme_data = []
        for item in string_list:
            if word2rhyme.get(item) is None:
                rhyme_data.append(word2rhyme.get('<unk>'))
            else:
                rhyme_data.append(word2rhyme.get(item))            

        enc_pos = mx.nd.array(np.array(pos_data).reshape(1, enc_len))
        enc_rhyme = mx.nd.array(np.array(rhyme_data).reshape(1, enc_len))
        # =================== knowledge beg =================================
        
        punctuations = [word2idx.get(u'，'), word2idx.get(u'。')]
        enc_dict = {w:1 for w in enc_list if w not in punctuations}
        enc_repetations = get_repetations(enc_list)
        chaizi_dic  = read_chaizi(mode=0) 
        enc_chaizi = get_chaizi(string_list, chaizi_dic)
        chaizi_dic  = read_chaizi(mode=1)
        he2chai_c, chai2he_c, he2chai_n, chai2he_n = get_chaizi_dict(chaizi_dic, word2idx)

        print 'repetations: ', enc_repetations
        print 'chaizi: ', enc_chaizi
        # =================== knowledge end =================================

        encoder, decoder = self.symbol_define(knowledge=knowledge)

        input_shapes = {}
        input_shapes['enc_data'] = (1, self.enc_len)
        if knowledge == 'view':
            input_shapes['enc_pos'] = (1, enc_len)
            input_shapes['enc_rhyme'] = (1, enc_len)
        encoder_executor = encoder.simple_bind(ctx = mx.cpu(), **input_shapes)
        for key in encoder_executor.arg_dict:
            if key in arg_params:
                arg_params[key].copyto(encoder_executor.arg_dict[key])
        enc_data.copyto(encoder_executor.arg_dict['enc_data'])
        if knowledge == 'view':
            enc_pos.copyto(encoder_executor.arg_dict['enc_pos'])
            enc_rhyme.copyto(encoder_executor.arg_dict['enc_rhyme'])
        encoder_executor.forward()
        enc_hidden = encoder_executor.outputs[1]
        dec_init_states = [('%s_l0_init_h' % self.dec_name, (1, self.dec_num_hidden))]
        state_name = [item[0] for item in dec_init_states]
        init_states_dict = dict(zip(state_name, [encoder_executor.outputs[0]]))
        if knowledge == 'view':
            dec_data_shape = [("dec_data", (1,1)), ("dec_pos", (1,1)), ("dec_rhyme", (1,1))]
        else:
            dec_data_shape = [("dec_data", (1,1))]
        enc_hidden_shape = [('enc_hidden', (1, 1, self.enc_num_hidden * 2))]
        dec_input_shapes = dict(dec_data_shape + dec_init_states + enc_hidden_shape)
        decoder_executor = decoder.simple_bind(ctx = mx.cpu(), **dec_input_shapes)
        for key in decoder_executor.arg_dict:
            if key in arg_params:
                arg_params[key].copyto(decoder_executor.arg_dict[key])
        # --------------------------- beam search ---------------------------------
        beam = 20
        active_sentences = [(0,[eos], copy.deepcopy(init_states_dict))]
        max_length = self.enc_len
        min_count = min(beam, len(active_sentences))
        for seqidx in xrange(max_length):
            tmp_sentences = []
            
            for i in xrange(min_count):
                states_dict  = active_sentences[i][2]
                for key in states_dict.keys():
                    states_dict[key].copyto(decoder_executor.arg_dict[key])
                decoder_executor.arg_dict["dec_data"][:] = active_sentences[i][1][-1]
                temp_pos_idx = word2pos.get(idx2word.get(active_sentences[i][1][-1]))
                temp_rhyme_idx = word2rhyme.get(idx2word.get(active_sentences[i][1][-1]))
                if knowledge == 'view':
                    decoder_executor.arg_dict["dec_pos"][:] = temp_pos_idx
                    decoder_executor.arg_dict["dec_rhyme"][:] = temp_rhyme_idx
                decoder_executor.arg_dict['enc_hidden'][:] = enc_hidden.asnumpy()[0, seqidx, :].reshape(1, 1, -1)
                decoder_executor.forward()
                new_states_dict = dict(zip(state_name, decoder_executor.outputs[1:]))
                tmp_states_dict = copy.deepcopy(new_states_dict)

                prob = decoder_executor.outputs[0].asnumpy()
                # === this order is from small to big =====
                indecies = np.argsort(prob, axis = 1)[0]
                diversity_score = 0
                
                # =================== knowledge beg ===========================
                # === process the punctuations ===
                if enc_list[seqidx] in punctuations:
                    enc_idx = int(enc_list[seqidx])
                    score = active_sentences[i][0] + math.log(prob[0][enc_idx])
                    sent = active_sentences[i][1][:]
                    sent.extend([enc_idx])
                    tmp_sentences.append((score, sent, tmp_states_dict))
                    continue 
                # === process the chaizi couplet ===
                if enc_chaizi[seqidx] is not None:
                    if enc_chaizi[seqidx][0] == 'B':
                        if enc_chaizi[seqidx][-1] == 'CC':
                            tmp_dic = chai2he_c
                        elif enc_chaizi[seqidx][-1] == 'HC':
                            tmp_dic = he2chai_c
                        elif enc_chaizi[seqidx][-1] == 'CN':
                            tmp_dic = chai2he_n
                        else:
                            tmp_dic = he2chai_n
                        for key in tmp_dic:
                            if key in enc_dict:
                                continue
                            enc_idx = key
                            score = active_sentences[i][0] + math.log(prob[0][enc_idx])
                            sent = active_sentences[i][1][:]
                            sent.extend([enc_idx])
                            tmp_sentences.append((score, sent, tmp_states_dict))
                    else: 
                        begin_index = enc_chaizi[seqidx][1]
                        mid_index = enc_chaizi[seqidx][2]
                        if enc_chaizi[begin_index][-1] == 'CC':
                            tmp_dic = chai2he_c
                        elif enc_chaizi[begin_index][-1] == 'HC':
                            tmp_dic = he2chai_c
                        elif enc_chaizi[begin_index][-1] == 'CN':
                            tmp_dic = chai2he_n
                        else:
                            tmp_dic = he2chai_n             
                        if enc_chaizi[seqidx][0] == 'I':
                            tmp_idx = 0
                        else:
                            tmp_idx = 1
                        index = active_sentences[i][1][begin_index+1]
                        for l_i in range(len(tmp_dic[index])):
                            enc_idx = tmp_dic[index][l_i][tmp_idx]
                            if tmp_idx == 1:
                                if active_sentences[i][1][mid_index+1] != tmp_dic[index][l_i][0]:
                                    continue
                            if enc_idx not in enc_dict:
                                score = active_sentences[i][0] + math.log(prob[0][enc_idx])
                                sent = active_sentences[i][1][:]
                                sent.extend([enc_idx])
                                tmp_sentences.append((score, sent, tmp_states_dict))
                        #for item in tmp_sentences:
                            #print tmp_idx, ''.join([idx2word[idx] for idx in item[1][1:]])
                        #print '==========================================='
                    continue
                if enc_repetations[seqidx] < seqidx:
                    enc_idx = active_sentences[i][1][enc_repetations[seqidx]+1]
                    score = active_sentences[i][0] + math.log(prob[0][enc_idx])
                    sent = active_sentences[i][1][:]
                    sent.extend([enc_idx])
                    tmp_sentences.append((score, sent, tmp_states_dict))
                    continue
                # =================== knowledge end ===========================
                
                j = 0
                num = 0
                while num <= beam and j <= 30:
                    score_idx = indecies[-j-1]
                    score = active_sentences[i][0] + math.log(prob[0][indecies[-j-1]])
                    sent = active_sentences[i][1][:]
                    sent.extend([indecies[-j-1]])
                    j += 1
                    if sent[-1] != eos and sent[-1] != unk and sent[-1] != pad and sent[-1] not in enc_dict:
                        repetations = get_repetations(sent[1:])
                        if repetations == enc_repetations[:seqidx+1]:
                            tmp_sentences.append((score, sent, tmp_states_dict))
                            num += 1

            # =================== ensure lines repeatation form is same ===========================
            '''
            new_sentences = []
            repetation_dict = {}
            for item in tmp_sentences:
                sent = ' '.join([str(it) for it in item[1]])
                if repetation_dict.get(sent) is None:
                    repetation_dict[sent] = 1 
                else:
                    continue
                repetations = get_repetations(item[1][1:])
                if repetations == enc_repetations[:seqidx+1]:
                    new_sentences.append(item)
            tmp_sentences = new_sentences
            '''
            # =================== first word is different ===========================
            '''
            new_sentences = []
            repetation_dict = {}
            for item in tmp_sentences:
                word = item[1][1]
                if repetation_dict.get(word) is None:
                    repetation_dict[word] = 1 
                else:
                    continue
                new_sentences.append(item)
            tmp_sentences = new_sentences
            '''
            # =================== choose top beam candidates =======================
            #min_count = min(beam, len(tmp_sentences))
            #active_sentences = sorted(tmp_sentences, reverse = True)[:min_count]
            # =================== choose diverse top beam candidates =======================
            min_count = min(beam, len(tmp_sentences))
            sorted_sentences = sorted(tmp_sentences, reverse = True)
            active_sentences = []
            for sentence in sorted_sentences:
                if len(active_sentences) >= min_count:
                    break
                flag = True
                for sent in active_sentences:
                    if distance.hamming(sentence[1], sent[1]) < seqidx:
                         flag = False
                         break
                if flag:
                    active_sentences.append(sentence)

            if len(active_sentences) < min_count:
                repeat_dic = {}
                for sent in active_sentences:
                    string = ' '.join([str(it) for it in sent[1]])
                    repeat_dic[string] = 1 
                for sent in sorted_sentences:
                    if len(active_sentences) >= min_count:
                        break
                    string = ' '.join([str(it) for it in sent[1]])
                    if repeat_dic.get(string) is None:
                        active_sentences.append(sent)
                        repeat_dic[string] = 1
            
        result_sentences = []
        for sent in active_sentences:
            result_sentences.append((sent[0], sent[1][1:]))
        #result = min(beam, len(result_sentences), 10)
        #result_sentences = sorted(result_sentences, reverse = True)[:result]
        result_sentences = sorted(result_sentences, reverse = True)
        return result_sentences