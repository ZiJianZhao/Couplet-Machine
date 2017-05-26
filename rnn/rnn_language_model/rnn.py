import numpy as np
import mxnet as mx
import argparse

class RNNLanguage(object):
    def __init__(self, input_size, seq_len, ignore_label=0, is_train=True):
        super(RNNLanguage, self).__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.ignore_label = ignore_label
        self.is_train = is_train
        
        self.dropout = 0.2
        self.num_layers = 1
        self.num_embed = 300
        self.num_hidden = 300
        self.mode = 'lstm'
        self.bidirectional = False

    def symbol_define(self):
        data = mx.sym.Variable('data')
        label = mx.sym.Variable('label')
        weight = mx.sym.Variable('embed_weight')
        embed = mx.sym.Embedding(
            data = data, 
            input_dim = self.input_size,
            output_dim = self.num_embed, 
            weight = weight,
            name = 'embed'
        )

        rnn_cell = mx.rnn.FusedRNNCell(
            num_hidden = self.num_hidden, 
            num_layers = self.num_layers, 
            mode = self.mode,
            bidirectional = self.bidirectional
        )
        rnn_cell.reset()
        output, _ = rnn_cell.unroll(
            length = self.seq_len, 
            inputs = embed, 
            merge_outputs = True, 
        )

        pred = mx.sym.Reshape(output, shape=(-1, self.num_hidden))
        pred = mx.sym.Dropout(pred, p = self.dropout)
        pred = mx.sym.FullyConnected(
            data = pred, 
            num_hidden = self.input_size, 
            weight = weight,
            name = 'pred'
        )
        label = mx.sym.Reshape(label, shape=(-1,))
        sm = mx.sym.SoftmaxOutput(
            data = pred, 
            label = label, 
            name='softmax',
            use_ignore = True, 
            ignore_label = self.ignore_label
        )
        return sm
