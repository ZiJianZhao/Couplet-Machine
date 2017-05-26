import mxnet as mx
import numpy as np

import re, os, sys, argparse, logging,collections
import codecs
import random

from rnn import RNNLanguage
from sequence_iter import read_dict, get_text_id, SequenceIter, generate_buckets

def init_logging(log_filename = 'Log'):
    logging.basicConfig(
        level    = logging.DEBUG,
        format   = '%(asctime)s %(message)s', #'%(filename)-20s LINE %(lineno)-4d %(levelname)-8s %(asctime)s %(message)s',
        datefmt  = '%m-%d %H:%M:%S',
        filename = log_filename,
        filemode = 'w'
    )
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(message)s');
    # tell the handler to use this format
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

init_logging()

# ----------------- 1. Process the data  ---------------------------------------
data_dir = '/slfs1/users/zjz17/github/data/ptb_data/'
vocab_path = os.path.join(data_dir, 'ptb.vocab.txt')
train_path = os.path.join(data_dir, 'ptb.train.txt')
valid_path = os.path.join(data_dir, 'ptb.valid.txt')
test_path = os.path.join(data_dir, 'ptb.test.txt')

word2idx = read_dict(vocab_path)
ignore_label = word2idx.get(u'<pad>')
print 'ignore label: ', ignore_label
data_train, label_train = get_text_id(train_path, word2idx)
data_valid, label_valid = get_text_id(valid_path, word2idx)

# -----------------2. Params Defination ----------------------------------------
num_buckets = 5
batch_size = 32

# training parameters
params_dir = 'params/'
params_prefix = 'ptb'

# program  parameters, setup seed for random module to reproduce the same result 
seed = 1
random.seed(seed)
np.random.seed(seed)
mx.random.seed(seed)

# ----------------------3. Data Iterator Defination ---------------------
sequence_length = []
for sent in data_train+data_valid:
    sequence_length.append(len(sent))
buckets = generate_buckets(sequence_length, num_buckets)

train_iter = SequenceIter(
	data = data_train, 
	label = label_train, 
    batch_size = batch_size,
    buckets = buckets,
	pad = word2idx.get('<pad>')
)
valid_iter = SequenceIter(
	data = data_valid, 
	label = label_valid, 
    batch_size = batch_size,
    buckets = buckets,
	pad = word2idx.get('<pad>')
)

frequent = train_iter.data_num / batch_size / 10
# ------------------4. Load paramters if exists ------------------------------

model_args = {}
if os.path.isfile('%s/%s-symbol.json' % (params_dir, params_prefix)):
	filelist = os.listdir(params_dir)
	paramfilelist = []
	for f in filelist:
		if f.startswith('%s-' % params_prefix) and f.endswith('.params'):
			paramfilelist.append( int(re.split(r'[-.]', f)[1]) )
	last_iteration = max(paramfilelist)
	print('laoding pretrained model %s/%s at epoch %d' % (params_dir, params_prefix, last_iteration))
	sym, arg_params, aux_params = mx.model.load_checkpoint('%s/%s' % (params_dir, params_prefix), last_iteration)
	model_args.update({
		'arg_params' : arg_params,
		'aux_params' : aux_params,
		'begin_epoch' : last_iteration
	})

# -----------------------5. Training ------------------------------------
def sym_gen(bucketkey):
    rnn_lan = RNNLanguage(
        input_size = len(word2idx), 
        seq_len = bucketkey, 
        ignore_label = ignore_label, 
        is_train = True
    )
    softmax_symbol = rnn_lan.symbol_define()
    data_names = [item[0] for item in train_iter.provide_data]
    label_names = [item[0] for item in train_iter.provide_label] 
    return (softmax_symbol, data_names, label_names)

if num_buckets == 1:
    mod = mx.mod.Module(*sym_gen(train_iter.default_bucket_key), context = [mx.gpu(1)])
else:
    mod = mx.mod.BucketingModule(
        sym_gen = sym_gen, 
        default_bucket_key = train_iter.default_bucket_key, 
        context = [mx.gpu(1)]
    )

if not os.path.exists(params_dir):
    os.makedirs(params_dir)

mod.fit(
    train_data = train_iter, 
    eval_data = valid_iter, 
    num_epoch = 30,
    eval_metric = mx.metric.Perplexity(ignore_label),
    epoch_end_callback = [mx.callback.do_checkpoint('%s%s' % (params_dir, params_prefix), 1)],
    batch_end_callback = [mx.callback.Speedometer(batch_size, frequent = frequent)],
    initializer = mx.init.Uniform(0.05),
    optimizer = 'adam',
    optimizer_params = { 'wd': 0.0000, 'clip_gradient': 0.25},
    **model_args
    #optimizer_params = {'learning_rate':0.01, 'momentum': 0.9, 'wd': 0.0000}
)


