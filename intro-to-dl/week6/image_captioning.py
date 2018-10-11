import sys
sys.path.append("..")
#import grading
import download_utils

download_utils.link_all_keras_resources()

import tensorflow as tf
from tensorflow.contrib import keras
import numpy as np
import matplotlib.pyplot as plt
L = keras.layers
K = keras.backend
import utils
import time
import zipfile
import json
from collections import defaultdict
import re
import random
from random import choice
#import grading_utils
import os
from keras_utils import reset_tf_session
#import tqdm_utils

# Download data
# train images http://msvocds.blob.core.windows.net/coco2014/train2014.zip
# validation images http://msvocds.blob.core.windows.net/coco2014/val2014.zip
# captions for both train and validation http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip
#
# captions_train-val2014.zip
# train2014_sample.zip
# train_img_embeds.pickle
# train_img_fns.pickle
# val2014_sample.zip
# val_img_embeds.pickle
# val_img_fns.pickle

# we downloaded them for you, just link them here
download_utils.link_week_6_resources()

# Extract image features
# We will use pre-trained InceptionV3 model for CNN encoder
# (https://research.googleblog.com/2016/03/train-your-own-image-classifier-with.html)
# and extract its last hidden layer as an embedding.

IMG_SIZE = 299

# we take the last hidden layer of IncetionV3 as an image embedding
def get_cnn_encoder():
    K.set_learning_phase(False)
    model = keras.applications.InceptionV3(include_top=False)
    preprocess_for_model = keras.applications.inception_v3.preprocess_input

    model = keras.models.Model(model.inputs, keras.layers.GlobalAveragePooling2D()(model.output))
    return model, preprocess_for_model

# Features extraction takes too much time on CPU:
# 
# Takes 16 minutes on GPU.
# 25x slower (InceptionV3) on CPU and takes 7 hours.
# 10x slower (MobileNet) on CPU and takes 3 hours.
# So we've done it for you with the following code:
# 
# # load pre-trained model
# reset_tf_session()
# encoder, preprocess_for_model = get_cnn_encoder()
# 
# # extract train features
# train_img_embeds, train_img_fns = utils.apply_model(
#     "train2014.zip", encoder, preprocess_for_model, input_shape=(IMG_SIZE, IMG_SIZE))
# utils.save_pickle(train_img_embeds, "train_img_embeds.pickle")
# utils.save_pickle(train_img_fns, "train_img_fns.pickle")
# 
# # extract validation features
# val_img_embeds, val_img_fns = utils.apply_model(
#     "val2014.zip", encoder, preprocess_for_model, input_shape=(IMG_SIZE, IMG_SIZE))
# utils.save_pickle(val_img_embeds, "val_img_embeds.pickle")
# utils.save_pickle(val_img_fns, "val_img_fns.pickle")
# 
# # sample images for learners
# def sample_zip(fn_in, fn_out, rate=0.01, seed=42):
#     np.random.seed(seed)
#     with zipfile.ZipFile(fn_in) as fin, zipfile.ZipFile(fn_out, "w") as fout:
#         sampled = filter(lambda _: np.random.rand() < rate, fin.filelist)
#         for zInfo in sampled:
#             fout.writestr(zInfo, fin.read(zInfo))
# 
# sample_zip("train2014.zip", "train2014_sample.zip")
# sample_zip("val2014.zip", "val2014_sample.zip")

# load prepared embeddings
train_img_embeds = utils.read_pickle("train_img_embeds.pickle")
train_img_fns = utils.read_pickle("train_img_fns.pickle")
val_img_embeds = utils.read_pickle("val_img_embeds.pickle")
val_img_fns = utils.read_pickle("val_img_fns.pickle")
# check shapes
print("train_img_embeds.shape = ", train_img_embeds.shape,
      "len(train_img_fns) = ", len(train_img_fns))  # (82783, 2048)  82783
print("val_img_embeds.shape = ", val_img_embeds.shape,
      "len(val_img_fns)= ", len(val_img_fns))  # (40504, 2048)  40504

print("train_img_embeds[:2] = ", train_img_embeds[:2])
print("val_img_fns[:2] = ", val_img_fns[:2])

# check prepared samples of images
list(filter(lambda x: x.endswith("_sample.zip"), os.listdir(".")))


# Extract captions for images

# extract captions from zip
def get_captions_for_fns(fns, zip_fn, zip_json_path):
    zf = zipfile.ZipFile(zip_fn)
    j = json.loads(zf.read(zip_json_path).decode("utf8"))
    id_to_fn = {img["id"]: img["file_name"] for img in j["images"]}
    fn_to_caps = defaultdict(list)
    for cap in j['annotations']:
        fn_to_caps[id_to_fn[cap['image_id']]].append(cap['caption'])
    fn_to_caps = dict(fn_to_caps)
    return list(map(lambda x: fn_to_caps[x], fns))


train_captions = get_captions_for_fns(train_img_fns, "captions_train-val2014.zip",
                                      "annotations/captions_train2014.json")

val_captions = get_captions_for_fns(val_img_fns, "captions_train-val2014.zip",
                                    "annotations/captions_val2014.json")

# check shape
print("len(train_captions) = ", len(train_captions))
print("len(val_captions) = ", len(val_captions))

# preview captions data
print("train_captions[:2] : ", train_captions[:2])

# look at training example (each has 5 captions)
def show_trainig_example(train_img_fns, train_captions, example_idx=0):
    """
    You can change example_idx and see different images
    """
    zf = zipfile.ZipFile("train2014_sample.zip")
    captions_by_file = dict(zip(train_img_fns, train_captions))
    all_files = set(train_img_fns)
    found_files = list(filter(lambda x: x.filename.rsplit("/")[-1] in all_files, zf.filelist))
    example = found_files[example_idx]
    img = utils.decode_image_from_buf(zf.read(example))
    plt.imshow(utils.image_center_crop(img))
    plt.title("\n".join(captions_by_file[example.filename.rsplit("/")[-1]]))
    plt.show()


show_trainig_example(train_img_fns, train_captions, example_idx=142)
#show_trainig_example(train_img_fns, train_captions, example_idx=100)
#show_trainig_example(train_img_fns, train_captions, example_idx=50)
#show_trainig_example(train_img_fns, train_captions, example_idx=250)


# Prepare captions for training

# special tokens
PAD = "#PAD#"
UNK = "#UNK#"
START = "#START#"
END = "#END#"


# split sentence into tokens (split into lowercased words)
def split_sentence(sentence):
    return list(filter(lambda x: len(x) > 0, re.split('\W+', sentence.lower())))

#print("split_sentence('orange cat on the roof') = ", split_sentence('orange cat on the roof'));


def generate_vocabulary(train_captions):
    """
    Return {token: index} for all train tokens (words) that occur 5 times or more,
        `index` should be from 0 to N, where N is a number of unique tokens in the resulting dictionary.
    Use `split_sentence` function to split sentence into tokens.
    Also, add PAD (for batch padding), UNK (unknown, out of vocabulary),
        START (start of sentence) and END (end of sentence) tokens into the vocabulary.
    """
    vocab = {PAD:5, UNK:5, START:5, END:5}
    for sentences in train_captions:
        for sentence in sentences:
            words = split_sentence(sentence)
            for w in words:
                if w in vocab:
                    vocab[w] += 1
                else:
                    vocab[w] = 1
    vocab = {k:v for k,v in vocab.items() if v >= 5}
    return {token: index for index, token in enumerate(sorted(vocab))}


#voc = generate_vocabulary(train_captions)
#print(voc)


def caption_tokens_to_indices(captions, vocab):
    """
    `captions` argument is an array of arrays:
    [
        [
            "image1 caption1",
            "image1 caption2",
            ...
        ],
        [
            "image2 caption1",
            "image2 caption2",
            ...
        ],
        ...
    ]
    Use `split_sentence` function to split sentence into tokens.
    Replace all tokens with vocabulary indices, use UNK for unknown words (out of vocabulary).
    Add START and END tokens to start and end of each sentence respectively.
    For the example above you should produce the following:
    [
        [
            [vocab[START], vocab["image1"], vocab["caption1"], vocab[END]],
            [vocab[START], vocab["image1"], vocab["caption2"], vocab[END]],
            ...
        ],
        ...
    ]
    """
    res = []  ### YOUR CODE HERE ###
    for l in captions:
        ll = []
        for s in l:
            words = split_sentence(s)
            lll = [vocab[START]]
            for w in words:
                if w in vocab:
                    lll.append(vocab[w])
                else:
                    lll.append(vocab[UNK])
            lll.append(vocab[END])
            ll.append(lll)
        res.append(ll)
    return res


#cid = caption_tokens_to_indices(train_captions, voc)
#print(cid[:5])


# prepare vocabulary
vocab = generate_vocabulary(train_captions)
vocab_inverse = {idx: w for w, idx in vocab.items()}
print("len(vocab) = ", len(vocab))


# replace tokens with indices
train_captions_indexed = caption_tokens_to_indices(train_captions, vocab)
val_captions_indexed = caption_tokens_to_indices(val_captions, vocab)


# Captions have different length, but we need to batch them, that's why we will add PAD tokens so that all sentences
# have an equal length.
# We will crunch LSTM through all the tokens, but we will ignore padding tokens during loss calculation.


# we will use this during training
def batch_captions_to_matrix(batch_captions, pad_idx, max_len=None):
    """
    `batch_captions` is an array of arrays:
    [
        [vocab[START], ..., vocab[END]],
        [vocab[START], ..., vocab[END]],
        ...
    ]
    Put vocabulary indexed captions into np.array of shape (len(batch_captions), columns),
        where "columns" is max(map(len, batch_captions)) when max_len is None
        and "columns" = min(max_len, max(map(len, batch_captions))) otherwise.
    Add padding with pad_idx where necessary.
    Input example: [[1, 2, 3], [4, 5]]
    Output example: np.array([[1, 2, 3], [4, 5, pad_idx]]) if max_len=None
    Output example: np.array([[1, 2], [4, 5]]) if max_len=2
    Output example: np.array([[1, 2, 3], [4, 5, pad_idx]]) if max_len=100
    Try to use numpy, we need this function to be fast!
    """
    if max_len is None:
        ncolumns = max(map(len, batch_captions))
    else:
        ncolumns = min(max_len, max(map(len, batch_captions)))
    matrix = np.full(shape=(len(batch_captions), ncolumns), fill_value=pad_idx) ###YOUR CODE HERE###
    for i in range(len(batch_captions)):
        l = batch_captions[i]
        row =  np.array(l)
        if len(row) > ncolumns:
            matrix[i] = row[:ncolumns]
        else:
            matrix[i,0:len(row)] = row  # padding is automatic
    return matrix


# TESTS BEGIN

def test_vocab(vocab, PAD, UNK, START, END):
    return [
        len(vocab),
        len(np.unique(list(vocab.values()))),
        int(all([_ in vocab for _ in [PAD, UNK, START, END]]))
    ]

#test_vocab(vocab, PAD, UNK, START, END)


def test_captions_indexing(train_captions_indexed, vocab, UNK):
    starts = set()
    ends = set()
    between = set()
    unk_count = 0
    for caps in train_captions_indexed:
        for cap in caps:
            starts.add(cap[0])
            between.update(cap[1:-1])
            ends.add(cap[-1])
            for w in cap:
                if w == vocab[UNK]:
                    unk_count += 1
    return [
        len(starts),
        len(ends),
        len(between),
        len(between | starts | ends),
        int(all([isinstance(x, int) for x in (between | starts | ends)])),
        unk_count
    ]

#test_captions_indexing(train_captions_indexed, vocab, UNK)


def test_captions_batching(batch_captions_to_matrix):
    return (batch_captions_to_matrix([[1, 2, 3], [4, 5]], -1, max_len=None).ravel().tolist()
            + batch_captions_to_matrix([[1, 2, 3], [4, 5]], -1, max_len=2).ravel().tolist()
            + batch_captions_to_matrix([[1, 2, 3], [4, 5]], -1, max_len=10).ravel().tolist())

#test_captions_batching(batch_captions_to_matrix)

# TESTS END


# make sure you use correct argument in caption_tokens_to_indices
assert len(caption_tokens_to_indices(train_captions[:10], vocab)) == 10
assert len(caption_tokens_to_indices(train_captions[:5], vocab)) == 5

# Define architecture
# Since our problem is to generate image captions, RNN text generator should be
# conditioned on image. The idea is to use image features as an initial state for RNN instead of zeros.
#
# Remember that you should transform image feature vector to RNN hidden state size by fully-connected
# layer and then pass it to RNN.
#
#During training we will feed ground truth tokens into the lstm to get predictions of next tokens.
#
#Notice that we don't need to feed last token (END) as input (http://cs.stanford.edu/people/karpathy/)

IMG_EMBED_SIZE = train_img_embeds.shape[1]
print("IMG_EMBED_SIZE = ", IMG_EMBED_SIZE)  # 2048
IMG_EMBED_BOTTLENECK = 120
WORD_EMBED_SIZE = 100
LSTM_UNITS = 300
LOGIT_BOTTLENECK = 120
pad_idx = vocab[PAD]

# remember to reset your graph if you want to start building it from scratch!
s = reset_tf_session()
tf.set_random_seed(42)

# Here we define decoder graph.
#
# We use Keras layers where possible because we can use them in functional style with weights reuse like this:
#
# dense_layer = L.Dense(42, input_shape=(None, 100) activation='relu')
# a = tf.placeholder('float32', [None, 100])
# b = tf.placeholder('float32', [None, 100])
# dense_layer(a)  # that's how we applied dense layer!
# dense_layer(b)  # and again

class decoder:
    # [batch_size, IMG_EMBED_SIZE] of CNN image features
    img_embeds = tf.placeholder('float32', [None, IMG_EMBED_SIZE])
    # [batch_size, time steps] of word ids
    sentences = tf.placeholder('int32', [None, None])

    # we use bottleneck here to reduce the number of parameters
    # image embedding -> bottleneck
    img_embed_to_bottleneck = L.Dense(IMG_EMBED_BOTTLENECK,
                                      input_shape=(None, IMG_EMBED_SIZE),
                                      activation='elu')
    # image embedding bottleneck -> lstm initial state
    img_embed_bottleneck_to_h0 = L.Dense(LSTM_UNITS,
                                         input_shape=(None, IMG_EMBED_BOTTLENECK),
                                         activation='elu')
    # word -> embedding
    # "The one-hot encoded words are mapped to the word vectors. If a multilayer
    #  Perceptron model is used, then the word vectors are concatenated before
    #  being fed as input to the model. If a recurrent neural network is used,
    #  then each word may be taken as one input in a sequence."
    # vocab is used to create one-hot encoding
    # WORD_EMBED_SIZE - output dimension
    # "Input: 2D[batch_size, seq_len], output 3D[batch_size, seq_len=?time_steps, output_dim]"
    word_embed = L.Embedding(len(vocab), WORD_EMBED_SIZE)

    # lstm cell (from tensorflow)
    lstm = tf.nn.rnn_cell.LSTMCell(LSTM_UNITS)

    # initial lstm cell state of shape (None, LSTM_UNITS),
    # we need to condition it on `img_embeds` placeholder.
    c0 = h0 = img_embed_bottleneck_to_h0( img_embed_to_bottleneck( img_embeds ) ) ### YOUR CODE HERE ###

    # embed all tokens but the last for lstm input,
    # remember that L.Embedding is callable,
    # use `sentences` placeholder as input.
    # the last is NOT always END (may be PAD)...??? (AK)
    # shape is [batch_size=None, time steps, LSTM_UNITS]? (AK)
    word_embeds = word_embed(sentences[:,:-1]) ### YOUR CODE HERE ###

    # during training we use ground truth tokens `word_embeds` as context for next token prediction.
    # that means that we know all the inputs for our lstm and can get
    # all the hidden states with one tensorflow operation (tf.nn.dynamic_rnn).
    # `hidden_states` has a shape of [batch_size, time steps, LSTM_UNITS].
    hidden_states, _ = tf.nn.dynamic_rnn(lstm, word_embeds,
                                         initial_state=tf.nn.rnn_cell.LSTMStateTuple(c0, h0))

    # now we need to calculate token logits for all the hidden states

    # first, we reshape `hidden_states` to [-1, LSTM_UNITS]
    # this is for one time step (AK)
    flat_hidden_states = tf.reshape(hidden_states, [-1, LSTM_UNITS])  ### YOUR CODE HERE ###

    # we use bottleneck here to reduce model complexity
    # lstm output -> logits bottleneck
    token_logits_bottleneck = L.Dense(LOGIT_BOTTLENECK,
                                      input_shape=(None, LSTM_UNITS),
                                      activation="elu")
    # logits bottleneck -> logits for next token prediction
    token_logits = L.Dense(len(vocab),
                           input_shape=(None, LOGIT_BOTTLENECK))

    # then, we calculate logits for next tokens using `token_logits_bottleneck` and `token_logits` layers
    flat_token_logits = token_logits(token_logits_bottleneck(flat_hidden_states)) ### YOUR CODE HERE ###

    # then, we flatten the ground truth token ids.
    # remember, that we predict next tokens for each time step,
    # use `sentences` placeholder.
    # START is not included? (AK)
    flat_ground_truth = tf.reshape(sentences[:,1:], [-1]) ### YOUR CODE HERE ###

    # we need to know where we have real tokens (not padding) in `flat_ground_truth`,
    # we don't want to propagate the loss for padded output tokens,
    # fill `flat_loss_mask` with 1.0 for real tokens (not pad_idx) and 0.0 otherwise.
    #flat_loss_mask = K.switch(K.equal(flat_ground_truth, pad_idx),
    #                          K.constant(0.0),
    #                          K.constant(1.0))
    flat_loss_mask = tf.not_equal(flat_ground_truth, pad_idx)  ### YOUR CODE HERE ###

    # compute cross-entropy between `flat_ground_truth` and `flat_token_logits` predicted by lstm
    # out: a tensor of the same shape as labels
    xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=flat_ground_truth,
        logits=flat_token_logits
    )

    # compute average `xent` over tokens with nonzero `flat_loss_mask`.
    # we don't want to account misclassification of PAD tokens, because that doesn't make sense,
    # we have PAD tokens for batching purposes only!
    loss = tf.reduce_mean(tf.boolean_mask(xent, flat_loss_mask)) ### YOUR CODE HERE ###


# define optimizer operation to minimize the loss
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train_step = optimizer.minimize(decoder.loss)

# will be used to save/load network weights.
# you need to reset your default graph and define it in the same way to be able to load the saved weights!
saver = tf.train.Saver()

# intialize all variables
s.run(tf.global_variables_initializer())


# TESTS BEGIN

def get_feed_dict_for_testing(decoder, IMG_EMBED_SIZE, vocab):
    return {
        decoder.img_embeds: np.random.random((32, IMG_EMBED_SIZE)),
        decoder.sentences: np.random.randint(0, len(vocab), (32, 20))
    }


def test_decoder_shapes(decoder, IMG_EMBED_SIZE, vocab, s):
    tensors_to_test = [
        decoder.h0,
        decoder.word_embeds,
        decoder.flat_hidden_states,
        decoder.flat_token_logits,
        decoder.flat_ground_truth,
        decoder.flat_loss_mask,
        decoder.loss
    ]
    all_shapes = []
    for t in tensors_to_test:
        _ = s.run(t, feed_dict=get_feed_dict_for_testing(decoder, IMG_EMBED_SIZE, vocab))
        all_shapes.extend(_.shape)
    return all_shapes


def test_random_decoder_loss(decoder, IMG_EMBED_SIZE, vocab, s):
    loss = s.run(decoder.loss, feed_dict=get_feed_dict_for_testing(decoder, IMG_EMBED_SIZE, vocab))
    return loss


#test_decoder_shapes(decoder, IMG_EMBED_SIZE, vocab, s)

#test_random_decoder_loss(decoder, IMG_EMBED_SIZE, vocab, s)

# TESTS END


# Training loop
# Evaluate train and validation metrics through training and log them. Ensure that loss decreases.

train_captions_indexed = np.array(train_captions_indexed)
val_captions_indexed = np.array(val_captions_indexed)

# generate batch via random sampling of images and captions for them,
# we use `max_len` parameter to control the length of the captions (truncating long captions)
def generate_batch(images_embeddings, indexed_captions, batch_size, max_len=None):
    """
    `images_embeddings` is a np.array of shape [number of images, IMG_EMBED_SIZE].
    `indexed_captions` holds 5 vocabulary indexed captions for each image:
    [
        [
            [vocab[START], vocab["image1"], vocab["caption1"], vocab[END]],
            [vocab[START], vocab["image1"], vocab["caption2"], vocab[END]],
            ...
        ],
        ...
    ]
    Generate a random batch of size `batch_size`.
    Take random images and choose one random caption for each image.
    Remember to use `batch_captions_to_matrix` for padding and respect `max_len` parameter.
    Return feed dict {decoder.img_embeds: ..., decoder.sentences: ...}.
    """
    #print("images_embeddings.shape = ", images_embeddings.shape)  # (82783, 2048)
    #print("len(indexed_captions) = ", len(indexed_captions))  # 82783
    assert images_embeddings.shape[0] == len(indexed_captions)
    images = np.zeros(shape=(batch_size,images_embeddings.shape[1]))
    caps = []
    maxind = len(indexed_captions)
    for i in range(batch_size):
        ind = random.randrange(maxind)
        images[i] = images_embeddings[ind]
        cap = indexed_captions[ind][random.randrange(len(indexed_captions[ind]))]
        caps.append(cap)
    batch_image_embeddings = images ### YOUR CODE HERE ###
    batch_captions_matrix = batch_captions_to_matrix(caps, pad_idx, max_len) ### YOUR CODE HERE ###
    return {decoder.img_embeds: batch_image_embeddings,
            decoder.sentences: batch_captions_matrix}


batch_size = 64
n_epochs = 12
n_batches_per_epoch = 1000
n_validation_batches = 100  # how many batches are used for validation after each epoch

# you can load trained weights here
# you can load "weights_{epoch}" and continue training
# uncomment the next line if you need to load weights
#saver.restore(s, os.path.abspath("weights"))

# actual training loop
MAX_LEN = 20  # truncate long captions to speed up training

# to make training reproducible
np.random.seed(42)
random.seed(42)

# Look at the training and validation loss, they should be decreasing!
# You can skip this if you have loaded final weights.

for epoch in range(n_epochs): #range(1):

    train_loss = 0
    #pbar = tqdm_utils.tqdm_notebook_failsafe(range(n_batches_per_epoch))
    counter = 0
    #for _ in pbar:
    for _ in range(n_batches_per_epoch):
        train_loss += s.run([decoder.loss, train_step],
                            generate_batch(train_img_embeds,
                                           train_captions_indexed,
                                           batch_size,
                                           MAX_LEN))[0]
        counter += 1
        #pbar.set_description("Training loss: %f" % (train_loss / counter))
        if 0 == counter % 100 :
            print("  Training loss: %f" % (train_loss / counter))

    train_loss /= n_batches_per_epoch

    val_loss = 0
    for _ in range(n_validation_batches):
        val_loss += s.run(decoder.loss, generate_batch(val_img_embeds,
                                                       val_captions_indexed,
                                                       batch_size,
                                                       MAX_LEN))
    val_loss /= n_validation_batches

    print('Epoch: {}, train loss: {}, val loss: {}'.format(epoch, train_loss, val_loss))

    # save weights after finishing epoch
    saver.save(s, os.path.abspath("weights_{}".format(epoch)))

print("Training finished!")


# TEST BEGIN

def test_validation_loss(decoder, s, generate_batch, val_img_embeds, val_captions_indexed):
    np.random.seed(300)
    random.seed(300)
    val_loss = 0
    batches_for_eval = 1000
    #for _ in tqdm_utils.tqdm_notebook_failsafe(range(batches_for_eval)):
    for _ in range(batches_for_eval):
        val_loss += s.run(decoder.loss, generate_batch(val_img_embeds,
                                                       val_captions_indexed,
                                                       32,
                                                       20))
    val_loss /= 1000.
    return val_loss

#test_validation_loss(decoder, s, generate_batch, val_img_embeds, val_captions_indexed)

# TEST END


# check that it's learnt something, outputs accuracy of next word prediction (should be around 0.5)
from sklearn.metrics import accuracy_score, log_loss

def decode_sentence(sentence_indices):
    return " ".join(list(map(vocab_inverse.get, sentence_indices)))

def check_after_training(n_examples):
    fd = generate_batch(train_img_embeds, train_captions_indexed, batch_size)
    logits = decoder.flat_token_logits.eval(fd)
    truth = decoder.flat_ground_truth.eval(fd)
    mask = decoder.flat_loss_mask.eval(fd).astype(bool)
    print("Loss:", decoder.loss.eval(fd))
    print("Accuracy:", accuracy_score(logits.argmax(axis=1)[mask], truth[mask]))
    for example_idx in range(n_examples):
        print("Example", example_idx)
        print("Predicted:", decode_sentence(logits.argmax(axis=1).reshape((batch_size, -1))[example_idx]))
        print("Truth:", decode_sentence(truth.reshape((batch_size, -1))[example_idx]))
        print("")

check_after_training(3)

# save graph weights to file!
saver.save(s, os.path.abspath("weights"))

# Applying model
# Here we construct a graph for our final model.
#
# It will work as follows:
#
# take an image as an input and embed it
# condition lstm on that embedding
# predict the next token given a START input token
# use predicted token as an input at next time step
# iterate until you predict an END token

class final_model:
    # CNN encoder
    encoder, preprocess_for_model = get_cnn_encoder()
    saver.restore(s, os.path.abspath("weights"))  # keras applications corrupt our graph, so we restore trained weights

    # containers for current lstm state
    lstm_c = tf.Variable(tf.zeros([1, LSTM_UNITS]), name="cell")
    lstm_h = tf.Variable(tf.zeros([1, LSTM_UNITS]), name="hidden")

    # input images
    input_images = tf.placeholder('float32', [1, IMG_SIZE, IMG_SIZE, 3], name='images')

    # get image embeddings
    img_embeds = encoder(input_images)

    # initialize lstm state conditioned on image
    init_c = init_h = decoder.img_embed_bottleneck_to_h0(decoder.img_embed_to_bottleneck(img_embeds))
    init_lstm = tf.assign(lstm_c, init_c), tf.assign(lstm_h, init_h)

    # current word index
    current_word = tf.placeholder('int32', [1], name='current_input')

    # embedding for current word
    word_embed = decoder.word_embed(current_word)

    # apply lstm cell, get new lstm states
    new_c, new_h = decoder.lstm(word_embed, tf.nn.rnn_cell.LSTMStateTuple(lstm_c, lstm_h))[1]

    # compute logits for next token
    new_logits = decoder.token_logits(decoder.token_logits_bottleneck(new_h))
    # compute probabilities for next token
    new_probs = tf.nn.softmax(new_logits)

    # `one_step` outputs probabilities of next token and updates lstm hidden state
    one_step = new_probs, tf.assign(lstm_c, new_c), tf.assign(lstm_h, new_h)


# look at how temperature works for probability distributions
# for high temperature we have more uniform distribution
_ = np.array([0.5, 0.4, 0.1])
for t in [0.01, 0.1, 1, 10, 100]:
    print(" ".join(map(str, _**(1/t) / np.sum(_**(1/t)))), "with temperature", t)


# this is an actual prediction loop
def generate_caption(image, t=1, sample=False, max_len=20):
    """
    Generate caption for given image.
    if `sample` is True, we will sample next token from predicted probability distribution.
    `t` is a temperature during that sampling,
        higher `t` causes more uniform-like distribution = more chaos.
    """
    # condition lstm on the image
    s.run(final_model.init_lstm,
          {final_model.input_images: [image]})

    # current caption
    # start with only START token
    caption = [vocab[START]]

    for _ in range(max_len):
        next_word_probs = s.run(final_model.one_step,
                                {final_model.current_word: [caption[-1]]})[0]
        next_word_probs = next_word_probs.ravel()

        # apply temperature
        next_word_probs = next_word_probs ** (1 / t) / np.sum(next_word_probs ** (1 / t))

        if sample:
            next_word = np.random.choice(range(len(vocab)), p=next_word_probs)
        else:
            next_word = np.argmax(next_word_probs)

        caption.append(next_word)
        if next_word == vocab[END]:
            break

    return list(map(vocab_inverse.get, caption))


# look at validation prediction example
def apply_model_to_image_raw_bytes(raw, fname=None, do_save=False):
    img = utils.decode_image_from_buf(raw)
    fig = plt.figure(figsize=(7, 7))
    plt.grid('off')
    plt.axis('off')
    plt.imshow(img)
    img = utils.crop_and_preprocess(img, (IMG_SIZE, IMG_SIZE), final_model.preprocess_for_model)
    plt.title(' '.join(generate_caption(img)[1:-1]))
    if do_save:
        plt.savefig(fname)
    plt.show()


def show_valid_example(val_img_fns, example_idx=0):
    zf = zipfile.ZipFile("val2014_sample.zip")
    all_files = set(val_img_fns)
    found_files = list(filter(lambda x: x.filename.rsplit("/")[-1] in all_files, zf.filelist))
    example = found_files[example_idx]
    apply_model_to_image_raw_bytes(zf.read(example))


show_valid_example(val_img_fns, example_idx=100)


# sample more images from validation
for idx in np.random.choice(range(len(zipfile.ZipFile("val2014_sample.zip").filelist) - 1), 10):
    show_valid_example(val_img_fns, example_idx=idx)
    time.sleep(1)


# You can download any image from the Internet and appply your model to it!

#download_utils.download_file(
#    "http://www.bijouxandbits.com/wp-content/uploads/2016/06/portal-cake-10.jpg",
#    "images/portal-cake-10.jpg"
#)

apply_model_to_image_raw_bytes(open("images/portal-cake-10.jpg", "rb").read(), 
                               "images/portal-cake-10_wcap", True)


# You can use images from validation set as follows:
#
# show_valid_example(val_img_fns, example_idx=...)
# You can use images from the Internet as follows:
#
#! wget ...
# apply_model_to_image_raw_bytes(open("...", "rb").read())
#
# That's it!
#
# Congratulations, you've trained your image captioning model and now can produce
# captions for any picture from the Internet!


### YOUR EXAMPLES HERE ###

