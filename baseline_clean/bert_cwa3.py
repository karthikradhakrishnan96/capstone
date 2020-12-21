# import necessary libraries, including convokit
import torch
from sklearn.metrics import f1_score
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import requests
import os
import sys
import random
import unicodedata
import itertools
from urllib.request import urlretrieve
from convokit import download, Corpus
import json

# define globals and constants

MAX_LENGTH = 120  # Maximum sentence length (number of tokens) to consider

# configure model
hidden_size = 768
context_encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 3
# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 1e-5
decoder_learning_ratio = 5.0
print_every = 100
train_epochs = 20

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token
UNK_token = 3  # Unknown word token
BERT_TYPE = 'bert-base-uncased'
BERT_TYPE = 'DeepPavlov/bert-base-cased-conversational'

SUBREDDIT = 'Coronavirus'
NUM_CLASSES = 2

# rule_config = [
#     '''Incivility isn’t allowed on this sub. We want to encourage a respectful discussion.  No expressions of ableism, homophobia, racism, sexism, transphobia or any expressions that in any other way fail to recognize the dignity of others.''',
#     "Purely political posts and comments will be removed"]

# rule_config = [
#     'Last christmas I gave you my heart',
#     'All I want for christmas is you'
# ]

#
# with open('rules.json') as f:
#     rule_configs = json.load(f)

rule_configs = {
    "subreddits": {
        "Coronavirus": [
            "incivility",
            "politics"
        ]
    },
    "rules": {
        "incivility": [
            "Incivility isn’t allowed on this sub. We want to encourage a respectful discussion.",
            "Remain civil towards other users, no expressions of ableism, homophobia, racism, sexism, transphobia, gendered slurs, ethnic slurs, slurs referring to disabilities and slurs against LGBT",
            "No trolling, hate speech, derogatory slurs, and personal attacks.",
            "Discrimination of any kind is not ok. No slurs or hate speech. Don't be a Jerk. Don't be Rude or Condescending. No trolling, personal attacks",
            "Stay respectful, polite, and friendly. No bigoted slurs, directed at other users. Don't insult people",
            "Personal attacks, insults, racial, homophobic, xenophobic, and sexist are not allowed",
        ],
        "politics": [
            "Purely political posts and comments will be removed",
            "No political debate",
            "Political discussion is not acceptable",
            "Comments cannot be inherently political, attempts to derail will be removed",
            "Off topic political, policy, and economic posts and comments will be removed",
            "Shaming campaigns, politician's take or political opinion pieces are not allowed",
            "No political opinions or hot takes or sensationalist controversies or tweets from president",
        ]
    }
}

def get_rule_texts(subreddit):
    return [rule_configs['rules'][rule_name] for rule_name in rule_configs['subreddits'][subreddit]]

class RuleConfig:
    def __init__(self, rules):
        self.rules = rules

    def __getitem__(self, index):
        return random.choice(self.rules[index])

    def __len__(self):
        return len(self.rules)

# Change this later
rule_config = RuleConfig(get_rule_texts(SUBREDDIT))

# model download paths
WORD2INDEX_URL = "http://zissou.infosci.cornell.edu/convokit/models/craft_cmv/word2index.json"
INDEX2WORD_URL = "http://zissou.infosci.cornell.edu/convokit/models/craft_cmv/index2word.json"
MODEL_URL = "http://zissou.infosci.cornell.edu/convokit/models/craft_cmv/craft_pretrained.tar"


# Helper functions for preprocessing and tokenizing text

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained(BERT_TYPE)


# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def encode(text, max_length=510):
    # simplify the problem space by considering only ASCII data
    cleaned_text = unicodeToAscii(text)

    # if the resulting string is empty, nothing else to do
    if not cleaned_text.strip():
        return []

    return tokenizer.encode(cleaned_text, add_special_tokens=True, max_length=max_length)


# Given a ConvoKit conversation, preprocess each utterance's text by tokenizing and truncating.
# Returns the processed dialog entry where text has been replaced with a list of
# tokens, each no longer than MAX_LENGTH - 1 (to leave space for the EOS token)
def processDialog(voc, dialog):
    processed = []
    for utterance in dialog.get_chronological_utterance_list():
        # skip the section header, which does not contain conversational content
        if utterance.meta['is_section_header']:
            continue
        tokens = utterance.text + " | "
        processed.append(
            {"tokens": tokens, "is_attack": int(utterance.meta['comment_has_personal_attack']), "id": utterance.id})

    return processed


# Load context-reply pairs from the Corpus, optionally filtering to only conversations
# from the specified split (train, val, or test).
# Each conversation, which has N comments (not including the section header) will
# get converted into N-1 comment-reply pairs, one pair for each reply
# (the first comment does not reply to anything).
# Each comment-reply pair is a tuple consisting of the conversational context
# (that is, all comments prior to the reply), the reply itself, the label (that
# is, whether the reply contained a derailment event), and the comment ID of the
# reply (for later use in re-joining with the ConvoKit corpus).
# The function returns a list of such pairs.
def get_wrong_rule_idx(correct_rule_idx, rule_config):
    total = len(rule_config)
    rule_idxs = [i for i in range(total) if i != correct_rule_idx]
    return random.choice(rule_idxs)


def loadPairs(voc, corpus, split=None, is_train=False):
    pairs = []
    rule_counts = {}
    for convo in corpus.iter_conversations():
        # consider only conversations in the specified split of the data
        last_utt = convo.get_chronological_utterance_list()[-1]
        rule_idx = None
        if last_utt.meta['comment_has_personal_attack']:
            rule_idx = last_utt.meta['rule_idx']

        if (split is None or convo.meta['split'] == split) and convo.get_utterance(convo.id).meta.get('subreddit',
                                                                                                      '') == SUBREDDIT and len(
            convo.get_utterance_ids()) > 2:
            rule_counts[rule_idx] = rule_counts.get(rule_idx, 0) + 1
            dialog = processDialog(voc, convo)
            idx = len(dialog) - 1
            reply = encode(dialog[idx]['tokens'])
            label = dialog[idx]['is_attack']
            comment_id = dialog[idx]["id"]
            if is_train:
                if label == 1:
                    correct_rule_idx = rule_idx
                    wrong_rule_idx = get_wrong_rule_idx(correct_rule_idx, rule_config)
                    context1 = [encode(u["tokens"] + rule_config[correct_rule_idx], max_length=MAX_LENGTH) for u in
                                dialog[:idx + 1]]
                    context2 = [encode(u["tokens"] + rule_config[wrong_rule_idx], max_length=MAX_LENGTH) for u in
                                dialog[:idx + 1]]
                    pairs.append((context1, reply, 1, convo.get_id() + "_" + comment_id + "_" + str(correct_rule_idx),
                                  encode('dummy')))
                    pairs.append((context2, reply, 0, convo.get_id() + "_" + comment_id + "_" + str(wrong_rule_idx),
                                  encode('dummy')))
                else:
                    correct_rule_idx = -1
                    wrong_rule_idx = get_wrong_rule_idx(correct_rule_idx, rule_config)
                    wrong_rule_idx2 = get_wrong_rule_idx(wrong_rule_idx, rule_config)
                    context1 = [encode(u["tokens"] + rule_config[wrong_rule_idx], max_length=MAX_LENGTH) for u in
                                dialog[:idx + 1]]
                    context2 = [encode(u["tokens"] + rule_config[wrong_rule_idx2], max_length=MAX_LENGTH) for u in
                                dialog[:idx + 1]]
                    pairs.append((context1, reply, 0, convo.get_id() + "_" + comment_id + "_" + str(wrong_rule_idx),
                                  encode('dummy')))
                    pairs.append(
                        (context2, reply, 0, convo.get_id() + "_" + comment_id + "_" + str(wrong_rule_idx2),
                         encode('dummy')))

            else:
                # Validation dataset with an escalation
                if label == 1:
                    correct_rule_idx = rule_idx
                    wrong_rule_idx = get_wrong_rule_idx(correct_rule_idx, rule_config)
                    iter_range = range(1, len(dialog))
                    for idx in iter_range:
                        reply = encode(dialog[idx]["tokens"])
                        comment_id = dialog[idx]["id"]
                        # gather as context all utterances preceding the reply
                        context = [encode(u["tokens"] + rule_config[correct_rule_idx], max_length=MAX_LENGTH) for u in
                                   dialog[:idx + 1]]
                        pairs.append(
                            (context, reply, 1, convo.get_id() + "_" + comment_id + "_" + str(correct_rule_idx),
                             encode('dummy')))

                    for idx in iter_range:
                        reply = encode(dialog[idx]["tokens"])
                        comment_id = dialog[idx]["id"]
                        # gather as context all utterances preceding the reply
                        context = [encode(u["tokens"] + rule_config[wrong_rule_idx], max_length=MAX_LENGTH) for u in
                                   dialog[:idx + 1]]
                        pairs.append((context, reply, 0, convo.get_id() + "_" + comment_id + "_" + str(wrong_rule_idx),
                                      encode('dummy')))
                else:
                    correct_rule_idx = -1
                    wrong_rule_idx = get_wrong_rule_idx(correct_rule_idx, rule_config)
                    iter_range = range(1, len(dialog))
                    for idx in iter_range:
                        reply = encode(dialog[idx]["tokens"])
                        comment_id = dialog[idx]["id"]
                        # gather as context all utterances preceding the reply
                        context = [encode(u["tokens"] + rule_config[wrong_rule_idx], max_length=MAX_LENGTH) for u in
                                   dialog[:idx + 1]]
                        pairs.append((context, reply, 0, convo.get_id() + "_" + comment_id + "_" + str(wrong_rule_idx),
                                      encode('dummy')))
                    wrong_rule_idx2 = get_wrong_rule_idx(wrong_rule_idx, rule_config)
                    for idx in iter_range:
                        reply = encode(dialog[idx]["tokens"])
                        comment_id = dialog[idx]["id"]
                        # gather as context all utterances preceding the reply
                        context = [encode(u["tokens"] + rule_config[wrong_rule_idx2], max_length=MAX_LENGTH) for u in
                                   dialog[:idx + 1]]
                        pairs.append((context, reply, 0, convo.get_id() + "_" + comment_id + "_" + str(wrong_rule_idx2),
                                      encode('dummy')))
    print(rule_counts)
    print(len([x for x in pairs if x[2] == 2]))
    print(len([x for x in pairs if x[2] == 1]))
    print(len([x for x in pairs if x[2] == 0]))
    return pairs


# Helper functions for turning dialog and text sequences into tensors, and manipulating those tensors

def indexesFromSentence(voc, sentence):
    return [voc.word2index[word] for word in sentence] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m


# Takes a batch of dialogs (lists of lists of tokens) and converts it into a
# batch of utterances (lists of tokens) sorted by length, while keeping track of
# the information needed to reconstruct the original batch of dialogs
def dialogBatch2UtteranceBatch(dialog_batch):
    utt_tuples = []  # will store tuples of (utterance, original position in batch, original position in dialog)
    for batch_idx in range(len(dialog_batch)):
        dialog = dialog_batch[batch_idx]
        for dialog_idx in range(len(dialog)):
            utterance = dialog[dialog_idx]
            utt_tuples.append((utterance, batch_idx, dialog_idx))
    # sort the utterances in descending order of length, to remain consistent with pytorch padding requirements
    utt_tuples.sort(key=lambda x: len(x[0]), reverse=True)
    # return the utterances, original batch indices, and original dialog indices as separate lists
    utt_batch = [u[0] for u in utt_tuples]
    batch_indices = [u[1] for u in utt_tuples]
    dialog_indices = [u[2] for u in utt_tuples]
    return utt_batch, batch_indices, dialog_indices


# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
    indexes_batch = l
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
    indexes_batch = l
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len


# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch, already_sorted=False):
    if not already_sorted:
        pair_batch.sort(key=lambda x: len(x[0]), reverse=True)
    input_batch, output_batch, label_batch, id_batch, rule_batch = [], [], [], [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
        label_batch.append(pair[2])
        id_batch.append(pair[3])
        rule_batch.append(pair[4])

    dialog_lengths = torch.tensor([len(x) for x in input_batch])
    input_utterances, batch_indices, dialog_indices = dialogBatch2UtteranceBatch(input_batch)
    inp, utt_lengths = inputVar(input_utterances, voc)
    rule, rule_lengths = inputVar(rule_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    label_batch = torch.FloatTensor(label_batch) if label_batch[0] is not None else None
    return inp, dialog_lengths, utt_lengths, batch_indices, dialog_indices, label_batch, id_batch, output, mask, max_target_len, rule, rule_lengths


def batchIterator(voc, source_data, batch_size, shuffle=True):
    cur_idx = 0
    if shuffle:
        random.shuffle(source_data)
    while True:
        if cur_idx >= len(source_data):
            cur_idx = 0
            if shuffle:
                random.shuffle(source_data)
        batch = source_data[cur_idx:(cur_idx + batch_size)]
        # the true batch size may be smaller than the given batch size if there is not enough data left
        true_batch_size = len(batch)
        # ensure that the dialogs in this batch are sorted by length, as expected by the padding module
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        # for analysis purposes, get the source dialogs and labels associated with this batch
        batch_dialogs = [x[0] for x in batch]
        batch_labels = [x[2] for x in batch]
        # convert batch to tensors
        batch_tensors = batch2TrainData(voc, batch, already_sorted=True)
        yield (batch_tensors, batch_dialogs, batch_labels, true_batch_size)
        cur_idx += batch_size


old = sys.stdout
f = open('dummy', 'w')
sys.stdout = f
# corpus = Corpus(filename=download("conversations-gone-awry-corpus"))
corpus_dir = '/content/drive/My Drive/Capstone_data/reddit-init7'
corpus = Corpus('%s/utterances.jsonl' % corpus_dir)
with open('%s/convo_meta.json' % corpus_dir) as f:
    conv_meta = json.load(f)
for convo in corpus.iter_conversations():
    first_id = convo.id
    convo.meta.update(conv_meta[first_id])
sys.stdout = old

voc = None

# Convert the test set data into a list of input/label pairs. Each input will represent the conversation as a list of lists of tokens.
convo_ids = [convo.id for convo in corpus.iter_conversations()]
from sklearn.model_selection import train_test_split

train_convo_ids, test_convo_ids = train_test_split(convo_ids, test_size=0.2, random_state=42)

for convo in corpus.iter_conversations():
    convo.meta['split'] = 'train' if convo.id in train_convo_ids else 'val'

train_pairs = loadPairs(voc, corpus, 'train', is_train=True)
length = len(train_pairs)

val_pairs = loadPairs(voc, corpus, 'val', is_train=False)
print(test_convo_ids)
# val_pairs = loadPairs(voc, corpus, "val", last_only=True)
# test_pairs = loadPairs(voc, corpus, "test")


from transformers import BertModel


class EncoderBERT(nn.Module):
    def __init__(self):
        super(EncoderBERT, self).__init__()
        self.model = BertModel.from_pretrained(BERT_TYPE)

    def forward(self, input_seq, input_lengths):
        input_seq = input_seq.T
        mask_ids = (input_seq != 0) * 1
        token_ids = torch.ones_like(input_seq).to(device)
        enc = self.model.forward(input_ids=input_seq, attention_mask=mask_ids, token_type_ids=token_ids)
        return enc[1]


class ContextEncoderRNN(nn.Module):
    """This module represents the context encoder component of CRAFT, responsible for creating an order-sensitive vector representation of conversation context"""

    def __init__(self, hidden_size, n_layers=1, dropout=0):
        super(ContextEncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.rule_encoder = BertModel.from_pretrained(BERT_TYPE)
        # only unidirectional GRU for context encoding
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=False)

    def get_initial_hidden(self, rule, rule_lengths):
        with torch.no_grad():
            rule = rule.T
            mask_ids = (rule != 0) * 1
            token_ids = torch.ones_like(rule).to(device)

            mask_ids = mask_ids.to(device)
            rule = rule.to(device)

            enc = self.rule_encoder.forward(input_ids=rule, attention_mask=mask_ids, token_type_ids=token_ids)
            return enc[1]

    def forward(self, input_seq, input_lengths, hidden=None, rule=None, rule_lengths=None):
        init_hidden = self.get_initial_hidden(rule, rule_lengths)
        init_hidden = torch.stack((init_hidden, init_hidden))
        # Pack padded batch of sequences for RNN module
        input_lengths = input_lengths.cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_seq, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hx=init_hidden)
        # Unpack padding
        outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        # return output and final hidden state
        return outputs, hidden


class SingleTargetClf(nn.Module):
    """This module represents the CRAFT classifier head, which takes the context encoding and uses it to make a forecast"""

    def __init__(self, hidden_size, dropout=0.1):
        super(SingleTargetClf, self).__init__()

        self.hidden_size = hidden_size

        # initialize classifier
        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.layer1_act = nn.LeakyReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size // 2)
        self.layer2_act = nn.LeakyReLU()
        self.clf = nn.Linear(hidden_size // 2, NUM_CLASSES)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, encoder_outputs, encoder_input_lengths):
        # from stackoverflow (https://stackoverflow.com/questions/50856936/taking-the-last-state-from-bilstm-bigru-in-pytorch)
        # First we unsqueeze seqlengths two times so it has the same number of
        # of dimensions as output_forward
        # (batch_size) -> (1, batch_size, 1)
        lengths = encoder_input_lengths.unsqueeze(0).unsqueeze(2)
        # Then we expand it accordingly
        # (1, batch_size, 1) -> (1, batch_size, hidden_size)
        lengths = lengths.expand((1, -1, encoder_outputs.size(2)))
        lengths = lengths.to(device)
        # take only the last state of the encoder for each batch
        last_outputs = torch.gather(encoder_outputs, 0, lengths - 1).squeeze()
        # forward pass through hidden layers
        layer1_out = self.layer1_act(self.layer1(self.dropout(last_outputs)))
        layer2_out = self.layer2_act(self.layer2(self.dropout(layer1_out)))
        # compute and return logits
        logits = self.clf(self.dropout(layer2_out))
        return logits


class Predictor(nn.Module):
    """This helper module encapsulates the CRAFT pipeline, defining the logic of passing an input through each consecutive sub-module."""

    def __init__(self, encoder, context_encoder, classifier):
        super(Predictor, self).__init__()
        self.encoder = encoder
        self.context_encoder = context_encoder
        self.classifier = classifier

    def forward(self, input_batch, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices,
                rule, rule_lengths,
                batch_size, max_length):
        # Forward input through encoder model
        utt_encoder_hidden = self.encoder(input_batch, utt_lengths)

        # Convert utterance encoder final states to batched dialogs for use by context encoder
        context_encoder_input = makeContextEncoderInput(utt_encoder_hidden, dialog_lengths_list, batch_size,
                                                        batch_indices, dialog_indices)

        # Forward pass through context encoder
        context_encoder_outputs, context_encoder_hidden = self.context_encoder(context_encoder_input, dialog_lengths,
                                                                               hidden=None, rule=rule,
                                                                               rule_lengths=rule_lengths)

        # Forward pass through classifier to get prediction logits
        logits = self.classifier(context_encoder_outputs, dialog_lengths)

        # Apply sigmoid activation
        # predictions = F.sigmoid(logits)
        predictions = logits
        return predictions


def makeContextEncoderInput(utt_encoder_hidden, dialog_lengths, batch_size, batch_indices, dialog_indices):
    """The utterance encoder takes in utterances in combined batches, with no knowledge of which ones go where in which conversation.
       Its output is therefore also unordered. We correct this by using the information computed during tensor conversion to regroup
       the utterance vectors into their proper conversational order."""
    # first, sum the forward and backward encoder states
    utt_encoder_summed = utt_encoder_hidden
    # we now have hidden state of shape [utterance_batch_size, hidden_size]
    # split it into a list of [hidden_size,] x utterance_batch_size
    last_states = [t.squeeze() for t in utt_encoder_summed.split(1, dim=0)]

    # create a placeholder list of tensors to group the states by source dialog
    states_dialog_batched = [[None for _ in range(dialog_lengths[i])] for i in range(batch_size)]

    # group the states by source dialog
    for hidden_state, batch_idx, dialog_idx in zip(last_states, batch_indices, dialog_indices):
        states_dialog_batched[batch_idx][dialog_idx] = hidden_state

    # stack each dialog into a tensor of shape [dialog_length, hidden_size]
    states_dialog_batched = [torch.stack(d) for d in states_dialog_batched]

    # finally, condense all the dialog tensors into a single zero-padded tensor
    # of shape [max_dialog_length, batch_size, hidden_size]
    return torch.nn.utils.rnn.pad_sequence(states_dialog_batched)


def train(input_variable, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices,
          rule, rule_lengths,
          labels,
          # input/output arguments
          encoder, context_encoder, attack_clf,  # network arguments
          encoder_optimizer, context_encoder_optimizer, attack_clf_optimizer,  # optimization arguments
          batch_size, clip, max_length=MAX_LENGTH):  # misc arguments

    # Zero gradients
    encoder_optimizer.zero_grad()
    context_encoder_optimizer.zero_grad()
    attack_clf_optimizer.zero_grad()
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 1]).to(device))
    # Set device options
    input_variable = input_variable.to(device)
    dialog_lengths = dialog_lengths.to(device)
    utt_lengths = utt_lengths.to(device)
    labels = labels.to(device)
    # Forward pass through utterance encoder
    utt_encoder_hidden = encoder(input_variable, utt_lengths)

    # Convert utterance encoder final states to batched dialogs for use by context encoder
    context_encoder_input = makeContextEncoderInput(utt_encoder_hidden, dialog_lengths_list, batch_size, batch_indices,
                                                    dialog_indices)

    # Forward pass through context encoder

    context_encoder_outputs, _ = context_encoder(context_encoder_input, dialog_lengths, hidden=None, rule=rule,
                                                 rule_lengths=rule_lengths)

    # Forward pass through classifier to get prediction logits
    logits = attack_clf(context_encoder_outputs, dialog_lengths)

    # Calculate loss
    # loss = F.binary_cross_entropy_with_logits(logits, labels)
    labels = labels.long()
    loss = criterion(logits, labels)

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(context_encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(attack_clf.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    context_encoder_optimizer.step()
    attack_clf_optimizer.step()

    return loss.item()


def evaluateBatch(encoder, context_encoder, predictor, voc, input_batch, dialog_lengths,
                  dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, rule, rule_lengths, batch_size,
                  device,
                  max_length=MAX_LENGTH):
    # Set device options
    input_batch = input_batch.to(device)
    dialog_lengths = dialog_lengths.to(device)
    utt_lengths = utt_lengths.to(device)
    # Predict future attack using predictor
    scores = predictor(input_batch, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices,
                       rule, rule_lengths,
                       batch_size, max_length)
    if len(scores.shape) == 1:
        scores = scores.unsqueeze(0)
    _, predictions = torch.max(scores, 1)

    return predictions, scores


def validate(dataset, encoder, context_encoder, predictor, voc, batch_size, device):
    # create a batch iterator for the given data
    batch_iterator = batchIterator(voc, dataset, batch_size, shuffle=False)
    # find out how many iterations we will need to cover the whole dataset
    n_iters = len(dataset) // batch_size + int(len(dataset) % batch_size > 0)
    # containers for full prediction results so we can compute accuracy at the end
    all_preds = []
    all_labels = []
    for iteration in range(1, n_iters + 1):
        batch, batch_dialogs, _, true_batch_size = next(batch_iterator)
        # Extract fields from batch
        input_variable, dialog_lengths, utt_lengths, batch_indices, dialog_indices, labels, convo_ids, target_variable, mask, max_target_len, rule, rule_lengths = batch
        dialog_lengths_list = [len(x) for x in batch_dialogs]
        # run the model
        predictions, scores = evaluateBatch(encoder, context_encoder, predictor, voc, input_variable,
                                            dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices,
                                            dialog_indices, rule, rule_lengths,
                                            true_batch_size, device)
        # print(predictions.shape)
        # print(predictions.min(), predictions.max())
        # aggregate results for computing accuracy at the end
        all_preds += [p.item() for p in predictions]
        all_labels += [l.item() for l in labels]
        if iteration % print_every == 0:
            print("Iteration: {}; Percent complete: {:.1f}%".format(iteration, iteration / n_iters * 100))

    # compute and return the accuracy
    return (np.asarray(all_preds) == np.asarray(all_labels)).mean()


def trainIters(voc, pairs, val_pairs, encoder, context_encoder, attack_clf,
               encoder_optimizer, context_encoder_optimizer, attack_clf_optimizer,
               n_iteration, batch_size, print_every, validate_every, clip):
    # create a batch iterator for training data
    batch_iterator = batchIterator(voc, pairs, batch_size)

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0

    # Training loop
    print("Training...")
    # keep track of best validation accuracy - only save when we have a model that beats the current best
    best_f1 = 0
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch, training_dialogs, _, true_batch_size = next(batch_iterator)
        # Extract fields from batch
        input_variable, dialog_lengths, utt_lengths, batch_indices, dialog_indices, labels, _, target_variable, mask, max_target_len, rule, rule_lengths = training_batch
        dialog_lengths_list = [len(x) for x in training_dialogs]

        # Run a training iteration with batch
        loss = train(input_variable, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices,
                     rule, rule_lengths,
                     labels,  # input/output arguments
                     encoder, context_encoder, attack_clf,  # network arguments
                     encoder_optimizer, context_encoder_optimizer, attack_clf_optimizer,  # optimization arguments
                     true_batch_size, clip)  # misc arguments
        # loss = 0
        print_loss += loss

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration,
                                                                                          iteration / n_iteration * 100,
                                                                                          print_loss_avg))
            print("Memory used : ", torch.cuda.max_memory_allocated())
            print_loss = 0

        # Evaluate on validation set
        if (iteration % validate_every == 0):
            print("Validating!")
            # put the network components into evaluation mode
            encoder.eval()
            context_encoder.eval()
            attack_clf.eval()

            predictor = Predictor(encoder, context_encoder, attack_clf)
            # accuracy = validate(val_pairs, encoder, context_encoder, predictor, voc, batch_size, device)
            f1, forecasts_df, predicted_corpus = get_best_val_f1(val_pairs, encoder, context_encoder, predictor, voc, batch_size, device)
            print("Validation set f1: {:.2f}%".format(f1 * 100))

            # keep track of our best model so far
            if f1 > best_f1:
                print("Validation accuracy better than current best; saving model...")
                best_f1 = f1
                out_file_name = 'val_forecasts_df_' + "{:.2f}%".format(f1 * 100)
                forecasts_df.to_csv(out_file_name, sep=',')

                with open('predicted_corpus' + "{:.2f}%".format(f1 * 100) + ".json", 'w') as f:
                    json.dump(predicted_corpus, f)

                torch.save({
                    'iteration': iteration,
                    'en': encoder.state_dict(),
                    'ctx': context_encoder.state_dict(),
                    'atk_clf': attack_clf.state_dict(),
                    'en_opt': encoder_optimizer.state_dict(),
                    'ctx_opt': context_encoder_optimizer.state_dict(),
                    'atk_clf_opt': attack_clf_optimizer.state_dict(),
                    'loss': loss,
                }, "finetuned_model.tar")

            # put the network components back into training mode
            encoder.train()
            context_encoder.train()
            attack_clf.train()


def evaluateDataset(dataset, encoder, context_encoder, predictor, voc, batch_size, device):
    # create a batch iterator for the given data
    batch_iterator = batchIterator(voc, dataset, batch_size, shuffle=False)
    # find out how many iterations we will need to cover the whole dataset
    n_iters = len(dataset) // batch_size + int(len(dataset) % batch_size > 0)
    output_df = {
        "id": [],
        "prediction": [],
        "score": [],
        'label': []
    }
    for iteration in range(1, n_iters + 1):
        batch, batch_dialogs, _, true_batch_size = next(batch_iterator)
        # Extract fields from batch
        input_variable, dialog_lengths, utt_lengths, batch_indices, dialog_indices, labels, convo_ids, target_variable, mask, max_target_len, rule, rule_lengths = batch
        dialog_lengths_list = [len(x) for x in batch_dialogs]
        # run the model
        predictions, scores = evaluateBatch(encoder, context_encoder, predictor, voc, input_variable,
                                            dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices,
                                            dialog_indices, rule, rule_lengths,
                                            true_batch_size, device)

        # format the output as a dataframe (which we can later re-join with the corpus)
        for i in range(true_batch_size):
            convo_id = convo_ids[i]
            pred = predictions[i].item()
            label = labels[i].item()
            score = -1  # scores[i].item()
            output_df["id"].append(convo_id)
            output_df["prediction"].append(pred)
            output_df["score"].append(score)
            output_df["label"].append(label)
        if iteration % print_every == 0:
            print("Iteration: {}; Percent complete: {:.1f}%".format(iteration, iteration / n_iters * 100))

    return pd.DataFrame(output_df).set_index("id")


# Utility fn to calculate val F1, used during training to check for best model
def get_best_val_f1(val_pairs, encoder, context_encoder, predictor, voc, batch_size, device):
    forecasts_df = evaluateDataset(val_pairs, encoder, context_encoder, predictor, voc, batch_size, device)

    predicted_corpus = {}
    for idx, row in forecasts_df.iterrows():
        # print(row)
        # print(_)
        # print(row['label'])
        # print(row['id'])
        convo_id, clean, utt_id, rule_id = idx.split('_')
        convo_id = convo_id + "_" + clean  # Because I messed up
        key = convo_id + "_" + rule_id
        if key not in predicted_corpus:
            predicted_corpus[key] = {'utts': [[utterance.id, -1] for utterance in
                                              corpus.get_conversation(convo_id).get_chronological_utterance_list()],
                                     'label': row['label']}
        preds = predicted_corpus[key]['utts']
        for idx in range(len(preds)):
            pred = preds[idx]
            if pred[0] == utt_id:
                pred[1] = row['prediction']
    conversational_forecasts_df = {
        'convo_rule_id': [],
        'label': [],
        'prediction': [],
    }
    for convo_rule_id in predicted_corpus:
        out = 0
        for utt_id, pred in predicted_corpus[convo_rule_id]['utts']:
            if pred == 1:
                out = 1
                break
        conversational_forecasts_df['prediction'].append(out)
        conversational_forecasts_df['label'].append(predicted_corpus[convo_rule_id]['label'])
        conversational_forecasts_df['convo_rule_id'] = convo_rule_id

    conversational_forecasts_df = pd.DataFrame(conversational_forecasts_df).set_index("convo_rule_id")
    return f1_score(conversational_forecasts_df.label, conversational_forecasts_df.prediction,
                    average='macro'), forecasts_df, predicted_corpus


# Fix random state (affect native Python code only, does not affect PyTorch and hence does not guarantee reproducibility)
random.seed(2019)

# Tell torch to use GPU. Note that if you are running this notebook in a non-GPU environment, you can change 'cuda' to 'cpu' to get the code to run.
device = torch.device('cuda')

# print("Loading saved parameters...")
# if not os.path.isfile("pretrained_model.tar"):
#     print("\tDownloading pre-trained CRAFT...")
#     urlretrieve(MODEL_URL, "pretrained_model.tar")
#     print("\t...Done!")
# checkpoint = torch.load("pretrained_model.tar")
# If running in a non-GPU environment, you need to tell PyTorch to convert the parameters to CPU tensor format.
# To do so, replace the previous line with the following:
# checkpoint = torch.load("model.tar", map_location=torch.device('cpu'))
# encoder_sd = checkpoint['en']
# context_sd = checkpoint['ctx']
# embedding_sd = checkpoint['embedding']
# voc.__dict__ = checkpoint['voc_dict']

print('Building encoders, decoder, and classifier...')
# Initialize word embeddings
# embedding = nn.Embedding(voc.num_words, hidden_size)
# embedding.load_state_dict(embedding_sd)
# Initialize utterance and context encoders
encoder = EncoderBERT()
context_encoder = ContextEncoderRNN(hidden_size, context_encoder_n_layers, dropout)
# encoder.load_state_dict(encoder_sd)
# context_encoder.load_state_dict(context_sd)
# Initialize classifier
attack_clf = SingleTargetClf(hidden_size, dropout)
# Use appropriate device
encoder = encoder.to(device)
context_encoder = context_encoder.to(device)
attack_clf = attack_clf.to(device)
print('Models built and ready to go!')

# Compute the number of training iterations we will need in order to achieve the number of epochs specified in the settings at the start of the notebook
n_iter_per_epoch = len(train_pairs) // batch_size + int(len(train_pairs) % batch_size == 1)
n_iteration = n_iter_per_epoch * train_epochs

# Put dropout layers in train mode
encoder.train()
context_encoder.train()
attack_clf.train()

# Initialize optimizers
print('Building optimizers...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
context_encoder_optimizer = optim.Adam(context_encoder.parameters(), lr=learning_rate)
attack_clf_optimizer = optim.Adam(attack_clf.parameters(), lr=learning_rate)
# Run training iterations, validating after every epoch
print("Starting Training!")
print("Will train for {} iterations".format(n_iteration))
trainIters(voc, train_pairs, val_pairs, encoder, context_encoder, attack_clf,
           encoder_optimizer, context_encoder_optimizer, attack_clf_optimizer,
           n_iteration, batch_size, print_every, n_iter_per_epoch, clip)
