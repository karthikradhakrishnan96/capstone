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
NUM_CLASSES = 3 # Ideally num_rules + 1

# model download paths
WORD2INDEX_URL = "http://zissou.infosci.cornell.edu/convokit/models/craft_cmv/word2index.json"
INDEX2WORD_URL = "http://zissou.infosci.cornell.edu/convokit/models/craft_cmv/index2word.json"
MODEL_URL = "http://zissou.infosci.cornell.edu/convokit/models/craft_cmv/craft_pretrained.tar"

# confidence score threshold for declaring a positive prediction.
# this value was previously learned on the validation set.
FORECAST_THRESH = 0.570617

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


def encode(text):
    # simplify the problem space by considering only ASCII data
    cleaned_text = unicodeToAscii(text)

    # if the resulting string is empty, nothing else to do
    if not cleaned_text.strip():
        return []

    return tokenizer.encode(cleaned_text, add_special_tokens=True, max_length=MAX_LENGTH)


# Given a ConvoKit conversation, preprocess each utterance's text by tokenizing and truncating.
# Returns the processed dialog entry where text has been replaced with a list of
# tokens, each no longer than MAX_LENGTH - 1 (to leave space for the EOS token)
def processDialog(voc, dialog):
    processed = []
    for utterance in dialog.get_chronological_utterance_list():
        # skip the section header, which does not contain conversational content
        if utterance.meta['is_section_header']:
            continue
        tokens = encode(utterance.text)
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
def loadPairs(voc, corpus, split=None, last_only=False):
    pairs = []
    for convo in corpus.iter_conversations():
        # consider only conversations in the specified split of the data
        last_utt = convo.get_chronological_utterance_list()[-1]
        attack_type = 0
        if last_utt.meta['comment_has_personal_attack']:
            attack_type = 1 + last_utt.meta['rule_idx']

        if (split is None or convo.meta['split'] == split) and convo.get_utterance(convo.id).meta.get('subreddit',
                                                                                                      '') == SUBREDDIT and len(
                convo.get_utterance_ids()) > 2:
            dialog = processDialog(voc, convo)
            iter_range = range(1, len(dialog)) if not last_only else [len(dialog) - 1]
            for idx in iter_range:
                reply = dialog[idx]["tokens"]
                label = dialog[idx]["is_attack"]
                if label == 1:
                    label = attack_type
                comment_id = dialog[idx]["id"]
                # gather as context all utterances preceding the reply
                context = [u["tokens"] for u in dialog[:idx+1]]
                pairs.append((context, reply, label, comment_id))
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
    utt_tuples = [] # will store tuples of (utterance, original position in batch, original position in dialog)
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
    input_batch, output_batch, label_batch, id_batch = [], [], [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
        label_batch.append(pair[2])
        id_batch.append(pair[3])
    dialog_lengths = torch.tensor([len(x) for x in input_batch])
    input_utterances, batch_indices, dialog_indices = dialogBatch2UtteranceBatch(input_batch)
    inp, utt_lengths = inputVar(input_utterances, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    label_batch = torch.FloatTensor(label_batch) if label_batch[0] is not None else None
    return inp, dialog_lengths, utt_lengths, batch_indices, dialog_indices, label_batch, id_batch, output, mask, max_target_len

def batchIterator(voc, source_data, batch_size, shuffle=True):
    cur_idx = 0
    if shuffle:
        random.shuffle(source_data)
    while True:
        if cur_idx >= len(source_data):
            cur_idx = 0
            if shuffle:
                random.shuffle(source_data)
        batch = source_data[cur_idx:(cur_idx+batch_size)]
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
corpus = Corpus('/content/drive/My Drive/Capstone_data/reddit-init5/utterances.jsonl', exclude_speaker_meta = True)
with open('/content/drive/My Drive/Capstone_data/reddit-init5/convo_meta.json') as f:
    conv_meta = json.load(f)
for convo in corpus.iter_conversations():
    first_id = convo.id
    convo.meta.update(conv_meta[first_id])
sys.stdout = old


voc = None

# Convert the test set data into a list of input/label pairs. Each input will represent the conversation as a list of lists of tokens.
convo_ids = [convo.id for convo in corpus.iter_conversations()]
from sklearn.model_selection import train_test_split
train_convo_ids, test_convo_ids  = train_test_split(convo_ids, test_size = 0.2, random_state = 42)

for convo in corpus.iter_conversations():
  convo.meta['split'] = 'train' if convo.id in train_convo_ids else 'val'

train_pairs = loadPairs(voc, corpus, 'train', last_only=True)
length = len(train_pairs)

val_pairs = loadPairs(voc, corpus, 'val', last_only=False)
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

        # only unidirectional GRU for context encoding
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=False)

    def forward(self, input_seq, input_lengths, hidden=None):
        # Pack padded batch of sequences for RNN module
        input_lengths = input_lengths.cpu()
        packed = torch.nn.utils.rnn.pack_padded_sequence(input_seq, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
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
                batch_size, max_length):
        # Forward input through encoder model
        utt_encoder_hidden = self.encoder(input_batch, utt_lengths)

        # Convert utterance encoder final states to batched dialogs for use by context encoder
        context_encoder_input = makeContextEncoderInput(utt_encoder_hidden, dialog_lengths_list, batch_size,
                                                        batch_indices, dialog_indices)

        # Forward pass through context encoder
        context_encoder_outputs, context_encoder_hidden = self.context_encoder(context_encoder_input, dialog_lengths)

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


def train(input_variable, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, labels,
          # input/output arguments
          encoder, context_encoder, attack_clf,  # network arguments
          encoder_optimizer, context_encoder_optimizer, attack_clf_optimizer,  # optimization arguments
          batch_size, clip, max_length=MAX_LENGTH):  # misc arguments

    # Zero gradients
    encoder_optimizer.zero_grad()
    context_encoder_optimizer.zero_grad()
    attack_clf_optimizer.zero_grad()
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor([1, 2, 4]).to(device))
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
    context_encoder_outputs, _ = context_encoder(context_encoder_input, dialog_lengths)

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
                  dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, batch_size, device,
                  max_length=MAX_LENGTH):
    # Set device options
    input_batch = input_batch.to(device)
    dialog_lengths = dialog_lengths.to(device)
    utt_lengths = utt_lengths.to(device)
    # Predict future attack using predictor
    scores = predictor(input_batch, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices,
                       batch_size, max_length)
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
        input_variable, dialog_lengths, utt_lengths, batch_indices, dialog_indices, labels, convo_ids, target_variable, mask, max_target_len = batch
        dialog_lengths_list = [len(x) for x in batch_dialogs]
        # run the model
        predictions, scores = evaluateBatch(encoder, context_encoder, predictor, voc, input_variable,
                                            dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices,
                                            dialog_indices,
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
        input_variable, dialog_lengths, utt_lengths, batch_indices, dialog_indices, labels, _, target_variable, mask, max_target_len = training_batch
        dialog_lengths_list = [len(x) for x in training_dialogs]

        # Run a training iteration with batch
        loss = train(input_variable, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices,
                     labels,  # input/output arguments
                     encoder, context_encoder, attack_clf,  # network arguments
                     encoder_optimizer, context_encoder_optimizer, attack_clf_optimizer,  # optimization arguments
                     true_batch_size, clip)  # misc arguments
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
            f1, forecasts_df = get_best_val_f1(val_pairs, encoder, context_encoder, predictor, voc, batch_size, device)
            print("Validation set f1: {:.2f}%".format(f1 * 100))

            # keep track of our best model so far
            if f1 > best_f1:
                print("Validation accuracy better than current best; saving model...")
                best_f1 = f1
                out_file_name = 'val_forecasts_df_' + "{:.2f}%".format(f1*100)
                forecasts_df.to_csv(out_file_name, sep=',')
                # torch.save({
                #     'iteration': iteration,
                #     'en': encoder.state_dict(),
                #     'ctx': context_encoder.state_dict(),
                #     'atk_clf': attack_clf.state_dict(),
                #     'en_opt': encoder_optimizer.state_dict(),
                #     'ctx_opt': context_encoder_optimizer.state_dict(),
                #     'atk_clf_opt': attack_clf_optimizer.state_dict(),
                #     'loss': loss,
                # }, "finetuned_model.tar")

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
        "score": []
    }
    for iteration in range(1, n_iters + 1):
        batch, batch_dialogs, _, true_batch_size = next(batch_iterator)
        # Extract fields from batch
        input_variable, dialog_lengths, utt_lengths, batch_indices, dialog_indices, labels, convo_ids, target_variable, mask, max_target_len = batch
        dialog_lengths_list = [len(x) for x in batch_dialogs]
        # run the model
        predictions, scores = evaluateBatch(encoder, context_encoder, predictor, voc, input_variable,
                                            dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices,
                                            dialog_indices,
                                            true_batch_size, device)

        # format the output as a dataframe (which we can later re-join with the corpus)
        for i in range(true_batch_size):
            convo_id = convo_ids[i]
            pred = predictions[i].item()
            score = -1  # scores[i].item()
            output_df["id"].append(convo_id)
            output_df["prediction"].append(pred)
            output_df["score"].append(score)
        if iteration % print_every == 0:
            print("Iteration: {}; Percent complete: {:.1f}%".format(iteration, iteration / n_iters * 100))

    return pd.DataFrame(output_df).set_index("id")


# Utility fn to calculate val F1, used during training to check for best model
def get_best_val_f1(val_pairs, encoder, context_encoder, predictor, voc, batch_size, device):
    forecasts_df = evaluateDataset(val_pairs, encoder, context_encoder, predictor, voc, batch_size, device)
    for convo in corpus.iter_conversations():
        # only consider test set conversations (we did not make predictions for the other ones)
        if convo.meta['split'] == 'val':
            for utt in convo.get_chronological_utterance_list():
                if utt.id in forecasts_df.index:
                    utt.meta['forecast_score'] = forecasts_df.loc[utt.id].prediction
    conversational_forecasts_df = {
        "convo_id": [],
        "label": [],
        "score": [],
        "prediction": []
    }

    for convo in corpus.iter_conversations():
        if convo.meta['split'] == 'val':
            forecast_scores = [utt.meta['forecast_score'] for utt in convo.get_chronological_utterance_list() if
                               'forecast_score' in utt.meta]
            if len(forecast_scores) == 0:
                continue
            forecast_ids_not = [utt.id for utt in convo.get_chronological_utterance_list() if
                                'forecast_score' not in utt.meta]
            forecast_ids_yes = [utt.id for utt in convo.get_chronological_utterance_list() if
                                'forecast_score' in utt.meta]
            forecast_ids = [utt.id for utt in convo.get_chronological_utterance_list()]
            # Checked by printing, diff is the post and first comment
            prediction = 0
            for score in forecast_scores:
                if score != 0:
                    # Get the first non-zero
                    prediction = score
                    break

            conversational_forecasts_df['convo_id'].append(convo.id)
            conversational_forecasts_df['score'] = np.max(prediction)
            conversational_forecasts_df['prediction'].append(prediction)

            last_utt = convo.get_chronological_utterance_list()[-1]
            attack_type = 0
            if last_utt.meta['comment_has_personal_attack']:
                attack_type = 1 + last_utt.meta['rule_idx']
            conversational_forecasts_df['label'].append(attack_type)
    conversational_forecasts_df = pd.DataFrame(conversational_forecasts_df).set_index("convo_id")
    return f1_score(conversational_forecasts_df.label, conversational_forecasts_df.prediction, average='macro'), forecasts_df


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
#checkpoint = torch.load("model.tar", map_location=torch.device('cpu'))
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