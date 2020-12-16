import copy

from utils import tokenize
from vocabulary import loadPrecomputedVoc
from shortuuid import uuid as new_id



from consts import *
from modeling import EncoderRNN, ContextEncoderRNN, SingleTargetClf, Predictor
from train_test_utils import evaluateDataset

import torch
import torch.nn as nn
import os
from urllib.request import urlretrieve



voc = loadPrecomputedVoc("wikiconv", WORD2INDEX_URL, INDEX2WORD_URL)
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

print("Loading saved parameters...")
if not os.path.isfile("model.tar"):
    print("\tDownloading trained CRAFT...")
    urlretrieve(MODEL_URL, "model.tar")
    print("\t...Done!")
checkpoint = torch.load("model.tar", map_location=torch.device('cpu'))
# If running in a non-GPU environment, you need to tell PyTorch to convert the parameters to CPU tensor format.
# To do so, replace the previous line with the following:
#checkpoint = torch.load("model.tar", map_location=torch.device('cpu'))
encoder_sd = checkpoint['en']
context_sd = checkpoint['ctx']
attack_clf_sd = checkpoint['atk_clf']
embedding_sd = checkpoint['embedding']
voc.__dict__ = checkpoint['voc_dict']

print('Building encoders, decoder, and classifier...')
# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
embedding.load_state_dict(embedding_sd)
# Initialize utterance and context encoders
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
context_encoder = ContextEncoderRNN(hidden_size, context_encoder_n_layers, dropout)
encoder.load_state_dict(encoder_sd)
context_encoder.load_state_dict(context_sd)
# Initialize classifier
attack_clf = SingleTargetClf(hidden_size, dropout)
attack_clf.load_state_dict(attack_clf_sd)
# Use appropriate device
encoder = encoder.to(device)
context_encoder = context_encoder.to(device)
attack_clf = attack_clf.to(device)
print('Models built and ready to go!')

# Set dropout layers to eval mode
encoder.eval()
context_encoder.eval()
attack_clf.eval()

# Initialize the pipeline
predictor = Predictor(encoder, context_encoder, attack_clf)











def processDialog(voc, dialog):
    processed = []
    for utterance in dialog:
        tokens = tokenize(utterance)
        # replace out-of-vocabulary tokens
        for i in range(len(tokens)):
            if tokens[i] not in voc.word2index:
                tokens[i] = "UNK"
        processed.append(
            {"tokens": tokens, "is_attack": 0, "id": new_id()})
    return processed



def loadPairs(voc, convo):
    pairs = []
    dialog = processDialog(voc, convo)
    for idx in range(1, len(dialog)):
        reply = dialog[idx]["tokens"][:(MAX_LENGTH - 1)]
        label = dialog[idx]["is_attack"]
        comment_id = dialog[idx]["id"]
        # gather as context all utterances preceding the reply
        context = [u["tokens"][:(MAX_LENGTH - 1)] for u in dialog[:idx+1]]
        pairs.append((context, reply, label, comment_id))
    return pairs



def toxicity(conversation):
    pairs = loadPairs(voc, conversation)
    if len(pairs) == 0:
        return 0
    if len(pairs) < 2:
        pairs = [pairs[-1]] + [copy.deepcopy(pairs[-1])] + [copy.deepcopy(pairs[-1])]
    forecasts_df = evaluateDataset(pairs, encoder, context_encoder, predictor, voc, 64, device)
    print(forecasts_df.to_dict()['score'], pairs[-1])
    return forecasts_df.to_dict()['score'][pairs[-1][-1]]


def cli():
    conversation = ["Belgian crisis is about a major crisis", "Open for review"]
    while True:
        new_comment = input(">>> Enter new comment: ")
        conversation.append(new_comment)
        pairs = loadPairs(voc, conversation)
        # Run the pipeline!
        forecasts_df = evaluateDataset(pairs, encoder, context_encoder, predictor, voc, 64, device)
        print(forecasts_df.to_dict()['score'], pairs[-1])
