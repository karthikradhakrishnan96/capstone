import json

from convokit import Corpus, download
from consts import *
from modeling import EncoderRNN, ContextEncoderRNN, SingleTargetClf, Predictor
from train_test_utils import evaluateDataset
from utils import loadPairs

import torch
import torch.nn as nn
import os
import random
from urllib.request import urlretrieve

from vocabulary import loadPrecomputedVoc

corpus = Corpus(filename=download("conversations-gone-awry-corpus"))
# corpus = Corpus("./ours/reddit-init/utterances.jsonl")
# with open('./ours/reddit-init/convo_meta.json') as f:
#     conv_meta = json.load(f)
#
# for convo in corpus.iter_conversations():
#     first_id = convo.get_utterance_ids()[0]
#     convo.meta.update(conv_meta[first_id])

# let's check some quick stats to verify that the corpus loaded correctly
print("Corpus has utts: ", len(corpus.get_utterance_ids()))
print("Corpus has users: ", len(corpus.get_usernames()))
print("Corpus has convs: ", len(corpus.get_conversation_ids()))

voc = loadPrecomputedVoc("wikiconv", WORD2INDEX_URL, INDEX2WORD_URL)

print("Voc has: ", voc.num_words) # expected vocab size is 50004: it was built using a fixed vocab size of 50k plus 4 spots for special tokens PAD, SOS, EOS, and UNK.
print("W2I: ", list(voc.word2index.items())[:10])
print("I2W: ", list(voc.index2word.items())[:10])

test_pairs = loadPairs(voc, corpus, "test")
print(sorted([a[-1] for a in test_pairs]))
# Fix random state for reproducibility
random.seed(2019)

# Tell torch to use GPU. Note that if you are running this notebook in a non-GPU environment, you can change 'cuda' to 'cpu' to get the code to run.
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

# Run the pipeline!
forecasts_df, _ = evaluateDataset(test_pairs, predictor, voc, batch_size)
forecasts_df.to_csv(out_file_name, sep=',')

