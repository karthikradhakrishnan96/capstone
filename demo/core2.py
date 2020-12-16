import copy

from utils import tokenize, unicodeToAscii
from vocabulary import loadPrecomputedVoc
from shortuuid import uuid as new_id



from modeling2 import EncoderBERT, ContextEncoderRNN, SingleTargetClf, Predictor
from train_test_utils2 import evaluateDataset

import torch
import torch.nn as nn
import os
from urllib.request import urlretrieve

BERT_TYPE = 'DeepPavlov/bert-base-cased-conversational'
from transformers import BertTokenizer
MAX_LENGTH = 250

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


tokenizer = BertTokenizer.from_pretrained(BERT_TYPE)

voc = None
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')


# checkpoint = torch.load("model.tar", map_location=torch.device('cpu'))
# # If running in a non-GPU environment, you need to tell PyTorch to convert the parameters to CPU tensor format.
# # To do so, replace the previous line with the following:
checkpoint = torch.load("finetuned_model70.11%.tar", map_location=torch.device('cpu'))
encoder_sd = checkpoint['en']
context_sd = checkpoint['ctx']
attack_clf_sd = checkpoint['atk_clf']

print('Building encoders, decoder, and classifier...')
# Initialize word embeddings
# embedding = nn.Embedding(voc.num_words, hidden_size)
# embedding.load_state_dict(embedding_sd)
# Initialize utterance and context encoders
encoder = EncoderBERT()
context_encoder = ContextEncoderRNN(hidden_size, context_encoder_n_layers, dropout)
encoder.load_state_dict(encoder_sd)
context_encoder.load_state_dict(context_sd)
# Initialize classifier
attack_clf = SingleTargetClf(hidden_size, dropout)
attack_clf.load_state_dict(attack_clf_sd)
# Use appropriate device
predictor = Predictor(encoder, context_encoder, attack_clf)
encoder = encoder.to(device)
context_encoder = context_encoder.to(device)
attack_clf = attack_clf.to(device)
predictor = predictor.to(device)
print('Models built and ready to go!')


# Put dropout layers in train mode
encoder.eval()
context_encoder.eval()
attack_clf.eval()
predictor.eval()








def processDialog(voc, dialog):
    processed = []
    for utterance in dialog:
        # skip the section header, which does not contain conversational content
        tokens = encode(utterance, max_length=MAX_LENGTH)
        processed.append(
            {"tokens": tokens, "is_attack": 0, "id": new_id()})

    return processed



def encode(text, max_length=510):
    # simplify the problem space by considering only ASCII data
    cleaned_text = unicodeToAscii(text)

    # if the resulting string is empty, nothing else to do
    if not cleaned_text.strip():
        return []

    return tokenizer.encode(cleaned_text, add_special_tokens=True, max_length=max_length)

def loadPairs(voc, convo, rule_text):
    pairs = []
    dialog = processDialog(voc, convo)
    for idx in range(1, len(dialog)):
        reply = dialog[idx]["tokens"]
        comment_id = dialog[idx]["id"]
        # gather as context all utterances preceding the reply
        context = [u["tokens"] for u in dialog[:idx+1]]
        pairs.append(
            (context, reply, 1, "dummy" + "_" + comment_id + "_" + str(0),
             encode(rule_text)))
    return pairs



def toxicity(conversation, rule_text):
    pairs = loadPairs(voc, conversation, rule_text)
    if len(pairs) == 0:
        return 0
    if len(pairs) < 2:
        pairs = [pairs[-1]] + [copy.deepcopy(pairs[-1])] + [copy.deepcopy(pairs[-1])]
    forecasts_df = evaluateDataset(pairs, encoder, context_encoder, predictor, voc, 64, device)
    print(forecasts_df.to_dict()['score'], pairs[-1])
    return forecasts_df.to_dict()['score'][pairs[-1][-1]]


def cli():
    conversation = [] # ["Totally Okay comment", "Open for review"]
    rule = 'No political opinions or hot takes or sensationalist controversies or tweets from president'
    while True:
        new_comment = input(">>> Enter new comment: ")
        temp = input(">>> Enter rule, leave blank to use prev : ")
        rule = rule if not temp else temp
        conversation.append(new_comment)
        if len(conversation) < 2:
            print("Too less to make prediction, continue")
            continue
        pairs = loadPairs(voc, conversation,  rule)
        # Run the pipeline!
        forecasts_df = evaluateDataset(pairs, encoder, context_encoder, predictor, voc, 64, device)
        print(forecasts_df.to_dict()['prediction'], pairs[-1])
        clear = input(">>> Clear conversation? leave blank to not clear : ")
        if clear:
            conversation = []


if __name__ == "__main__":
    cli()