from convokit import Corpus, download
from torch import optim

from consts import *
from modeling import make_model
from train_test_utils import evaluateDataset, trainDataset
from utils import loadPairs


corpus = Corpus(filename=download("conversations-gone-awry-corpus"))
predictor, voc = make_model('model.tar')
encoder_optimizer = optim.Adam(predictor.encoder.parameters(), lr=learning_rate)
context_encoder_optimizer = optim.Adam(predictor.context_encoder.parameters(), lr=learning_rate)
attack_clf_optimizer = optim.Adam(predictor.classifier.parameters(), lr=learning_rate)
optimizers = [encoder_optimizer, context_encoder_optimizer, attack_clf_optimizer]

train_pairs = loadPairs(voc, corpus, "train")
val_pairs = loadPairs(voc, corpus, "val")
test_pairs = loadPairs(voc, corpus, "test")

best_acc = -1
for epoch in range(1, train_epochs + 1):
    trainDataset(train_pairs, predictor, voc, batch_size, optimizers)
    _, acc = evaluateDataset(val_pairs, predictor, voc, batch_size)
    if acc > best_acc:
        print("Validation accuracy better than current best; saving model...")
        best_acc = acc
        torch.save({
            'iteration': epoch,
            'en': predictor.encoder.state_dict(),
            'ctx': predictor.context_encoder.state_dict(),
            'atk_clf': predictor.classifier.state_dict(),
            'en_opt': encoder_optimizer.state_dict(),
            'ctx_opt': context_encoder_optimizer.state_dict(),
            'atk_clf_opt': attack_clf_optimizer.state_dict(),
            'voc_dict': voc.__dict__,
            'embedding': predictor.encoder.embedding.state_dict()
        }, "finetuned_model.tar")