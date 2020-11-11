import pandas as pd
import torch
import numpy as np

from consts import MAX_LENGTH, clip, device
from utils import batchIterator
import torch.nn.functional as F


def evaluateBatch(predictor, voc, input_batch, dialog_lengths,
                  dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, batch_size,
                  max_length=MAX_LENGTH):
    # Set device options
    input_batch = input_batch.to(device)
    dialog_lengths = dialog_lengths.to(device)
    utt_lengths = utt_lengths.to(device)
    # Predict future attack using predictor
    scores = predictor(input_batch, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices,
                       batch_size, max_length)
    scores = F.sigmoid(scores)
    predictions = (scores > 0.5).float()
    return predictions, scores


def evaluateDataset(dataset, predictor, voc, batch_size):
    # create a batch iterator for the given data
    batch_iterator = batchIterator(voc, dataset, batch_size, shuffle=False)
    # find out how many iterations we will need to cover the whole dataset
    n_iters = len(dataset) // batch_size + int(len(dataset) % batch_size > 0)
    output_df = {
        "id": [],
        "prediction": [],
        "score": []
    }
    predictor.eval()
    all_preds = []
    all_labels = []
    for iteration in range(1, n_iters + 1):
        batch, batch_dialogs, _, true_batch_size = next(batch_iterator)
        # Extract fields from batch
        input_variable, dialog_lengths, utt_lengths, batch_indices, dialog_indices, labels, convo_ids, target_variable, mask, max_target_len = batch
        dialog_lengths_list = [len(x) for x in batch_dialogs]
        # run the model

        predictions, scores = evaluateBatch(predictor, voc, input_variable,
                                            dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices,
                                            dialog_indices,
                                            true_batch_size)
        all_preds += [p.item() for p in predictions]
        all_labels += [l.item() for l in labels]
        # format the output as a dataframe (which we can later re-join with the corpus)
        for i in range(true_batch_size):
            convo_id = convo_ids[i]
            pred = predictions[i].item()
            score = scores[i].item()
            output_df["id"].append(convo_id)
            output_df["prediction"].append(pred)
            output_df["score"].append(score)
        #print("Iteration: {}; Percent complete: {:.1f}%".format(iteration, iteration / n_iters * 100))
    acc = (np.asarray(all_preds) == np.asarray(all_labels)).mean()
    print("Validation accuracy : ", acc)
    return pd.DataFrame(output_df).set_index("id"), acc



def trainDataset(dataset, predictor, voc, batch_size, optimizers):
    batch_iterator = batchIterator(voc, dataset, batch_size)
    n_iters = len(dataset) // batch_size + int(len(dataset) % batch_size > 0)
    total_loss = 0
    print_every  = 20
    predictor.train()
    all_preds = []
    all_labels = []
    for iteration in range(1, n_iters + 1):
        batch, batch_dialogs, _, true_batch_size = next(batch_iterator)
        input_batch, dialog_lengths, utt_lengths, batch_indices, dialog_indices, labels, convo_ids, target_variable, mask, max_target_len = batch
        dialog_lengths_list = [len(x) for x in batch_dialogs]
        input_batch = input_batch.to(device)
        dialog_lengths = dialog_lengths.to(device)
        utt_lengths = utt_lengths.to(device)
        y_train = labels.to(device)
        y_pred = predictor(input_batch, dialog_lengths, dialog_lengths_list, utt_lengths, batch_indices, dialog_indices, batch_size, MAX_LENGTH)
        scores = F.sigmoid(y_pred)
        predictions = (scores > 0.5).float()
        all_preds += [p.item() for p in predictions]
        all_labels += [l.item() for l in labels]
        loss = F.binary_cross_entropy_with_logits(y_pred, y_train)
        loss.backward()
        _ = torch.nn.utils.clip_grad_norm_(predictor.encoder.parameters(), clip)
        _ = torch.nn.utils.clip_grad_norm_(predictor.context_encoder.parameters(), clip)
        _ = torch.nn.utils.clip_grad_norm_(predictor.classifier.parameters(), clip)
        [optimizer.step() for optimizer in optimizers]
        total_loss += loss.item()
        if iteration % print_every == 0:
            total_loss_avg = total_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f} accuracy: {:.4f}".format(iteration,
                                                                                          iteration / n_iters * 100,
                                                                                          total_loss_avg,  (np.asarray(all_preds) == np.asarray(all_labels)).mean()))
            total_loss = 0
    return (np.asarray(all_preds) == np.asarray(all_labels)).mean()