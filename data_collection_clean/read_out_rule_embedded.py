import json

from convokit import Corpus

with open('predicted_corpus74.46%.json') as f:
    predicted_corpus = json.load(f)

print('ok')

corpus = Corpus("./reddit-init7/utterances.jsonl")
with open('./reddit-init7/convo_meta.json') as f:
    conv_meta = json.load(f)
print("loaded")


acc_scores = {}
for convo_rule_id in predicted_corpus:
    convo_id, clean, rule_id = convo_rule_id.split('_')
    convo_id = convo_id + "_" + clean

    num_corr, num_tot = acc_scores.get(clean+"_" + rule_id, (0,0))


    conversation = corpus.conversations.get(convo_id)
    idx = 0
    preds = predicted_corpus[convo_rule_id]['utts']
    label = predicted_corpus[convo_rule_id]['label']
    if label == 0:
        # continue
        pass
    print(f"Conversation {convo_rule_id} with label {label}")
    pred = 0

    for utt in conversation.get_chronological_utterance_list():
        print(f'[{preds[idx]} --- {utt.text[:500]}')
        if preds[idx][1]>0:
            pred = 1
        idx += 1

    if pred == int(label):
        num_corr += 1
    num_tot += 1
    acc_scores[clean+"_" + rule_id] = (num_corr, num_tot)

    print('*'*100)

print(acc_scores)

# {'mod_0': (57, 87), 'mod_1': (70, 87), 'clean_0': (64, 69), 'clean_1': (67, 69)}


# {'mod_0': (145, 213), 'mod_1': (160, 213), 'clean_0': (167, 292), 'clean_1': (292, 292)}
# {'mod_0': (139, 213), 'mod_1': (173, 213), 'clean_0': (244, 292), 'clean_1': (277, 292)}

'''
Most landlords in NYC and a lot of other big cities are Jews. Not exactly known for their generosity.
Gays being irresponsible during a pandemic? Groundbreaking.
Apparently noone matters, except black lives. According to media that is. It's all I hear about. Just sayin. Not sure what Trump has to do w that.
Still started in the degenerate markets. Trump is just as at fault tho but this virus is Chinese and spawned from an animalistic cuisine culture of filth.
Mexico really should start building that wall, please stay in your inbred country
And the Asian kid ate the hamster


UP yours


many fires were reported. This is a way for tughlaq to distract people into tribalism ... 
and his fanboys are probably here doing their usual misinformation thing.... its totally a rabid cult now.


LOL, Taiwan ren are just silly, they can dehumanize Chinese through language but I am sure mainland Chinese would one day get everything back

 Lol they wanna sell you their anus oil which protects you of COVID
 
 lmao based Viet.
 
 
 
 
 
 
 
 Dataset error
 
 Conversation foq2j4_mod_1 with label 0.0
[['foq2j4_mod', -1] --- New study casts doubt on hyrdoxychloroquine’s effectiveness in treating coronavirus
[['flgh2rh', -1] --- 30 patients lol... 🤦🏻‍♂️ that sample size is a joke.
[['flgh7ei', 1.0] --- The study that trumptards were promoting had 26 patients
'''


