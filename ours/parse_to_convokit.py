import json

from convokit import Speaker, Utterance, Corpus, ConvoKitMeta
from shortuuid import uuid as new_id

from tqdm import tqdm

convos = []
# with open('toy_data.jsonl') as f:
with open('2018-02.threads.sample.json') as f:
    for line in f:
        convos.append(json.loads(line))

speakers = {}


def build_utterances(convo):
    utterances = []

    def process_utterance(utt, utterances, convo_id):
        global speakers
        speaker_name = utt['author']
        speaker_id = speaker_name if speaker_name != '[deleted]' else new_id()
        if speaker_id not in speakers:
            speakers[speaker_id] = Speaker(id=speaker_id, name=speaker_name, meta={'name' : speaker_name})
        utt_speaker = speakers[speaker_id]
        utt_id = utt['id']
        utt_toxicity = utt.get('toxicity', 0)
        utt_has_attack = utt_toxicity > 0.7
        utt_root = utt.get('link_id', utt_id)
        utt_reply_to = utt.get('parent_id', None)
        utt_timestamp = utt['created_utc']
        utt_is_header = utt_reply_to is None
        utt_text = utt['title'] if utt_is_header else utt['body']
        utt_obj = Utterance(id=utt_id, root=utt_root, reply_to=utt_reply_to, timestamp=utt_timestamp,
                            speaker=utt_speaker, text=utt_text, user=utt_speaker,
                            meta={'is_section_header': utt_is_header, 'comment_has_personal_attack': utt_has_attack,
                                  'toxicity': utt_toxicity}, conversation_id=convo_id)

        utterances.append(utt_obj)
        for child in utt['children']:
            process_utterance(child, utterances, convo_id)

    convo_id = new_id()
    process_utterance(convo, utterances, convo_id)
    return utterances

conv_meta = {}
all_utterances = []
for convo in tqdm(convos):
    # for convo in tqdm(convos[:500]):
    convo_nature = convo['has_toxic_child']
    convo['split'] = 'init'
    convo['annotation_year'] = '2020'
    conv_meta[convo['id']] = {'split' : 'init', 'annotation_year' : 2020, 'conversation_has_personal_attack' : convo_nature}
    utterances = build_utterances(convo)
    all_utterances.extend(utterances)

reddit_corpus = Corpus(utterances = all_utterances)
for convo in reddit_corpus.iter_conversations():
    first_id = convo.get_utterance_ids()[0]
    convo.meta.update(conv_meta[first_id])


for convo in reddit_corpus.iter_conversations():
    print(convo.meta)
with open("reddit-init/convo_meta.json", 'w') as f:
    json.dump(conv_meta, f)
# reddit_corpus.dump(name = 'reddit_init', base_path='./', force_version=1)
reddit_corpus.dump('reddit-init', base_path='./', force_version=1)


# reddit2 = Corpus("../ours/reddit-init/utterances.jsonl")
print("done")
