import json
from tqdm import tqdm

def print_comment(post, depth):
    print("  "*depth , "|", file=f)
    # print("  |"*depth)
    post_body = post['body'].replace('\n', ' ')
    if post['toxicity'] >= 0.7:
        print('  '*depth, ' ', '--'*depth, post['author'], " $$$ TOXIC (" + str(post['toxicity']) +")$$$ ", post_body, file=f)
    else:
        print('  '*depth, ' ', '--'*depth, post['author'], " : ", post_body, file=f)

    for child in post['children']:
        print_comment(child, depth+1)

posts = []
# with open('toy_data.jsonl') as f:
with open('2018-02.threads.sample.json') as f:
    for line in f:
        posts.append(json.loads(line))

print("Finished loading the file")
f = open("conversation_tree.txt", "w", encoding='utf-8')

for post in tqdm(posts):
# for post in tqdm(posts[:500]):
    post_nature = "TOXIC" if post['has_toxic_child'] else "NORMAL"
    print("\n################ ", post_nature," CONVERSATION --- Sub-Reddit",  post['subreddit'],"################ \n", file=f)
    post_body = post['title'].replace('\n', ' ')
    print(post['author'], " : ", post['title'], file=f)
    for child in post['children']:
        print_comment(child, 1)
