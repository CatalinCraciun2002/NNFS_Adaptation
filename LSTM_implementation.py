import nltk
import os

corpora_dir = "/Users/craci/AppData/Roaming/nltk_data/corpora/state_union"

# Read all file paths in corpora directory
file_list = []
for root, _, files in os.walk(corpora_dir):
    for filename in files:
        file_list.append(os.path.join(root, filename))

print("Read ", len(file_list), " files...")

# Extract text from all documents
docs = []

for files in file_list:
    with open(files, 'r') as fin:
        try:
            str_form = fin.read().lower().replace('\n', '')
            docs.append(str_form)
        except UnicodeDecodeError:
            # Some sentences have wierd characters. Ignore them for now
            pass
# Combine them all into a string of text
text = ' '.join(docs)

print('corpus length:', len(text))