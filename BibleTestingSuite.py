import csv
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import pickle

def alphaify(string):
    return "".join([c for c in string if c.isalpha()]).lower()
#
# path = "archive/t_kjv.csv"
#
# bible_corpus = []
# bible_corpus_nouns = []
# bible_corpus_notnouns = []
#
# with open(path, 'r', newline='') as csvfile:
#     csv_reader = csv.reader(csvfile)
#
#     header = next(csv_reader)
#
#     for row in csv_reader:
#         for i in row[4].split(" "):
#             entry = pos_tag(word_tokenize(alphaify(i)))
#             if entry != []:
#                 bible_corpus.append(entry[0][0])
#                 if entry[0][1] == "NN":
#                     bible_corpus_nouns.append(entry[0][0])
#                 else:
#                     bible_corpus_notnouns.append(entry[0][0])
#
# with open('bible_corpus.pkl', 'wb') as f:
#     pickle.dump(bible_corpus, f)
#
# with open('bible_corpus_nouns.pkl', 'wb') as f:
#     pickle.dump(bible_corpus_nouns, f)
#
# with open('bible_corpus_notnouns.pkl', 'wb') as f:
#     pickle.dump(bible_corpus_notnouns, f)
with open('bible_corpus.pkl', 'rb') as f:
    loaded_obj = pickle.load(f)

print(loaded_obj)