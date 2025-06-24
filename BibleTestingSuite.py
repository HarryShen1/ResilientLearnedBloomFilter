import csv

def alphaify(string):
    return "".join([c for c in string if c.isalpha()]).lower()

path = "archive/t_kjv.csv"

bible_corpus = []

with open(path, 'r', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)

    header = next(csv_reader)

    for row in csv_reader:
        for i in row[4].split(" "):
            bible_corpus.append(alphaify(i))

print(bible_corpus)