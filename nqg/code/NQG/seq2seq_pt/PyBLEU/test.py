import nltk_bleu_score
import sys
from typing import List

ref_file : str = sys.argv[1]
system_file : str = sys.argv[2]

systems : List[str] = [] # Hyphothesis Sentences
refs : List[List[str]] = [] # For references Sentences

with open(system_file, encoding='utf-8') as f:
    for line in f:
        if not line:
            break 
        systems.append(line.strip().split(" "))

with open(ref_file, encoding="utf-8") as f:
    for line in f:
        if not line:
            break
        refs.append([line.strip().split(" ")])

blue : float = nltk_bleu_score.corpus_bleu(refs, systems)
print("BLEU: {0}".format(blue))
