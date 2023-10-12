from __future__ import division
import sys
import operator
from typing import List, Dict, Tuple

# OriginalFilename : CollectVocab.py 
# NewFilename : CollectVocabulary.py
# This file work as expected

DEFAULT_SPECIAL_WORDS : List[str] = ["<blank>", "<unk>", "<s>", "</s>"]

def Collect( inputFiles : List[str],
             vocabPath : str, toLower : bool = False, userDefinedSpecialWords : List[str] = None 
             ):
    """ 
        This Function saving the token and its cumulative percentage 
    """
    specialWords : List[str] = []
    if userDefinedSpecialWords:
        for word in userDefinedSpecialWords:
            if word not in specialWords:
                specialWords.append(word)
    else :
        specialWords = DEFAULT_SPECIAL_WORDS
    
    dictionary : Dict[str, int] = CollectVocabFromFiles(inputFiles, toLower)
    totalTokens : int = sum(dictionary.values()) # Cumulative, not unique
    dictionaryIncreasing = sorted(dictionary.items(), key=operator.itemgetter(1), reverse=True) # Sort based on values in increasing order 
    accuracy : float = 0
    
    with open(vocabPath, 'w', encoding='utf-8') as SRCwriter:
        uniqueTokenCounter : int = 0 
        for word in specialWords:
            SRCwriter.write("{0} {1}\n".format(word, uniqueTokenCounter))
            uniqueTokenCounter += 1
        
        for token, total in dictionaryIncreasing:
            if token in specialWords:
                continue
            accuracy += total
            SRCwriter.write("{0} {1} {2} {3}\n".format(token, uniqueTokenCounter, total, 1.0 * accuracy / totalTokens))
            uniqueTokenCounter+=1
    

def CollectVocabFromFiles(files : List[str],
                          toLower : bool = False) -> Dict[str, int]:
    """ 
        Previousfilename: CollectVocab
        This function is just counting tokens
        
        return dictionary[token, total] 
    """
   
    dictionary : Dict[str, int] = {}
    
    for _file in files:
        
        with open(_file, encoding='utf-8') as source:
            for line in source:
                line = line.strip() 
                if toLower :
                    line = line.lower()
                
                splittedString : List[str] = line.split()
               #spllitedString = filter(None, splittedString) # Not going to use this 
                for token in splittedString:
                    if token not in dictionary:
                        dictionary[token] = 0
                    dictionary[token] += 1

    return dictionary

if __name__ == '__main__':
    if len(sys.argv) > 3 :
        files : List[str] = sys.argv[:-1]
        vocab_file : str = sys.argv[-1] 
        Collect(files, vocab_file, False, None)
    else:
        print('CollectVocab.py: Collect vocabulary from multiple files.')
        print('Usage:')
        print('python CollectVocab.py file_1 file_2 ... file_n out.vocab.txt')