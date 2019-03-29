import numpy as np
import h5py
import os

if __name__ ==  "__main__":
    split = 'train'
    datafn = os.path.join('../', '%s_imfeats.h5' % split)
    dataset = h5py.File(datafn, 'r')
    phrases = list(dataset['phrases'])
    phrased = {}
    for phrase in phrases:
        phrased[phrase] = 0
    pairs = dataset["pairs"]
    phrasesInPair = pairs[1]
    for phraseInpair in phrasesInPair:
        if phraseInpair not in phrased:
            print(phraseInpair + "not in!!!")
        else:
            phrased[phraseInpair] = phrased[phraseInpair] + 1
    fre = phrased.values()
    fre.sort(reverse=True)
    print(fre[0:20])



