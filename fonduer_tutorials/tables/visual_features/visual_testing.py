import random
import copy
from pprint import pprint

class syntheticListGenerator:
    def __init__(self, scramble_pct=0, scramble_rng=0, duplicate_pct=0, 
                 mismatch_pct=0, partial_pct=0, offset=0):
        self.scramble_pct = scramble_pct
        self.scramble_rng = scramble_rng
        self.duplicate_pct = duplicate_pct
        self.duplicates = None
        self.mismatch_pct = mismatch_pct
        self.mismatches = None
        self.partial_pct = partial_pct
        self.partials = None
        self.offset = offset
        self.aligned = None
        
    def make_lists(self, N) :
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        duplicate_num = 0
        mismatch_num = 0
        partial_num = 0
        uid = 0
        
        # make parallel lists
        listA = [(i, ''.join(random.sample(alphabet, random.randint(3,10)))) for i in range(N)]
        listB = copy.deepcopy(listA)
        uid = N
        
        # make some duplicate words
        for i in range(N):
            if random.random() < self.duplicate_pct * 0.5:
                duplicate_num += 2
                listA[i] = (uid, listA[random.randint(0, N-1)][1])
                listB[i] = listA[i]
                uid += 1
        
        # replace some words
        for i in range(N):
            if random.random() < self.mismatch_pct:
                mismatch_num += 1
                listA[i] = (None, listA[i][1])
                listB[i] = (None, ''.join(random.sample(alphabet, random.randint(3,10))))
                uid += 1
        
        # cut some words in half
        for i in range(N):
            if random.random() < self.partial_pct:
                partial_num += 1
                if random.random() < 0.5:
                    listA[i] = (listA[i][0], listA[i][1][:-1])
                else:
                    listB[i] = (listB[i][0], listB[i][1][:-1])
                                    
        # scramble locally
        old_listB = listB
        listB = [None] * N
        i = 0
        j = 0
        for i in range(N):
            if i < (N - self.scramble_rng) and random.random() < self.scramble_pct:
                jump = random.randint(1, self.scramble_rng)
                k = j + jump
                while listB[k] is not None:
                    k += 1
                listB[k] = old_listB[i]
            else:
                while listB[j] is not None:
                    j += 1
                listB[j] = old_listB[i]  

        # offset 
        for i in range(self.offset):
            listB = [(None, ''.join(random.sample(alphabet, random.randint(3,10))))] + listB
        
        self.duplicates = float(duplicate_num)/N
        self.mismatches = float(mismatch_num)/N
        self.partials = float(partial_num)/N
        self.aligned = (sum([listA[i][0] == listB[i][0] for i in range(N)]) / float(N))

        return (listA, listB)

def check_matches(links, listA, listB, display=False):
    N = len(links)
    dictB = {k:v for k,v in listB}
    matches = []
    for i, a in enumerate(listA):
        b = (links[a[0]], dictB[links[a[0]]])
        correct = 'O' if a[0] == b[0] else 'X'
        matches.append((correct, a, b))
    nMatches = sum([match[2] is not None and match[1][0]==match[2][0] for match in matches])
    print "Matches: %d/%d (%.2f)" % (nMatches, N, float(nMatches)/N)
    if display:
        pprint(matches)