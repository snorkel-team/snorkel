import bz2
import os

from snorkel.contrib.babble import Explanation

def get_user_lists():
    def strip_special(s):
        return ''.join(c for c in s if ord(c) < 128)
    
    def last_name(s):
        name_parts = s.split(' ')
        return name_parts[-1] if len(name_parts) > 1 else None  

    spouses_pickle_path = os.path.join(os.environ['SNORKELHOME'], 
        'tutorials/intro/data/spouses_dbpedia.csv.bz2') 
    # Read in known spouse pairs and save as set of tuples
    with bz2.BZ2File(spouses_pickle_path, 'rb') as f:
        known_spouses = set(
            tuple(strip_special(x).strip().split(',')) for x in f.readlines()
        )
    # Last name pairs for known spouses
    last_names = set([(last_name(x), last_name(y)) for x, y in known_spouses if last_name(x) and last_name(y)])
    
    user_lists = {
        'spouse':  ['spouse', 'wife', 'husband', 'ex-wife', 'ex-husband'],
        'family':  ['father', 'father', 'mother', 'sister', 'sisters', 
                    'brother', 'brothers', 'son', 'sons', 'daughter', 'daughters',
                    'grandfather', 'grandmother', 'uncle', 'uncles', 'aunt', 'aunts', 
                    'cousin', 'cousins'],
        'other':  ['boyfriend', 'girlfriend', 'boss', 'employee', 'secretary', 'co-worker'],
        'known_spouses': known_spouses,
        'last_names': last_names}    
    user_lists['family'] += ["{}-in-law".format(f) for f in user_lists['family']]
    return user_lists

basic = [
    Explanation(
        name='LF_spouse_between',
        condition="Label true because there is at least one spouse word in the words between arg 1 and arg 2",
        candidate=-7068856930834066321,
        label=1,
        semantics=None),
    Explanation(
        name='LF_spouse_to_left',
        condition="Label true because there is at least one spouse word within two words to the left of arg 1 or arg 2",
        candidate=164897408906223198,
        label=1,
        semantics=None),
    # Explanation(
    #     name='LF_same_last_name',
    #     condition="Label true because the last word of arg 1 is the same as the last word of arg 2",
    #     candidate=None
    #     label=1,
    #     semantics=None),        
    Explanation(
        name='LF_no_spouse_in_sentence',
        condition="Label false because there are no spouse words in the sentence",
        candidate=-2695529595395795063,
        label=-1,
        semantics=None),
    Explanation(
        name='LF_arg1_and_arg2_married',
        condition="Label true because the word 'and' is between arg 1 and arg 2 and 'married' is in the sentence",
        candidate=-4636660390324264964,
        label=1,
        semantics=None),
    Explanation(
        name='LF_family_between',
        condition="Label false because there is at least one family word between arg 1 and arg 2",
        candidate=4207243625790983703,
        label=-1,
        semantics=None),
    Explanation(
        name='LF_family_to_left',
        condition="Label false because there is at least one family word within three words to the left of arg 1 or arg 2",
        candidate=5598460573200470481,
        label=-1,
        semantics=None),
    Explanation(
        name='LF_other_between',
        condition="Label false because there is an other word between arg 1 and arg 2",
        candidate=8943619341736037326,
        label=-1,
        semantics=None),
]

distant = [
    Explanation(
            name='LF_distant',
            condition="Label true because either the pair of arg 1 and arg 2 or the pair arg 2 and arg 1 is in known_spouses",
            candidate=-2597662937532403956,
            label=1,
            semantics=None),    
    # Explanation(
    #         name='LF_distant_last_names',
    #         condition="Label true because the arg 1 is not arg 2 and the pair of the last word of arg 1 and the last word of arg 2 is in last_names or the pair of the last word of arg 2 and the last word of arg 1 is in last_names",
    #         candidate=6734564861298611614,
    #         label=1,
    #         semantics=None),    
]

additional = [
    Explanation(
        name='LF_too_far_apart',
        condition="Label false because the number of words between arg 1 and arg 2 is larger than 10",
        candidate=-350026044943837397,
        label=-1,
        semantics=None),
    Explanation(
        name='LF_third_wheel',
        condition="Label false because there is a person between arg 1 and arg 2",
        candidate=4045249959449521689,
        label=-1,
        semantics=None),
    Explanation(
        name='LF_identical_args',
        condition="Label false because arg 1 is identical to arg 2",
        candidate=-8721107193035604739,
        label=-1,
        semantics=None),
]

examples = basic + distant + additional