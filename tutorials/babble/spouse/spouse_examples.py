import bz2
import os

from snorkel.contrib.babble import Explanation, link_explanation_candidates

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
    # Explanation(
    #     name='LF_spouse_between',
    #     condition="there is a spouse word between arg 1 and arg 2",
    #     candidate='2ca6dbbb-870c-4e34-8053-0ac2dbd850f5::span:798:808~~2ca6dbbb-870c-4e34-8053-0ac2dbd850f5::span:839:851',
    #     label=True,
    #     semantics=None),
    Explanation(
        name='LF_spouse_to_left',
        condition="there is a spouse word within two words to the left of arg 1 or arg 2",
        candidate='03a1e1a0-93c3-41a8-a905-a535ce8f2b09::span:6822:6837~~03a1e1a0-93c3-41a8-a905-a535ce8f2b09::span:6855:6858',
        label=True,
        semantics=None),
    # Explanation(
    #     name='LF_same_last_name',
    #     condition="the last word of arg 1 is the same as the last word of arg 2",
    #     candidate=None
    #     label=True,
    #     semantics=None),        
    Explanation(
        name='LF_no_spouse_in_sentence',
        condition="there are no spouse words in the sentence",
        candidate='d0de6a86-66d5-40e0-b345-6c86d2047c07::span:1634:1638~~d0de6a86-66d5-40e0-b345-6c86d2047c07::span:1650:1659',
        label=False,
        semantics=None),
    Explanation(
        name='LF_married_after',
        condition="the word 'and' is between arg 1 and arg 2 and 'married' or 'marriage' is after arg 2",
        candidate='e522e66f-ad1f-4b8b-a532-4f030a8e7a75::span:4054:4059~~e522e66f-ad1f-4b8b-a532-4f030a8e7a75::span:4085:4091',
        label=True,
        semantics=None),
    Explanation(
        name='LF_family_between',
        condition="there is a family word between arg 1 and arg 2",
        candidate='768f241b-786d-475e-a55c-9683ecdeeb86::span:518:529~~768f241b-786d-475e-a55c-9683ecdeeb86::span:637:638',
        label=False,
        semantics=None),
    Explanation(
        name='LF_family_to_left',
        condition="there is a family word within three words to the left of arg 1 or arg 2",
        candidate='b86261b6-62c3-456d-8ed0-458f781776f7::span:42:53~~b86261b6-62c3-456d-8ed0-458f781776f7::span:72:91',
        label=False,
        semantics=None),
    Explanation(
        name='LF_other_between',
        condition="there is an other word between arg 1 and arg 2",
        candidate='3375a3c2-9b8a-423a-8334-32fe860be60e::span:3939:3948~~3375a3c2-9b8a-423a-8334-32fe860be60e::span:3967:3981',
        label=False,
        semantics=None),
]

distant = [
    Explanation(
            name='LF_distant',
            condition="either the pair of arg 1 and arg 2 or the pair arg 2 and arg 1 is in known_spouses",
            candidate='bd72fb43-2c7d-4067-9e11-36a33623b855::span:3406:3419~~bd72fb43-2c7d-4067-9e11-36a33623b855::span:3496:3507',
            label=True,
            semantics=None),    
    # Explanation(
    #         name='LF_distant_last_names',
    #         condition="arg 1 is not arg 2 and the pair of the last word of arg 1 and the last word of arg 2 is in last_names or the pair of the last word of arg 2 and the last word of arg 1 is in last_names",
    #         candidate='68308b6b-c1c9-44f1-a236-ecb8936b8d48::span:1545:1556~~68308b6b-c1c9-44f1-a236-ecb8936b8d48::span:1567:1585',
    #         label=True,
    #         semantics=None),    
]

additional = [
    Explanation(
        name='LF_too_far_apart',
        condition="the number of words between arg 1 and arg 2 is larger than 10",
        candidate='2ca6dbbb-870c-4e34-8053-0ac2dbd850f5::span:839:851~~2ca6dbbb-870c-4e34-8053-0ac2dbd850f5::span:985:989',
        label=False,
        semantics=None),
    Explanation(
        name='LF_third_wheel',
        condition="there is a person between arg 1 and arg 2",
        candidate='3982c852-c844-47fa-98a3-ffa4c52a9c67::span:523:530~~3982c852-c844-47fa-98a3-ffa4c52a9c67::span:548:566',
        label=False,
        semantics=None),
    Explanation(
        name='LF_identical_args',
        condition="arg 1 is identical to arg 2",
        candidate='c313f020-d5f8-480f-85f5-dc639157f7e5::span:2957:2960~~c313f020-d5f8-480f-85f5-dc639157f7e5::span:3175:3178',
        label=False,
        semantics=None),
    ### GOOD ONE: "False because a person is between arg 1 and arg2"
]

all_explanations = basic + distant + additional

def get_explanations():
    return all_explanations