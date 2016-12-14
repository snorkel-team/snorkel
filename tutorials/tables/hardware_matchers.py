from snorkel.matchers import *
from snorkel.lf_helpers import *
import csv
matchers = {}


### PART ###
eeca_rgx = '([ABC][A-Z][WXYZ]?[0-9]{3,5}(?:[A-Z]){0,5}[0-9]?[A-Z]?(?:-[A-Z0-9]{1,7})?(?:[-][A-Z0-9]{1,2})?(?:\/DG)?)'
jedec_rgx = '(2N\d{3,4}[A-Z]{0,5}[0-9]?[A-Z]?)'
jis_rgx = '(2S[ABCDEFGHJKMQRSTVZ]{1}[\d]{2,4})'
others_rgx = '((?:NSVBC|SMBT|MJ|MJE|MPS|MRF|RCA|TIP|ZTX|ZT|ZXT|TIS|STD|BUV|TIPL|DTC|MMBT|SMMBT|PZT|FZT){1}[\d]{2,4}[A-Z]{0,3}(?:-[A-Z0-9]{0,6})?(?:[-][A-Z0-9]{0,1})?)'
part_rgx = '|'.join([eeca_rgx, jedec_rgx, jis_rgx, others_rgx])
# modifiers = '(?:[\/\-][A-Z]{,2})*'
# part_rgx = '(' + '|'.join([eeca_rgx, jedec_rgx, jis_rgx, others_rgx]) + ')' + modifiers
part_rgx_matcher = RegexMatchSpan(rgx=part_rgx, longest_match_only=False)

def part_conditions(part):
    """throttle parts that are in tables of device/replacement parts"""
    aligned_ngrams = set(get_aligned_ngrams(part))
    return not (overlap(['replacement'], aligned_ngrams) or
        (len(aligned_ngrams) > 25 and 'device' in aligned_ngrams) or
        get_prev_sibling_tags(part).count('p') > 25 or # CentralSemiconductorCorp_2N4013.pdf:
        overlap(['complementary', 'complement', 'empfohlene'], 
                chain.from_iterable([
                    get_left_ngrams(part, window=10),
                    get_aligned_ngrams(part)])))
part_lambda_matcher = LambdaFunctionMatch(func=part_conditions)
matchers['part_rgx'] = part_rgx_matcher

def attr_in_table(attr):
    return attr.is_tabular()
attr_in_table_matcher = LambdaFunctionMatch(func=attr_in_table)


### POLARITY ####
polarity_rgx_matcher = RegexMatchSpan(rgx=r'NPN|PNP', longest_match_only=False, ignore_case=True)
def polarity_conditions(attr):
    return not overlap(['complement','complementary'], get_phrase_ngrams(attr))

polarity_lambda_matcher = LambdaFunctionMatch(func=polarity_conditions)


### CE_V_MAX ###
ce_keywords = set(['collector emitter', 'collector-emitter', 'collector - emitter'])
ce_abbrevs = set(['ceo', 'vceo'])
ce_v_max_rgx_matcher = RegexMatchSpan(rgx=r'\d{1,2}[05]', longest_match_only=False)
def ce_v_max_conditions(attr):
    return overlap(ce_keywords.union(ce_abbrevs), get_row_ngrams(attr, spread=[0,3], n_max=3))
ce_v_max_row_matcher = LambdaFunctionMatch(func=ce_v_max_conditions)

matchers['ce_v_max_rgx'] = ce_v_max_rgx_matcher


matchers['part'] = Intersect(part_rgx_matcher, part_lambda_matcher)
matchers['stg_temp_max'] = RegexMatchSpan(rgx=r'(?:[1][5-9]|20)[05]', longest_match_only=False)
matchers['stg_temp_min'] = RegexMatchSpan(rgx=r'-[56][05]', longest_match_only=False)
matchers['polarity'] = Intersect(polarity_rgx_matcher, polarity_lambda_matcher)
matchers['ce_v_max'] = Intersect(ce_v_max_rgx_matcher, attr_in_table_matcher, ce_v_max_row_matcher)

### GETTER ###

def get_digikey_parts_set(path):
    """
    Reads in the digikey part dictionary and yeilds each part.
    """
    all_parts = set()
    with open(path, "r") as csvinput:
        reader = csv.reader(csvinput)
        for line in reader:
            try:
                (part, url) = line
            except:
                import pdb; pdb.set_trace()
            all_parts.add(part)
    return all_parts

def get_matcher(attr, dict_path=None):
    if attr.startswith("part"):
        if dict_path:
            # If no path is provided, just get the normal parts matcher
            parts_dict_matcher = DictionaryMatch(d=get_digikey_parts_set(dict_path))
            combined_matcher = Union(parts_dict_matcher, matchers[attr])
            print "Using combined matcher."
            return combined_matcher
    return matchers[attr]
