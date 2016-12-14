from snorkel.matchers import *
import csv
matchers = {}

eeca_rgx = '([ABC][A-Z][WXYZ]?[0-9]{3,5}(?:[A-Z]){0,5}[0-9]?[A-Z]?(?:-[A-Z0-9]{1,7})?(?:[-][A-Z0-9]{1,2})?(?:\/DG)?)'
jedec_rgx = '(2N\d{3,4}[A-Z]{0,5}[0-9]?[A-Z]?)'
jis_rgx = '(2S[ABCDEFGHJKMQRSTVZ]{1}[\d]{2,4})'
others_rgx = '((?:NSVBC|SMBT|MJ|MJE|MPS|MRF|RCA|TIP|ZTX|ZT|ZXT|TIS|STD|BUV|TIPL|DTC|MMBT|SMMBT|PZT|FZT){1}[\d]{2,4}[A-Z]{0,3}(?:-[A-Z0-9]{0,6})?(?:[-][A-Z0-9]{0,1})?)'
part_rgx = '|'.join([eeca_rgx, jedec_rgx, jis_rgx, others_rgx])
# modifiers = '(?:[\/\-][A-Z]{,2})*'
# part_rgx = '(' + '|'.join([eeca_rgx, jedec_rgx, jis_rgx, others_rgx]) + ')' + modifiers

matchers['part'] = RegexMatchSpan(rgx=part_rgx, longest_match_only=False)
matchers['stg_temp_max'] = RegexMatchSpan(rgx=r'(?:[1][5-9]|20)[05]', longest_match_only=False)
matchers['stg_temp_min'] = RegexMatchSpan(rgx=r'-[56][05]', longest_match_only=False)
matchers['polarity'] = RegexMatchSpan(rgx=r'NPN|PNP', longest_match_only=False, ignore_case=True)
matchers['ce_v_max'] = RegexMatchSpan(rgx=r'\d{1,2}[05]', longest_match_only=False)

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
    if attr == "part":
        if dict_path:
            # If no path is provided, just get the normal parts matcher
            parts_dict = DictionaryMatch(d=get_digikey_parts_set(dict_path))
            combined_matcher = Union(parts_dict, matchers[attr])
            print "Using combined matcher."
            return combined_matcher
    return matchers[attr]
