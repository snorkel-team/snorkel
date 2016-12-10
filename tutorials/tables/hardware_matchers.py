from snorkel.matchers import *

matchers = {}

eeca_matcher = RegexMatchSpan(rgx='([ABC][A-Z][WXYZ]?[0-9]{3,5}(?:[A-Z]){0,5}[0-9]?[A-Z]?(?:-[A-Z0-9]{1,7})?(?:[-][A-Z0-9]{1,2})?(?:\/DG)?)')
jedec_matcher = RegexMatchSpan(rgx='(2N\d{3,4}[A-Z]{0,5}[0-9]?[A-Z]?)')
jis_matcher = RegexMatchSpan(rgx='(2S[ABCDEFGHJKMQRSTVZ]{1}[\d]{2,4})')
others_matcher = RegexMatchSpan(rgx='((?:NSVBC|SMBT|MJ|MJE|MPS|MRF|RCA|TIP|ZTX|ZT|ZXT|TIS|TIPL|DTC|MMBT|SMMBT|PZT|FZT){1}[\d]{2,4}[A-Z]{0,3}(?:-[A-Z0-9]{0,6})?(?:[-][A-Z0-9]{0,1})?)')

matchers['part'] = Union(eeca_matcher, jedec_matcher, jis_matcher, others_matcher)
matchers['stg_temp_max'] = RegexMatchSpan(rgx=r'(?:[1][5-9]|20)[05]', longest_match_only=False)
matchers['ce_v_max'] = RegexMatchSpan(rgx=r'1?\d(?:\.0)?', longest_match_only=False)

def get_matcher(attr):
    return matchers[attr]