from snorkel.matchers import *

matchers = {}

eeca_rgx = '([ABC][A-Z][WXYZ]?[0-9]{3,5}(?:[A-Z]){0,5}[0-9]?[A-Z]?(?:-[A-Z0-9]{1,7})?(?:[-][A-Z0-9]{1,2})?(?:\/DG)?)'
jedec_rgx = '(2N\d{3,4}[A-Z]{0,5}[0-9]?[A-Z]?)'
jis_rgx = '(2S[ABCDEFGHJKMQRSTVZ]{1}[\d]{2,4})'
others_rgx = '((?:NSVBC|SMBT|MJ|MJE|MPS|MRF|RCA|TIP|ZTX|ZT|ZXT|TIS|TIPL|DTC|MMBT|SMMBT|PZT|FZT){1}[\d]{2,4}[A-Z]{0,3}(?:-[A-Z0-9]{0,6})?(?:[-][A-Z0-9]{0,1})?)'
part_rgx = '|'.join([eeca_rgx, jedec_rgx, jis_rgx, others_rgx])

matchers['part'] = RegexMatchSpan(rgx=part_rgx, longest_match_only=False)
matchers['stg_temp_max'] = RegexMatchSpan(rgx=r'(?:[1][5-9]|20)[05]', longest_match_only=False)
matchers['stg_temp_min'] = RegexMatchSpan(rgx=r'-[56][05]', longest_match_only=False)
matchers['polarity'] = RegexMatchSpan(rgx=r'NPN|PNP', longest_match_only=False, ignore_case=True)
matchers['ce_v_max'] = RegexMatchSpan(rgx=r'\d{1,2}[05]', longest_match_only=False)

def get_matcher(attr):
    for a in ['part', 'stg_temp_max', 'stg_temp_min', 'polarity', 'ce_v_max']:
        if attr.startswith(a):
            return matchers[a]
    raise ValueError('Attribute name is invalid.')