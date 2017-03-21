from snorkel.matchers import *
from snorkel.lf_helpers import *
import csv
matchers = {}


### PART ###
eeca_rgx = '([ABC][A-Z][WXYZ]?[0-9]{3,5}(?:[A-Z]){0,5}[0-9]?[A-Z]?(?:-[A-Z0-9]{1,7})?(?:[-][A-Z0-9]{1,2})?(?:\/DG)?)'
jedec_rgx = '(2N\d{3,4}[A-Z]{0,5}[0-9]?[A-Z]?)'
jis_rgx = '(2S[ABCDEFGHJKMQRSTVZ]{1}[\d]{2,4})'
others_rgx = '((?:NSVBC|SMBT|MJ|MJE|MPS|MRF|RCA|TIP|ZTX|ZT|ZXT|TIS|TIPL|DTC|MMBT|SMMBT|PZT|FZT|STD|BUV|PBSS|KSC|CXT|FCX|CMPT){1}[\d]{2,4}[A-Z]{0,5}(?:-[A-Z0-9]{0,6})?(?:[-][A-Z0-9]{0,1})?)'

add_rgx = '^[A-Z0-9\-]{5,15}$'

part_rgx = '|'.join([eeca_rgx, jedec_rgx, jis_rgx, others_rgx])
# modifiers = '(?:[\/\-][A-Z]{,2})*'
# part_rgx = '(' + '|'.join([eeca_rgx, jedec_rgx, jis_rgx, others_rgx]) + ')' + modifiers
part_rgx_matcher = RegexMatchSpan(rgx=part_rgx, longest_match_only=True)
matchers['part_rgx'] = part_rgx_matcher

# def part_conditions(part):
#     """throttle parts that are in tables of device/replacement parts"""
#     aligned_ngrams = set(get_aligned_ngrams(part))
#     return not (overlap(['replacement'], aligned_ngrams) or
#         (len(aligned_ngrams) > 25 and 'device' in aligned_ngrams) or
#         get_prev_sibling_tags(part).count('p') > 125 or # CentralSemiconductorCorp_2N4013.pdf:
#         overlap(['complementary', 'complement', 'empfohlene'], 
#                 chain.from_iterable([
#                     get_left_ngrams(part, window=10),
#                     get_aligned_ngrams(part)])))
def part_conditions(part):
    col_ngrams = set(get_col_ngrams(part))
    return not (overlap(['replacement'], col_ngrams) or
        (len(col_ngrams) > 25 and 'device' in col_ngrams) or 
        ('top' in col_ngrams and 'mark' in col_ngrams) or
        get_prev_sibling_tags(part).count('p') > 125 or
        overlap(['complement','complementary', 'empfohlene'], 
                         chain.from_iterable([
                             get_left_ngrams(part, window=10),
                             get_aligned_ngrams(part),
                             get_neighbor_phrase_ngrams(part)])) or
        'please' in get_left_ngrams(part, window=99) or 
        get_max_col_num(part) > 4)
part_filter= LambdaFunctionMatch(func=part_conditions)

def common_prefix_length_diff(str1, str2):
    for i in range(min(len(str1), len(str2))):
        if str1[i] != str2[i]:
            return min(len(str1), len(str2)) - i
    return 0

def part_file_name_conditions(attr):
    file_name = attr.parent.document.name
    if len(file_name.split('_')) != 2: return False
    if attr.get_span()[0] == '-': return False
    name = attr.get_span().replace('-', '')
    return any(char.isdigit() for char in name) and any(char.isalpha() for char in name) and common_prefix_length_diff(file_name.split('_')[1], name) <= 2

spart_file_name_lambda_matcher = LambdaFunctionMatch(func=part_file_name_conditions)
part_file_name_matcher = Intersect(RegexMatchSpan(rgx=add_rgx, longest_match_only=True), spart_file_name_lambda_matcher)


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

def ce_v_max_more_conditions1(attr):
    text = attr.parent.text
    if attr.char_start > 1 and text[attr.char_start - 1] == '-' and text[attr.char_start - 2] not in [' ', '='] : return False
    return True

def ce_v_max_more_conditions(attr):
    text = attr.parent.text
    if attr.char_start !=0 and text[attr.char_start - 1] == '/': return False
    if attr.char_start > 1 and text[attr.char_start - 1] == '-' and text[attr.char_start - 2] not in [' ', '='] : return False
    if 'vcb' in attr.parent.text.lower(): return False
    for i in range(attr.char_end + 1, len(text)):
        if text[i] == ' ': continue
        if text[i].isdigit(): break
        if text[i].upper() != 'V': return False
        else: break
    return True

ce_v_whole_number = LambdaFunctionMatch(func=ce_v_max_more_conditions)

matchers['ce_v_max_rgx'] = ce_v_max_rgx_matcher

matchers['stg_temp_max'] = RegexMatchSpan(rgx=r'(?:[1][5-9]|20)[05]', longest_match_only=False)
matchers['stg_temp_min'] = RegexMatchSpan(rgx=r'-[56][05]', longest_match_only=False)
matchers['polarity'] = polarity_rgx_matcher
#matchers['polarity'] = Intersect(polarity_rgx_matcher, polarity_lambda_matcher)
matchers['ce_v_max'] = Intersect(ce_v_max_rgx_matcher, attr_in_table_matcher, ce_v_max_row_matcher, ce_v_whole_number)

### GETTER ###

def get_digikey_parts_set(path):
    """
    Reads in the digikey part dictionary and yeilds each part.
    """
    all_parts = set()
    with open(path, "r") as csvinput:
        reader = csv.reader(csvinput)
        for line in reader:
            (part, url) = line
            all_parts.add(part)
    return all_parts

def get_matcher(attr, dict_path=None):
    if dict_path:
        # If no path is provided, just get the normal parts matcher
        part_dict_matcher = DictionaryMatch(d=get_digikey_parts_set(dict_path))
        part_matchers = Union(part_rgx_matcher, part_dict_matcher, part_file_name_matcher)
        print "Using combined matcher."
    else:
        part_matchers = Union(part_rgx_matcher, part_file_name_matcher)
    matchers['part'] = Intersect(part_matchers, part_filter)
    return matchers[attr]
