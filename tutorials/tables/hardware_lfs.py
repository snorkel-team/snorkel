from snorkel.lf_helpers import *
from hardware_matchers import get_matcher
import re
from itertools import chain

### PART ###
def LF_replacement_table(c):
    aligned_ngrams = list(get_aligned_ngrams(c.part))
    return -1 if (overlap(['replacement'], aligned_ngrams) or
                 (len(aligned_ngrams) > 25 and 'device' in aligned_ngrams)) else 0

def LF_many_p_siblings(c):
    return -1 if get_prev_sibling_tags(c.part).count('p') > 25 else 0

def LF_part_complement(c):
    return -1 if overlap(['complement','complementary', 'empfohlene'], 
                         chain.from_iterable([
                             get_left_ngrams(c.part, window=10),
                             get_aligned_ngrams(c.part),
                             get_neighbor_phrase_ngrams(c.part)])) else 0

def LF_top_mark_col_part(c):
    return -1 if overlap(['top','mark','marking'],
                         get_col_ngrams(c.part)) else 0

def LF_please_to_left(c):
    # e.g., DiodesIncorporated_FZT651TC
    return -1 if 'please' in get_left_ngrams(c.part, window=7) else 0

def LF_part_num_in_high_col_num(c):
    return -1 if get_max_col_num(c.part) > 4 else 0

part_lfs = [
    LF_replacement_table,
    LF_many_p_siblings,
    LF_part_complement,
    LF_top_mark_col_part,
    LF_please_to_left,
    LF_part_num_in_high_col_num
]

### POLARITY ###
def LF_default_positive(c):
    return 1 

def LF_polarity_complement(c):
    return -1 if overlap(['complement','complementary'], 
                         get_phrase_ngrams(c.attr)) else 0

def LF_polarity_complement_neighbor(c):
    return -1 if overlap(['complement','complementary'], 
                         get_neighbor_phrase_ngrams(c.attr)) else 0

polarity_lfs = [
    LF_default_positive,
    LF_polarity_complement,
    LF_polarity_complement_neighbor
]


### TEMPERATURE ###

def LF_storage_row(c):
    return 1 if 'storage' in get_row_ngrams(c.attr) else 0

def LF_operating_row(c):
    return 1 if 'operating' in get_row_ngrams(c.attr) else 0

def LF_temperature_row(c):
    return 1 if 'temperature' in get_row_ngrams(c.attr) else 0

def LF_tstg_row(c):
    return 1 if overlap(
        ['tstg','stg','ts'], 
        list(get_row_ngrams(c.attr))) else 0

def LF_not_temp_relevant(c):
    return -1 if not overlap(
        ['storage','temperature','tstg','stg', 'ts'],
        list(get_aligned_ngrams(c.attr))) else 0

def LF_temp_outside_table(c):
    return -1 if not c.attr.is_tabular() is None else 0

def LF_too_many_numbers_row(c):
    num_numbers = list(get_row_ngrams(c.attr, attrib="ner_tags")).count('number')
    return -1 if num_numbers >= 3 else 0

def LF_collector_aligned(c):
    return -1 if overlap(
        ['collector', 'collector-current', 'collector-base', 'collector-emitter'],
        list(get_aligned_ngrams(c.attr))) else 0

def LF_current_aligned(c):
    ngrams = get_aligned_ngrams(c.attr)
    return -1 if overlap(
        ['current', 'dc', 'ic'],
        list(get_aligned_ngrams(c.attr))) else 0

def LF_voltage_row_temp(c):
    ngrams = get_aligned_ngrams(c.attr)
    return -1 if overlap(
        ['voltage', 'cbo', 'ceo', 'ebo', 'v'],
        list(get_aligned_ngrams(c.attr))) else 0

def LF_voltage_row_part(c):
    ngrams = get_aligned_ngrams(c.part)
    return -1 if overlap(
        ['voltage', 'cbo', 'ceo', 'ebo', 'v'],
        list(get_aligned_ngrams(c.attr))) else 0

def LF_typ_row(c):
    return -1 if overlap(
        ['typ', 'typ.'],
        list(get_row_ngrams(c.attr))) else 0

def LF_test_condition_aligned(c):
    return -1 if overlap(
        ['test', 'condition'],
        list(get_aligned_ngrams(c.attr))) else 0

def LF_complement_left_row(c):
    return -1 if (
        overlap(['complement','complementary'], 
        chain.from_iterable([get_row_ngrams(c.part), get_left_ngrams(c.part, window=10)]))) else 0

def LF_temp_on_high_page_num(c):
    return -1 if c.attr.get_attrib_tokens('page')[0] > 2 else 0

stg_temp_lfs = [
    LF_storage_row,
    LF_operating_row,
    LF_temperature_row,
    LF_tstg_row,
    LF_not_temp_relevant,
    LF_temp_outside_table,
    LF_too_many_numbers_row,
    LF_collector_aligned,
    LF_current_aligned,
    LF_voltage_row_temp,
    LF_voltage_row_part,
    LF_typ_row,
    LF_test_condition_aligned,
    LF_complement_left_row,
    LF_temp_on_high_page_num
]

# STG_TEMP_MAX #

def LF_to_left(c):
    return 1 if 'to' in get_left_ngrams(c.attr, window=2) else 0

def LF_negative_number_left(c):
    return 1 if any([re.match(r'-\s*\d+', ngram) for ngram in get_left_ngrams(c.attr, window=4)]) else 0


stg_temp_max_lfs = stg_temp_lfs + [
    LF_to_left,
    LF_negative_number_left
]

# STG_TEMP_MIN #

def LF_to_right(c):
    return 1 if 'to' in get_right_ngrams(c.attr, window=2) else 0

def LF_positive_number_right(c):
    return 1 if any([re.match(r'\d+', ngram) for ngram in get_right_ngrams(c.attr, window=4)]) else 0

def LF_other_minus_signs_in_row(c):
    return -1 if '-' in get_row_ngrams(c.attr) else 0

stg_temp_min_lfs = stg_temp_lfs + [
    LF_to_right,
    LF_positive_number_right
]

### VOLTAGE ###

def LF_aligned_or_global(c):
    return 1 if (same_row(c) or
                 is_horz_aligned(c) or
                 not c.part.is_tabular()) else -1

parts_rgx = get_matcher('part').rgx
part_sniffer = re.compile(parts_rgx)
def LF_cheating_with_another_part(c):
    return -1 if (any(part_sniffer.match(x) for x in get_horz_aligned_ngrams(c.attr)) and 
                     not is_horz_aligned(c)) else 0

def LF_same_table_must_align(c):
    return -1 if (same_table(c) and not is_horz_aligned(c)) else 0

def LF_voltage_not_in_table(c):
    return -1 if c.attr.parent.table is None else 0

def LF_low_table_num(c):
    return -1 if (c.attr.parent.table and
        c.attr.parent.table.position > 2) else 0

bad_keywords = set(['continuous', 'cut-off', 'gain'])
def LF_bad_keywords_in_row(c):
    return -1 if overlap(bad_keywords, get_row_ngrams(c.attr)) else 0

def LF_equals_in_row(c):
    return -1 if overlap('=', get_row_ngrams(c.attr)) else 0

def LF_i_in_row(c):
    return -1 if overlap('i', get_row_ngrams(c.attr)) else 0

def LF_too_many_numbers_row(c):
    num_numbers = list(get_row_ngrams(c.attr, attrib="ner_tags")).count('number')
    return -1 if num_numbers >= 4 else 0

voltage_lfs = [
    LF_aligned_or_global,
    LF_cheating_with_another_part,
    LF_same_table_must_align,
    LF_low_table_num,
    LF_voltage_not_in_table,
    LF_bad_keywords_in_row,
    LF_equals_in_row,
    LF_i_in_row,
    LF_too_many_numbers_row
]

# CE_V_MAX #
ce_keywords = set(['collector emitter', 'collector-emitter', 'collector - emitter'])
def LF_ce_keywords_in_row(c):
    return 1 if overlap(ce_keywords, get_row_ngrams(c.attr, spread=[0,3], n_max=3)) else -1

def LF_ce_keywords_horz(c):
    return 1 if overlap(ce_keywords, get_horz_aligned_ngrams(c.attr)) else 0

ce_abbrevs = set(['ceo', 'vceo']) # 'value', 'rating'
def LF_ce_abbrevs_in_row(c):
    return 1 if overlap(ce_abbrevs, get_row_ngrams(c.attr, spread=[0,3])) else 0

def LF_ce_abbrevs_horz(c):
    return 1 if overlap(ce_abbrevs, get_horz_aligned_ngrams(c.attr)) else 0

non_ce_voltage_keywords = set(['collector-base', 'collector - base', 'emitter-base', 'emitter - base'])
def LF_non_ce_voltages_in_row(c):
    return -1 if overlap(non_ce_voltage_keywords, get_row_ngrams(c.attr, spread=[0,3], n_max=3)) else 0

ce_v_max_lfs = voltage_lfs + [
    LF_ce_keywords_in_row,
    LF_ce_keywords_horz,
    LF_ce_abbrevs_in_row,
    LF_ce_abbrevs_horz,
    LF_non_ce_voltages_in_row
]

### GETTER ###

def get_lfs(attr):
    if attr == 'stg_temp_max':
        attr_lfs = stg_temp_max_lfs
    elif attr == 'stg_temp_min':
        attr_lfs = stg_temp_min_lfs
    elif attr == 'polarity':
        attr_lfs = polarity_lfs
    elif attr == 'ce_v_max':
        attr_lfs = ce_v_max_lfs
    return part_lfs + attr_lfs