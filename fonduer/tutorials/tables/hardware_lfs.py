from snorkel.lf_helpers import *
from hardware_matchers import get_matcher
from hardware_spaces import expand_part_range
import re
from itertools import chain
from random import random

### PART ###

def LF_part_in_header_tag(c):
    return 1 if get_tag(c.part).startswith('h') else 0

parts_rgx = get_matcher('part_rgx').rgx
part_sniffer = re.compile(parts_rgx)
def LF_cheating_with_another_part(c):
    return -1 if (any(part_sniffer.match(x) for x in get_horz_ngrams(c.attr)) and 
                     not is_horz_aligned(c)) else 0

def LF_replacement_table(c):
    col_ngrams = list(get_col_ngrams(c.part))
    return -1 if (overlap(['replacement'], col_ngrams) or
                 (len(col_ngrams) > 25 and 'device' in col_ngrams)) else 0

def LF_many_p_siblings(c):
    # e.g., CentralSemiconductorCorp_2N4013.pdf
    return -1 if get_prev_sibling_tags(c.part).count('p') > 125 else 0

def LF_part_complement(c):
    return -1 if overlap(['complement','complementary', 'empfohlene'], 
                         chain.from_iterable([
                             get_left_ngrams(c.part, window=10),
                             get_aligned_ngrams(c.part),
                             get_neighbor_phrase_ngrams(c.part)])) else 0

# def LF_top_mark_col_part(c):
#     return -1 if overlap(['top','mark','marking'],
#                          get_col_ngrams(c.part)) else 0

def LF_please_to_left(c):
    # e.g., DiodesIncorporated_FZT651TC
    return -1 if 'please' in get_left_ngrams(c.part, window=99) else 0

def LF_part_num_in_high_col_num(c):
    return -1 if get_max_col_num(c.part) > 4 else 0


part_lfs = [
#    LF_part_in_header_tag, for ce_v_max
#    LF_cheating_with_another_part,
    LF_replacement_table,
    LF_many_p_siblings,
    LF_part_complement,
    # LF_top_mark_col_part,
    LF_please_to_left
#    LF_part_num_in_high_col_num
]

ce_v_max_part_lfs = [
#    LF_part_in_header_tag, for ce_v_max
    LF_cheating_with_another_part,
    LF_replacement_table,
    LF_many_p_siblings,
    LF_part_complement,
    # LF_top_mark_col_part,
    LF_please_to_left
#    LF_part_num_in_high_col_num
]

### POLARITY ###

# def polarity_random():
#     return int(random() < 0.2)

def LF_default_positive(c):
    return 1 

def LF_polarity_part_tabular_align(c):
    return 1 if same_row(c) or same_col(c) else 0

def LF_polarity_part_horz_align(c):
    return 1 if is_horz_aligned(c) else 0

def LF_polarity_part_vert_align(c):
    return 1 if is_vert_aligned(c) else 0

def LF_both_in_top_third(c):
    return 1 if (get_page(c.part) == 1 and 
                 get_page(c.attr) == 1 and 
                 get_page_vert_percentile(c.part) > 0.33 and 
                 get_page_vert_percentile(c.attr) > 0.33) else 0

def LF_polarity_description(c):
    aligned_ngrams = set(get_aligned_ngrams(c.attr))
    return 1 if overlap(['description', 'polarity'], aligned_ngrams) else 0

def LF_polarity_transistor_type(c):
    return 1 if overlap(['silicon','power', 'darlington', 'epitaxial', 'low noise', 'ampl/switch', 'switch', 'surface', 'mount'], 
                         chain.from_iterable([
                             get_phrase_ngrams(c.attr), 
                             get_neighbor_phrase_ngrams(c.attr)])) else 0

def LF_polarity_right_of_part(c):
    right_ngrams = set(get_right_ngrams(c.part, lower=False))
    return 1 if ((c.attr.get_span()=='NPN' and 'NPN' in right_ngrams) or (c.attr.get_span()=='PNP' and 'PNP' in right_ngrams)) else 0

def LF_polarity_in_header_tag(c):
    return 1 if get_tag(c.attr).startswith('h') else 0

def LF_polarity_complement(c):
    return -1 if overlap(['complement','complementary'], 
                         chain.from_iterable([
                             get_phrase_ngrams(c.attr), 
                             get_neighbor_phrase_ngrams(c.attr)])) else 1

def LF_cheating_with_another_polarity(c):
    return -1 if ((c.attr.get_span()=='NPN' and 'PNP' in get_horz_ngrams(c.part, lower=False)) or
                  (c.attr.get_span()=='PNP' and 'NPN' in get_horz_ngrams(c.part, lower=False))) else 0

def LF_both_present(c):
    phrase_ngrams = set(get_phrase_ngrams(c.attr))
    return -1 if ('npn' in phrase_ngrams and 'pnp' in phrase_ngrams) else 0

def filter_non_polarity(c):
    ret = set()
    for _ in c:
        if re.match(r'NPN|PNP', _):
            ret.add(_)
    return ret

def LF_part_miss_match_part(c):
    if not c[1].is_tabular(): return 0
#    ngrams_part = set(list(get_col_ngrams(c[1], n_max=1)))
    ngrams_part = set(list(get_row_ngrams(c[1], n_max=1, lower=False)))
#    print '~~', ngrams_part
    ngrams_part = filter_non_parts(ngrams_part)
#    print '~~', ngrams_part
    return 0 if len(ngrams_part) == 0 or any([c[0].get_span().lower().startswith(_.lower()) for _ in ngrams_part]) else -1


def LF_part_miss_match_polarity(c):
#    ngrams_part = set(list(get_col_ngrams(c[0], n_max=1)))
    ngrams_part = set(list(get_row_ngrams(c[0], n_max=1, lower=False)))
#    print '!!', ngrams_part
    ngrams_part = filter_non_polarity(ngrams_part)
#    print '!!', ngrams_part
    return 0 if len(ngrams_part) == 0 or any([c[1].get_span().lower().startswith(_.lower()) for _ in ngrams_part]) else -1


polarity_lfs = [
    # LF_default_positive,
#    LF_polarity_description,
    LF_polarity_transistor_type,
    LF_polarity_part_tabular_align,
    LF_polarity_part_horz_align,
    LF_polarity_part_vert_align,
    LF_polarity_right_of_part,
    # LF_both_in_top_third,
    LF_polarity_in_header_tag,
    LF_polarity_complement,
    LF_both_present
    # LF_cheating_with_another_polarity
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
    LF_positive_number_right,
    LF_other_minus_signs_in_row
]

### VOLTAGE ###

def LF_aligned_or_global(c):
    return 1 if (same_row(c) or
                 is_horz_aligned(c) or
                 not c.part.is_tabular()) else -1

def LF_same_table_must_align(c):
    return -1 if (same_table(c) and not (is_horz_aligned(c) or is_vert_aligned(c))) else 0

def LF_voltage_not_in_table(c):
    return -1 if c.attr.parent.table is None else 0

def LF_low_table_num(c):
    return -1 if (c.attr.parent.table and
        c.attr.parent.table.position > 2) else 0

bad_keywords = set(['continuous', 'cut-off', 'gain', 'breakdown'])
def LF_bad_keywords_in_row(c):
    return -1 if overlap(bad_keywords, get_row_ngrams(c.attr)) else 0

def LF_equals_in_row(c):
    return -1 if overlap('=', get_row_ngrams(c.attr)) else 0

def LF_current_in_row(c):
    return -1 if overlap(['i', 'ic', 'mA'], get_row_ngrams(c.attr)) else 0

def LF_V_aligned(c):
    return 1 if overlap('V', get_aligned_ngrams(c, lower=False)) else 0

def LF_too_many_numbers_horz(c):
    num_numbers = list(get_horz_ngrams(c.attr, attrib="ner_tags")).count('number')
    return -1 if num_numbers > 3 else 0

voltage_lfs = [
#    LF_aligned_or_global,
#    LF_same_table_must_align,
#    LF_low_table_num,
    LF_voltage_not_in_table,
    LF_bad_keywords_in_row,
#    LF_equals_in_row,
    LF_current_in_row,
#    LF_V_aligned,
#    LF_too_many_numbers_horz
]

# CE_V_MAX #
ce_keywords = set(['collector emitter', 'collector-emitter', 'collector - emitter'])
def LF_ce_keywords_in_row(c):
    return 1 if overlap(ce_keywords, get_row_ngrams(c.attr, n_max=3)) else 0

def LF_ce_keywords_horz(c):
    return 1 if overlap(ce_keywords, get_horz_ngrams(c.attr)) else 0

ce_abbrevs = set(['ceo', 'vceo']) # 'value', 'rating'
def LF_ce_abbrevs_in_row(c):
    return 1 if overlap(ce_abbrevs, get_row_ngrams(c.attr)) else 0

def LF_ce_abbrevs_horz(c):
    return 1 if overlap(ce_abbrevs, get_horz_ngrams(c.attr)) else 0

def LF_head_ends_with_ceo(c):
    return 1 if any(ngram.endswith('ceo') for ngram in get_head_ngrams(c.attr)) else 0

non_ce_voltage_keywords = set(['collector-base', 'collector - base', 'collector base', 'vcbo', 'cbo', 'vces',
                               'emitter-base', 'emitter - base', 'emitter base', 'vebo', 'ebo', 'breakdown voltage', 
                               'emitter breakdown', 'emitter breakdown voltage', 'current'])
def LF_non_ce_voltages_in_row(c):
    return -1 if overlap(non_ce_voltage_keywords, get_row_ngrams(c.attr, n_max=3)) else 0

def LF_first_two_pages(c):
    return 1 if get_page(c) in [1, 2] else -1

def filter_non_parts(c):
    ret = set()
    for _ in c:
        for __ in expand_part_range(_):
            if re.match("^([0-9]+[A-Z]+|[A-Z]+[0-9]+)[0-9A-Z]*$", __) and len(__) > 2:
                ret.add(__)
    return ret

def LF_part_ce_keywords_in_rows(c):
    return 1 if overlap(ce_keywords.union(ce_abbrevs), get_row_ngrams(c.attr, n_max=3)) and \
        overlap(set([c[0].get_span().lower()]), get_row_ngrams(c.attr, n_max=3)) else 0 

def LF_part_ce_keywords_in_rows_cols_prefix(c):
    ngrams = set(list(get_row_ngrams(c.attr, n_max=3)))
    ngrams = ngrams.union(set(list(get_col_ngrams(c.attr, n_max=3))))
    ngrams_part = filter_non_parts(ngrams)
#    print ngrams
    return 1 if overlap(ce_keywords.union(ce_abbrevs), ngrams) and \
        any([c[0].get_span().lower().startswith(_) for _ in ngrams_part]) else 0 

def LF_part_ce_keywords_in_rows_cols_prefix_1(c):
    ngrams = set(list(get_horz_ngrams(c.attr)))
    ngrams = ngrams.union(set(list(get_vert_ngrams(c.attr))))
    ngrams_part = filter_non_parts(ngrams)
#    print ngrams
    return 1 if overlap(ce_keywords.union(ce_abbrevs), ngrams) and \
        any([c[0].get_span().lower().startswith(_) for _ in ngrams_part]) else 0 
    
def LF_part_ce_keywords_horz(c):
    return 1 if overlap(ce_keywords.union(ce_abbrevs), get_horz_ngrams(c.attr)) and \
        overlap(set([c[0].get_span().lower()]), get_horz_ngrams(c.attr)) else 0 

def LF_part_ce_keywords_horz_prefix(c):
    return 1 if overlap(ce_keywords.union(ce_abbrevs), get_horz_ngrams(c.attr)) and \
        any([c[0].get_span().lower().startswith(_) for _ in get_horz_ngrams(c.attr)]) and \
        not overlap(non_ce_voltage_keywords, get_horz_ngrams(c.attr)) else 0

def LF_part_ce_keywords_in_row_prefix(c):
    ngrams_part = filter_non_parts(get_row_ngrams(c.attr, n_max=3))

    return 1 if overlap(ce_keywords.union(ce_abbrevs), get_row_ngrams(c.attr, n_max=3)) and \
        any([c[0].get_span().lower().startswith(_) for _ in ngrams_part]) and \
        not overlap(non_ce_voltage_keywords, get_row_ngrams(c.attr, n_max=3)) and \
        not LF_current_in_row(c) else 0

def LF_part_ce_keywords_in_row_prefix_same_table(c):
    ngrams_part = filter_non_parts(get_row_ngrams(c.attr, n_max=3))

    return 1 if same_table(c) and \
        is_horz_aligned(c) and \
        overlap(ce_keywords.union(ce_abbrevs), get_row_ngrams(c.attr, n_max=3)) and \
        overlap(ce_keywords.union(ce_abbrevs), get_row_ngrams(c.part, n_max=3)) and \
        any([c[0].get_span().lower().startswith(_) for _ in ngrams_part]) and \
        not overlap(non_ce_voltage_keywords, get_row_ngrams(c.part, n_max=3)) and \
        not overlap(non_ce_voltage_keywords, get_row_ngrams(c.attr, n_max=3)) and \
        not LF_current_in_row(c) else 0

def LF_part_ce_keywords_in_col_prefix_same_table(c):
    ngrams_part = filter_non_parts(get_col_ngrams(c.attr, n_max=3))
    
    return 1 if same_table(c) and \
        is_vert_aligned(c) and \
        overlap(ce_keywords.union(ce_abbrevs), get_row_ngrams(c.attr, n_max=3)) and \
        not overlap(non_ce_voltage_keywords, get_row_ngrams(c.attr, n_max=3)) and \
        not LF_current_in_row(c) and \
        bbox_from_span(c.part).top < bbox_from_span(c.attr).top else 0
        
def LF_ce_keywords_not_part_in_row_col_prefix(c):
    ngrams_part = set(list(get_col_ngrams(c.attr, n_max=3)))
    ngrams_part = filter_non_parts(ngrams_part.union(set(list(get_row_ngrams(c.attr, n_max=3)))))

    return 1 if not same_table(c) and \
        overlap(ce_keywords.union(ce_abbrevs), get_row_ngrams(c.attr, n_max=3)) and \
        len(ngrams_part) == 0 and \
        not overlap(non_ce_voltage_keywords, get_row_ngrams(c.part, n_max=3)) and \
        not overlap(non_ce_voltage_keywords, get_row_ngrams(c.attr, n_max=3)) and \
        not LF_current_in_row(c) else 0

       
def LF_part_miss_match(c):
    ngrams_part = set(list(get_vert_ngrams(c[1], n_max=1)))
    ngrams_part = filter_non_parts(ngrams_part.union(set(list(get_horz_ngrams(c[1], n_max=1)))))
#    print '~~', ngrams_part
    return 0 if len(ngrams_part) == 0 or any([c[0].get_span().lower().startswith(_.lower()) for _ in ngrams_part]) else -1

        
def LF_not_valid_value(c):
#    ngrams_part = set(list(get_col_ngrams(c.attr, n_max=3)))
#    ngrams_part = filter_non_parts(ngrams_part.union(set(list(get_row_ngrams(c.attr, n_max=3)))))

    return -1 if not overlap(ce_keywords.union(ce_abbrevs), get_row_ngrams(c.attr, n_max=3)) else 0
        
def LF_ce_keywords_no_part_in_rows(c):
    for _ in get_row_ngrams(c.attr, n_max=3):
        if re.match("^([0-9]+[a-zA-Z]+|[a-zA-Z]+[0-9]+)[0-9a-zA-Z]*$", _):
            return 0
    return 1 if overlap(ce_keywords.union(ce_abbrevs), get_row_ngrams(c.attr, n_max=3)) else 0 

def LF_ce_keywords_no_part_horz(c):
    for _ in get_horz_ngrams(c.attr):
        if re.match("^([0-9]+[a-zA-Z]+|[a-zA-Z]+[0-9]+)[0-9a-zA-Z]*$", _):
            return 0
    return 1 if overlap(ce_keywords.union(ce_abbrevs), get_horz_ngrams(c.attr)) else 0 

ce_v_max_lfs = voltage_lfs + [
#    LF_ce_keywords_in_row,
#    LF_ce_keywords_horz,
#    LF_ce_abbrevs_in_row,
#    LF_ce_abbrevs_horz,
#    LF_head_ends_with_ceo,
    LF_non_ce_voltages_in_row,
#    LF_first_two_pages
#    LF_part_ce_keywords_in_rows_cols_prefix,
    LF_part_ce_keywords_in_row_prefix_same_table,
    LF_part_ce_keywords_in_col_prefix_same_table,
    LF_part_miss_match
#    LF_part_ce_keywords_in_row_prefix,
#    LF_ce_keywords_not_part_in_row_col_prefix,
#    LF_part_ce_keywords_horz_prefix
#    LF_not_valid_value
#    LF_ce_keywords_no_part_in_rows,
#    LF_ce_keywords_no_part_horz
]

### GETTER ###

def get_lfs(attr):
    if attr=='part':
        attr_lfs = []
    elif attr == ('stg_temp_max'):
        attr_lfs = stg_temp_max_lfs
    elif attr == ('stg_temp_min'):
        attr_lfs = stg_temp_min_lfs
    elif attr == ('polarity'):
        attr_lfs = polarity_lfs
    elif attr == ('ce_v_max'):
        attr_lfs = ce_v_max_lfs
    if attr == ('ce_v_max'):
        return ce_v_max_part_lfs + attr_lfs
    return part_lfs + attr_lfs