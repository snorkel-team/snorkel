from hardware_utils import expand_implicit_text
from copy import deepcopy

class Expander(object):
    def apply(self, tc):
        raise NotImplementedError

class IdentityExpander(Expander):
    def apply(self, tc):
        yield tc

class PartExpander(Expander):
    def apply(self, part_range):
        phrase = part_range.parent
        part_nos = [part_no for part_no in expand_implicit_text(part_range.get_span())]
        parts = {}
        parts['text'] = ', '.join(part_nos)
        parts['words'] = part_nos
        parts['lemmas'] = part_nos
        parts['pos_tags'] = [phrase.pos_tags[0]] * len(part_nos)
        parts['ner_tags'] = [phrase.ner_tags[0]] * len(part_nos)
        parts['char_offsets'] = 
        parts['dep_parents'] = sort_X_on_Y(dep_par, dep_order)
        parts['dep_labels'] = sort_X_on_Y(dep_lab, dep_order)
        parts['text']
        parts['position'] = position
        parts['document']
        parts['stable_id']
        parts['document'] = document
        parts['table'] = table
        parts['cell'] = cell
        parts['phrase_id'] = self.phrase_idx
        parts['row_num'] = cell.row_num if cell is not None else None
        parts['col_num'] = cell.col_num if cell is not None else None
        parts['html_tag'] = tag.name
        parts['html_attrs'] = tag.attrs
        parts['html_anc_tags'] = anc_tags
        parts['html_anc_attrs'] = anc_attrs

        # for parts in self.corenlp_handler.parse(context.document, expanded_text):
        #     parts['document'] = context.document
        #     parts['table'] = context.table
        #     parts['cell'] = context.cell
        #     if context.cell is not None:
        #         parts['row_num'] = context.cell.row_num
        #         parts['col_num'] = context.cell.col_num
        #     parts['html_tag'] = context.html_tag
        #     parts['html_attrs'] = context.html_attrs
        #     parts['html_anc_tags'] = context.html_anc_tags
        #     parts['html_anc_attrs'] = context.html_anc_attrs
        #     p = Phrase(**parts)
        #     yield TemporarySpan(char_start=0, char_end=(len(expanded_text)-1), context=p)
        # for part in ):


        # expanded_text = " ".join(expanded_texts)