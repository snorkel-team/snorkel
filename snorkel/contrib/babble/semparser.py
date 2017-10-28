import matplotlib.pyplot as plt
from pandas import DataFrame, Series

from snorkel.models import Sentence

from core import core_grammar
from text import text_grammar
from image import image_grammar
from grammar import Grammar, validate_semantics, stopwords
from explanation import Explanation

class SemanticParser(object):
    """
    SemanticParser varies based on:
    --domain: [spouse, cdr, coco]
        candidate_class
        user_lists
        domain-specific rules
        annotators
    --mode: [text/image/table]
        mode-specific rules
    """
    def __init__(self, mode='core', string_format='implicit', **kwargs):
        grammar_mixins = [core_grammar]
        if mode == 'core':
            pass
        elif mode == 'text':
            grammar_mixins.append(text_grammar)
        elif mode == 'image':
            grammar_mixins.append(image_grammar)
        elif mode == 'table':
            pass
        else:
            raise Exception("You must specify a mode in ['text', 'image', 'table']")
        self.grammar = Grammar(grammar_mixins, **kwargs)
        self.mode = mode
        self.string_format = string_format
        if string_format == 'implicit':
            self.unquotable = [' '.join(key) for key in self.grammar.lexical_rules] + stopwords
        self.explanation_counter = 0

    def name_explanations(self, explanations, names):
        if names:
            if len(names) != len(explanations):
                raise Exception("If argument _names_ is provided, _names_ and "
                    "_explanations_ must have same length.")
            else:
                for exp, name in zip(explanations, names):
                    exp.name = name
        else:
            for i, exp in enumerate(explanations):
                if not exp.name:
                    exp.name = "Explanation{}".format(i)

    def parse(self, explanations, names=None, verbose=False, return_parses=False):
        """
        Converts Explanation objects into labeling functions.

        :param explanations: An instance or list of Explanation objects
        """
        LFs = []
        parses = []
        num_parses_by_exp = []
        explanations = explanations if isinstance(explanations, list) else [explanations]
        names = names if isinstance(names, list) or names is None else [names]
        self.name_explanations(explanations, names)
        for i, exp in enumerate(explanations):
            exp_normalized = u'label {} if {}'.format(exp.label, exp.condition)
            if (self.mode == 'text' and self.string_format == 'implicit' and 
                getattr(exp.candidate, 'get_parent', None) and
                isinstance(exp.candidate.get_parent(), Sentence)):
                exp_normalized = self.mark_implicit_strings(exp_normalized, exp.candidate)
            exp_parses = self.grammar.parse_string(exp_normalized)
            num_parses_by_exp.append(len(exp_parses))
            for j, parse in enumerate(exp_parses):
                parse.explanation = exp
                lf = self.grammar.evaluate(parse)
                if return_parses:
                    parse.function = lf
                    parses.append(parse)
                lf.__name__ = "{}_{}".format(exp.name, j)
                LFs.append(lf)
            self.explanation_counter += 1
        if verbose:
            return_object = 'parse(s)' if return_parses else "LF(s)"
            print("{} explanation(s) out of {} were parseable.".format(
                len(explanations) - num_parses_by_exp.count(0), 
                len(explanations)))
            print("{} {} generated from {} explanation(s).".format(
                len(LFs), return_object, len(explanations)))
        if return_parses:
            return parses
        else:
            if verbose:
                plt.hist(num_parses_by_exp, 
                    bins=range(max(num_parses_by_exp) + 2), align='left')
                plt.xticks(range(max(num_parses_by_exp) + 2))
                plt.xlabel("# of LFs")
                plt.ylabel("# of Explanations")
                plt.title('# LFs per Explanation')
                plt.show()
            return LFs

    def parse_and_evaluate(self, 
                           explanations, 
                           show_everything=False,
                           show_nothing=False,
                           show_explanation=False, 
                           show_candidate=False,
                           show_sentence=False, 
                           show_parse=False,
                           show_semantics=False,
                           show_correct=False,
                           show_passing=False, 
                           show_failing=False,
                           show_redundant=False,
                           show_erroring=False,
                           show_unknown=False,
                           pseudo_python=False,
                           remove_paren=False,
                           paraphrases=False,
                           only=[]):
        """
        Calls SemanticParser.parse and evaluates the accuracy of resulting LFs.
        
        Results are stored in self.results, which contains a pandas DataFrame.
        """
        assert(not (show_everything and show_nothing))
        if show_everything:
            if any([show_explanation, show_candidate, show_sentence, show_parse, show_semantics]):
                print("Note: show_everything = True. This overrides all other show_x commands.")
            show_explanation = show_candidate = show_sentence = show_parse = show_semantics = True
        if show_semantics:
            show_correct = show_passing = show_failing = True
            show_redundant = show_erroring = show_unknown = True
        if show_nothing:
            if any([show_explanation, show_candidate, show_sentence, show_parse, show_semantics, 
            show_correct, show_passing, show_failing, show_redundant, show_erroring, show_unknown]):
                print("Note: show_nothing = True. This will override all other show_ commands.")
            show_explanation = show_candidate = show_sentence = show_parse = show_semantics = False
            show_correct = show_passing = show_failing = show_redundant = show_erroring = show_unknown = False
        self.explanation_counter = 0
        explanations = explanations if isinstance(explanations, list) else [explanations]
        col_names = ['Correct', 'Passing', 'Failing', 'Redundant', 'Erroring', 'Unknown','Index']
        dataframe = {}
        indices = []

        nCorrect = [0] * len(explanations)
        nPassing = [0] * len(explanations)
        nFailing = [0] * len(explanations)
        nRedundant = [0] * len(explanations)
        nErroring = [0] * len(explanations)
        nUnknown = [0] * len(explanations)

        LFs = {
            'correct'  : [],
            'passing'  : [],
            'failing'  : [],
            'redundant': [],
            'erroring' : [],
            'unknown'  : [],
        }
        
        for i, explanation in enumerate(explanations):
            if only and i not in only:
                continue
            indices.append(i)
            if paraphrases and not explanation.paraphrase:
                raise Exception('Keyword argument paraphrases == True '
                                'but explanation has no paraphrase.')
            # TODO: remove remove_paren keyword and code
            if remove_paren:
                condition = explanation.replace('(', '')
                condition = explanation.replace(')', '')
            if show_explanation: 
                print("Explanation {}: {}\n".format(i, explanation))
            if show_candidate:
                print("CANDIDATE: {}\n".format(explanation.candidate))
            if show_sentence and not isinstance(explanation.candidate[0], str):
                print("SENTENCE: {}\n".format(explanation.candidate[0].get_parent()._asdict()['text']))
            semantics = set()
            parses = self.parse(
                        explanation, 
                        explanation.name,
                        return_parses=True)
            for parse in parses:
                if show_parse:
                    print("PARSE: {}\n".format(parse))
                semantics_ = self.translate(parse.semantics) if pseudo_python else parse.semantics
                # REDUNDANT
                if parse.semantics in semantics:
                    if show_redundant: print("R: {}\n".format(semantics_))
                    nRedundant[i] += 1
                    LFs['redundant'].append(parse.function)
                    continue
                semantics.add(parse.semantics)
                # ERRORING
                try:
                    condition_passes = parse.function(explanation.candidate)
                except:
                    if show_erroring: 
                        print("E: {}\n".format(semantics_))
                        print parse.semantics
                        print parse.function(explanation.candidate)  # to display traceback
                        import pdb; pdb.set_trace()
                    nErroring[i] += 1 
                    LFs['erroring'].append(parse.function)
                    continue
                # CORRECT             
                if explanation.semantics and parse.semantics == explanation.semantics:
                    if show_correct: print("C: {}\n".format(semantics_))
                    nCorrect[i] += 1
                    LF = parse.function
                    LF.__name__ = LF.__name__[:(LF.__name__).rindex('_')] + '*'
                    LFs['correct'].append(parse.function)
                    continue
                # PASSING
                if condition_passes:
                    if show_passing: print("P: {}\n".format(semantics_))
                    nPassing[i] += 1
                    LFs['passing'].append(parse.function)
                    continue
                else:
                # FAILING
                    if show_failing: print("F: {}\n".format(semantics_))
                    nFailing[i] += 1
                    LFs['failing'].append(parse.function)
                    continue
                # UNKNOWN
                if explanation.candidate is None:
                    nUnknown[i] += 1
                    LFs['unknown'].append(parse.function)
                    continue
                raise Error('This should not be reached.')
                            
            if nCorrect[i] + nPassing[i] == 0:
                print("WARNING: No correct or passing parses found for the following explanation:")
                print("EXPLANATION {}: {}\n".format(i, explanation))

        explanation_names = [exp.name for exp in explanations]
        dataframe['Correct'] = Series(data=[nCorrect[i] for i in indices], index=explanation_names)
        dataframe['Passing'] = Series(data=[nPassing[i] for i in indices], index=explanation_names)
        dataframe['Failing'] = Series(data=[nFailing[i] for i in indices], index=explanation_names)
        dataframe['Redundant'] = Series(data=[nRedundant[i] for i in indices], index=explanation_names)
        dataframe['Erroring'] = Series(data=[nErroring[i] for i in indices], index=explanation_names)
        dataframe['Unknown'] = Series(data=[nUnknown[i] for i in indices], index=explanation_names)
        dataframe['Index'] = Series(data=indices, index=explanation_names)
        
        self.results = DataFrame(data=dataframe, index=explanation_names)[col_names]
        return LFs

    def mark_implicit_strings(self, condition, candidate):
        """
        Puts quotation marks around words that are likely quotes from candidate.

        To be quoted, a phrase must:
        a) not already be in quotes
        b) occur in the candidate's sentence
        c) not be a part of any existing lexical rule in the grammar

        If a phrase is a component span of the candidate, it is replaced with
        _arg 1_ or _arg 2_ instead (without underscores).
        """
        # TEMP
        # original_condition = condition
        # TEMP

        # First, replace direct mentions of candidate components with _arg x_
        candidate_words = set(candidate.get_parent().words)
        candidate_text = candidate.get_parent().text
        for argnum in [1, 2]:
            if candidate[argnum - 1].get_span() in condition:
                # Replace name with _arg x_
                condition = condition.replace(candidate[argnum - 1].get_span(), 'arg {}'.format(argnum))
                # # Remove spurious quotes
                # condition = condition.replace('"arg {}"'.format(argnum), 'arg {}'.format(argnum))
                # condition = condition.replace('"arg {}\'s"'.format(argnum), 'arg {}'.format(argnum))
        
        # Identify potential quoted words
        condition_words = condition.split()
        quote_list = []
        quoting = False
        for i, word in enumerate(condition_words):
            if word.startswith('"'):
                quoting = True
            if word in candidate_words and not quoting:
                if (quote_list and  # There is something to compare to
                    quote_list[-1][1] == i - 1 and  # The previous word was also added 
                    ' '.join(condition_words[quote_list[-1][0]:i + 1]) in candidate_text):  # The complete phrase appears in candidate
                        quote_list[-1] = (quote_list[-1][0], i)
                else:
                    quote_list.append((i, i))
            if word.endswith('"'):
                quoting = False
        if not quote_list:
            return condition

        # Quote the quotable words
        new_condition_words = []
        i = 0
        j = 0
        while i < len(condition_words):
            if j < len(quote_list) and i == quote_list[j][0]:
                text_to_quote = ' '.join(condition_words[quote_list[j][0]:quote_list[j][1] + 1])
                if text_to_quote.lower() in self.unquotable or all(w in self.unquotable for w in text_to_quote.lower().split()):
                    j += 1
                else:
                    new_condition_words.append('"{}"'.format(text_to_quote))
                    i = quote_list[j][1] + 1
                    j += 1
                    continue
            new_condition_words.append(condition_words[i])
            i += 1
        new_condition = ' '.join(new_condition_words)
        # TEMP
        # if condition != new_condition:
        #     print("Before: {}".format(original_condition))
        #     print("After:  {}".format(new_condition))
        # TEMP
        return new_condition        

    def translate(self, sem):
        """Converts a parse's semantics into a pseudocode string."""
        validate_semantics(sem)
        return self.grammar.translate(sem)