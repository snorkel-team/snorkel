from grammar import Grammar
from snorkel_grammar import snorkel_rules, snorkel_ops, sem_to_str
from annotator import *

from pandas import DataFrame, Series

class Example(object):
    def __init__(self, name=None, explanation=None, paraphrase=None,
                 candidate=None, denotation=None, semantics=None):
        self.name = name
        self.explanation = explanation
        self.paraphrase = paraphrase
        self.candidate = candidate
        self.denotation = denotation  # True label on this candidate
        self.semantics = semantics

    def __str__(self):
        return 'Example("%s")' % (self.explanation)
    
    def display(self):
        print 'Example'
        print('%-12s %s' % ('name', self.name))
        print('%-12s %s' % ('explanation', self.explanation))
        print('%-12s %s' % ('candidate', self.candidate))
        print('%-12s %d' % ('denotation', self.denotation))
        print('%-12s %s' % ('semantics', self.semantics))


class SemanticParser():
    def __init__(self, candidate_class, user_lists={}, beam_width=None, top_k=None):
        annotators = [TokenAnnotator(), PunctuationAnnotator(), IntegerAnnotator()]
        self.grammar = Grammar(rules=snorkel_rules, 
                               ops=snorkel_ops, 
                               candidate_class=candidate_class,
                               annotators=annotators,
                               user_lists=user_lists,
                               beam_width=beam_width,
                               top_k=top_k)
        self.explanation_counter = 0
        self.LFs = tuple([None] * 6)

    def preprocess(self, explanations):
        for explanation in explanations:
            explanation = explanation.replace("'", '"')
            yield explanation

    def parse(self, explanations, names=None, verbose=False, return_parses=False):
        """
        Converts natural language explanations into labeling functions.
        """
        LFs = []
        parses = []
        explanations = explanations if isinstance(explanations, list) else [explanations]
        names = names if isinstance(names, list) else [names]
        for i, exp in enumerate(self.preprocess(explanations)):
            exp_parses = self.grammar.parse_input(exp)
            for j, parse in enumerate(exp_parses):
                # print(parse.semantics)
                lf = self.grammar.evaluate(parse)
                if return_parses:
                    parse.function = lf
                    parses.append(parse)
                if len(names) > i and names[i]:
                    lf.__name__ = "{}_{}".format(names[i], j)
                else:
                    lf.__name__ = "exp%d_%d" % (self.explanation_counter, j)
                LFs.append(lf)
            self.explanation_counter += 1
        if return_parses:
            if verbose:
                print("{} parses created from {} explanations".format(len(LFs), len(explanations)))
            return parses
        else:
            if verbose:
                print("{} LFs created from {} explanations".format(len(LFs), len(explanations)))
            return LFs

    def parse_and_evaluate(self, 
                           examples, 
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
                           remove_paren=True,
                           paraphrases=False,
                           only=[]):
        """
        Calls SemanticParser.parse and evaluates the accuracy of resulting LFs.
        
        Results are stored in self.results, which contains a pandas DataFrame.
        """
        assert(not (show_everything and show_nothing))
        if show_everything:
            show_explanation = show_candidate = show_sentence = show_parse = show_semantics = True
        if show_semantics:
            show_correct = show_passing = show_failing = True
            show_redundant = show_erroring = show_unknown = True
        if show_nothing:
            show_explanation = show_candidate = show_sentence = show_parse = show_semantics = False
            show_correct = show_passing = show_failing = show_redundant = show_erroring = show_unknown = False
        self.explanation_counter = 0
        examples = examples if isinstance(examples, list) else [examples]
        col_names = ['Correct', 'Passing', 'Failing', 'Redundant', 'Erroring', 'Unknown','Index']
        d = {}
        example_names = []
        indices = []

        nCorrect = [0] * len(examples)
        nPassing = [0] * len(examples)
        nFailing = [0] * len(examples)
        nRedundant = [0] * len(examples)
        nErroring = [0] * len(examples)
        nUnknown = [0] * len(examples)

        LFs = {
            'correct'  : [],
            'passing'  : [],
            'failing'  : [],
            'redundant': [],
            'erroring' : [],
            'unknown'  : [],
        }
        
        for i, example in enumerate(examples):
            if only and i not in only:
                continue
            # if example.explanation is None or (paraphrases and example.paraphrase is None):
            #     continue
            indices.append(i)
            if paraphrases and not example.paraphrase:
                raise Exception('Keyword argument paraphrases == True '
                                'but Example has no paraphrase.')
            explanation = example.explanation if not paraphrases else example.paraphrase
            if show_explanation: 
                print("Example {}: {}\n".format(i, explanation))
            if show_candidate:
                print("CANDIDATE: {}\n".format(example.candidate))
            if show_sentence and not isinstance(example.candidate[0], str):
                print("SENTENCE: {}\n".format(example.candidate[0].get_parent()._asdict()['text']))
            semantics = set()
            # TODO: remove remove_paren keyword and code
            if remove_paren:
                explanation = explanation.replace('(', '')
                explanation = explanation.replace(')', '')
            parses = self.parse(
                        explanation, 
                        example.name,
                        return_parses=True)
            for parse in parses:
                if show_parse:
                    print("PARSE: {}\n".format(parse))
                semantics_ = sem_to_str(parse.semantics) if pseudo_python else parse.semantics
                # REDUNDANT
                if parse.semantics in semantics:
                    if show_redundant: print("R: {}\n".format(semantics_))
                    nRedundant[i] += 1
                    LFs['redundant'].append(parse.function)
                    continue
                semantics.add(parse.semantics)
                # ERRORING
                try:
                    denotation = parse.function(example.candidate)
                except:
                    if show_erroring: 
                        print("E: {}\n".format(semantics_))
                        print parse.semantics
                        print parse.function(example.candidate) #to display traceback
                        import pdb; pdb.set_trace()
                    nErroring[i] += 1 
                    LFs['erroring'].append(parse.function)
                    continue
                # CORRECT             
                if example.semantics and parse.semantics==example.semantics:
                    if show_correct: print("C: {}\n".format(semantics_))
                    nCorrect[i] += 1
                    LF = parse.function
                    LF.__name__ = LF.__name__[:(LF.__name__).rindex('_')] + '*'
                    LFs['correct'].append(parse.function)
                    continue
                # PASSING
                if denotation==example.denotation:
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
                if example.candidate is None:
                    nUnknown[i] += 1
                    LFs['unknown'].append(parse.function)
                    continue
                raise Error('This should not be reached.')
                            
            if nCorrect[i] + nPassing[i] == 0:
                print("WARNING: No correct or passing parses found for the following explanation:")
                print("EXPLANATION {}: {}\n".format(i, explanation))

            if example.name:
                example_names.append(example.name)
            else:
                example_names.append("Example{}".format(i))

        d['Correct'] = Series(data=[nCorrect[i] for i in indices], index=example_names)
        d['Passing'] = Series(data=[nPassing[i] for i in indices], index=example_names)
        d['Failing'] = Series(data=[nFailing[i] for i in indices], index=example_names)
        d['Redundant'] = Series(data=[nRedundant[i] for i in indices], index=example_names)
        d['Erroring'] = Series(data=[nErroring[i] for i in indices], index=example_names)
        d['Unknown'] = Series(data=[nUnknown[i] for i in indices], index=example_names)
        d['Index'] = Series(data=indices, index=example_names)
        
        self.results = DataFrame(data=d, index=example_names)[col_names]
        return LFs