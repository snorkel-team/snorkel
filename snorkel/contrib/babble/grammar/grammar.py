from __future__ import print_function

from collections import defaultdict, namedtuple
from itertools import product
import re

from types import FunctionType

from snorkel.parser.spacy_parser import Spacy
from rule import Rule
from parse import Parse
import utils


class GrammarMixin(object):
    def __init__(self, rules, ops, helpers, annotators, translate_ops):
        self.rules = rules
        self.ops = ops
        self.helpers = helpers
        self.annotators = annotators
        self.translate_ops = translate_ops


class Grammar(object):
    def __init__(self, bases, candidate_class=None, user_lists={}, 
        beam_width=10, top_k=-1, start_symbol='$ROOT'):
       
        # Extract from bases
        bases = bases if isinstance(bases, list) else [bases]
        rules = []
        self.ops = {}
        self.helpers = {}
        self.annotators = []
        self.translate_ops = {}
        for base in bases:
            rules += base.rules
            self.ops.update(base.ops)
            self.helpers.update(base.helpers)
            self.annotators += base.annotators
            self.translate_ops.update(base.translate_ops)
        # Add candidate-specific rules and user_lists
        if candidate_class:
            for i, arg in enumerate(candidate_class.__argnames__):
                rules.append(Rule('$ArgX', arg, ('.arg', ('.int', i + 1))))
                rules.append(Rule('$ArgX', arg + 's', ('.arg', ('.int', i + 1))))
        self.candidate_class = candidate_class
        self.user_lists = user_lists

        # Set parameters
        self.beam_width = beam_width
        self.top_k = top_k
        
        # Initialize
        self.categories = set()
        self.lexical_rules = defaultdict(list)
        self.unary_rules = defaultdict(list)
        self.binary_rules = defaultdict(list)
        self.start_symbol = start_symbol
        self.parser = Spacy()
        for rule in rules:
            self.add_rule(rule)
        print('Created grammar with %d rules' % \
            (len(self.lexical_rules) + len(self.unary_rules) + len(self.binary_rules)))

    def parse_string(self, string):
        """
        Returns the list of parses for the given string which can be derived
        using this grammar.

        :param string:
        """
        # Tokenize input string
        # string = string.lower()
        if string.endswith('.'):
            string = string[:-1]
        string = re.sub(r'\s+', ' ', string)
        output = self.parser.parse(None, string).next()
        tokens = map(lambda x: dict(zip(['word', 'pos', 'ner'], x)), 
                     zip(output['words'], output['pos_tags'], output['ner_tags']))
        
        # Lowercase all non-quoted words; doesn't handle nested quotes
        quoting = False
        for token in tokens:
            if not quoting:
                token['word'] = token['word'].lower()
            if token['pos'] in ["``", "\'\'"]:
                quoting = not quoting

        # Add start and stop _after_ parsing to not confuse the CoreNLP parser
        start = {'word': '<START>', 'pos': '<START>', 'ner': '<START>'}
        stop = {'word': '<STOP>', 'pos': '<STOP>', 'ner': '<STOP>'}
        tokens = [start] + tokens + [stop]
        words = [t['word'] for t in tokens]
        self.words = words # (for print_chart)
        
        # ABANDONED:
        # Add temporary string rules
        # if self.string_format == 'implicit':
        #     for word in words:
        #         if word not in stopwords:
        #             self.add_rule(Rule('$String', word, ('.string', word))))

        chart = defaultdict(list)
        for j in range(1, len(tokens) + 1):
            for i in range(j - 1, -1, -1):
                self.apply_annotators(chart, tokens, i, j) # tokens[i:j] should be tagged?
                self.apply_user_lists(chart, words, i, j) # words[i:j] is the name of a UserList?
                self.apply_lexical_rules(chart, words, i, j) # words[i:j] matches lexical rule?
                self.apply_binary_rules(chart, i, j) # any split of words[i:j] matches binary rule?
                self.apply_absorb_rules(chart, i, j)
                self.apply_unary_rules(chart, i, j) # add additional tags if chart[(i,j)] matches unary rule
                if self.beam_width:
                    self.apply_beam(chart, i, j)
        parses = chart[(0, len(tokens))]
        if self.start_symbol:
            parses = [parse for parse in parses if parse.rule.lhs == self.start_symbol]
        self.chart = chart
        if self.top_k:
            # If top_k is negative, accept all parses that are tied for the 
            # fewest absorptions, then second fewest absorptions, ..., then k-fewest absorptions
            if self.top_k < 0:
                k = abs(self.top_k)
                levels = sorted(list(set(p.absorbed for p in parses)))
                parses = [p for p in parses if p.absorbed in levels[:k]]
            else:
                parses = sorted(parses, key=lambda x: x.absorbed)[:self.top_k]
        return parses

    def add_rule(self, rule):
        if rule.contains_optionals():
            self.add_rule_containing_optional(rule)
        elif rule.is_lexical():
            self.lexical_rules[rule.rhs].append(rule)
        elif rule.is_unary():
            self.unary_rules[rule.rhs].append(rule)
        elif rule.is_binary():
            self.binary_rules[rule.rhs].append(rule)
        elif all([utils.is_cat(rhsi) for rhsi in rule.rhs]):
            self.add_n_ary_rule(rule)
        else:
            raise Exception('RHS mixes terminals and non-terminals: %s' % rule)

    def add_rule_containing_optional(self, rule):
        """
        Handles adding a rule which contains an optional element on the RHS.
        We find the leftmost optional element on the RHS, and then generate
        two variants of the rule: one in which that element is required, and
        one in which it is removed.  We add these variants in place of the
        original rule.  (If there are more optional elements further to the
        right, we'll wind up recursing.)
        For example, if the original rule is:
            Rule('$Z', '$A ?$B ?$C $D')
        then we add these rules instead:
            Rule('$Z', '$A $B ?$C $D')
            Rule('$Z', '$A ?$C $D')
        """
        # Find index of the first optional element on the RHS.
        first = next((idx for idx, elt in enumerate(rule.rhs) if utils.is_optional(elt)), -1)
        assert first >= 0
        assert len(rule.rhs) > 1, 'Entire RHS is optional: %s' % rule
        prefix = rule.rhs[:first]
        suffix = rule.rhs[(first + 1):]
        # First variant: the first optional element gets deoptionalized.
        deoptionalized = (rule.rhs[first][1:],)
        self.add_rule(Rule(rule.lhs, prefix + deoptionalized + suffix, rule.sem))
        # Second variant: the first optional element gets removed.
        # If the semantics is a value, just keep it as is.
        sem = rule.sem
        # But if it's a function, we need to supply a dummy argument for the removed element.
        if isinstance(rule.sem, FunctionType):
            sem = lambda sems: rule.sem(sems[:first] + [None] + sems[first:])
        self.add_rule(Rule(rule.lhs, prefix + suffix, sem))

    def add_n_ary_rule(self, rule):
        """
        Handles adding a rule with three or more non-terminals on the RHS.
        We introduce a new category which covers all elements on the RHS except
        the first, and then generate two variants of the rule: one which
        consumes those elements to produce the new category, and another which
        combines the new category which the first element to produce the
        original LHS category.  We add these variants in place of the
        original rule.  (If the new rules still contain more than two elements
        on the RHS, we'll wind up recursing.)
        For example, if the original rule is:
            Rule('$Z', '$A $B $C $D')
        then we create a new category '$Z_$A' (roughly, "$Z missing $A to the left"),
        and add these rules instead:
            Rule('$Z_$A', '$B $C $D')
            Rule('$Z', '$A $Z_$A')
        """
        def add_category(base_name):
            assert utils.is_cat(base_name)
            name = base_name
            while name in self.categories:
                name = name + '_'
            self.categories.add(name)
            return name
        category = add_category('%s_%s' % (rule.lhs, rule.rhs[0]))
        self.add_rule(Rule(category, rule.rhs[1:], lambda sems: sems))
        self.add_rule(Rule(rule.lhs, (rule.rhs[0], category),
                            lambda sems: rule.apply_semantics([sems[0]] + sems[1])))

    def apply_user_lists(self, chart, words, i, j):
        """Add parses to chart cell (i, j) by applying user lists."""
        if self.user_lists:
            key = ' '.join(words[i:j])
            if key in self.user_lists:
                lhs = '$UserList'
                rhs = tuple(key.split())
                semantics = ('.user_list', ('.string', key))
                rule = Rule(lhs, rhs, semantics)
                chart[(i, j)].append(Parse(rule, words[i:j]))

    def apply_annotators(self, chart, tokens, i, j):
        """Add parses to chart cell (i, j) by applying annotators."""
        if self.annotators:
            words = [t['word'] for t in tokens]
            for annotator in self.annotators:
                for category, semantics in annotator.annotate(tokens[i:j]):
                    rule = Rule(category, tuple(words[i:j]), semantics)
                    chart[(i, j)].append(Parse(rule, words[i:j]))

    def apply_lexical_rules(self, chart, words, i, j):
        """Add parses to chart cell (i, j) by applying lexical rules."""
        for rule in self.lexical_rules[tuple(words[i:j])]:
            chart[(i, j)].append(Parse(rule, words[i:j]))

    def apply_binary_rules(self, chart, i, j):
        """Add parses to chart cell (i, j) by applying binary rules."""
        for k in range(i + 1, j):
            for parse_1, parse_2 in product(chart[(i, k)], chart[(k, j)]):
                for rule in self.binary_rules[(parse_1.rule.lhs, parse_2.rule.lhs)]:
                    chart[(i, j)].append(Parse(rule, [parse_1, parse_2]))
    
    def apply_absorb_rules(self, chart, i, j):
        """Add parses to chart cell (i, j) that require absorbing."""
        if j - i > 2: # Otherwise, there's no chance for absorption
            for m in range(i + 1, j - 1):
                for n in range(m + 1, j):
                    for parse_1, parse_2 in product(chart[(i, m)], chart[(n, j)]):
                        # Don't absorb unmatched quote marks
                        if sum(parse.rule.lhs=='$Quote' for p in range(m, n) for parse in chart[(p, p+1)]) % 2 != 0:
                            break
                        for rule in self.binary_rules[(parse_1.rule.lhs, parse_2.rule.lhs)]:
                            # Don't allow $StringStub to absorb (to control growth)
                            if rule.lhs=='$StringStub':
                                continue
                            absorbed = n - m
                            chart[(i, j)].append(Parse(rule, [parse_1, parse_2], absorbed))

    def apply_unary_rules(self, chart, i, j):
        """Add parses to chart cell (i, j) by applying unary rules."""
        # Note that the last line of this method can add new parses to chart[(i,
        # j)], the list over which we are iterating.  Because of this, we
        # essentially get unary closure "for free".  (However, if the grammar
        # contains unary cycles, we'll get stuck in a loop, which is one reason for
        # check_capacity().)
        for parse in chart[(i, j)]:
            for rule in self.unary_rules[(parse.rule.lhs,)]:
                chart[(i, j)].append(Parse(rule, [parse]))
        # while True:
        #     nStart = len(chart[(i, j)])
        #     for parse in list(chart[(i, j)]):
        #         for rule in self.unary_rules[(parse.rule.lhs,)]:
        #             chart[(i, j)].append(Parse(rule, [parse]))
        #     nEnd = len(chart[(i,j)])
        #     if nEnd == nStart:
        #         return

    def apply_beam(self, chart, i, j):
        chart[(i,j)] = sorted(chart[(i,j)], key=lambda x: x.absorbed)[:self.beam_width]

    def evaluate(self, parse):
        def recurse(sem):
            if isinstance(sem, tuple):
                op = self.ops[sem[0]]
                args = [recurse(arg) for arg in sem[1:]]
                return op(*args) if args else op
            else:
                return sem
        LF = recurse(parse.semantics)
        return lambda candidate: LF({'helpers': self.helpers, 'user_lists': self.user_lists, 'candidate': candidate})

    def translate(self, sem):
        def recurse(sem):
            if isinstance(sem, tuple):
                if sem[0] in self.translate_ops:
                    op = self.translate_ops[sem[0]]
                    args_ = [recurse(arg) for arg in sem[1:]]
                    return op(*args_) if args_ else op
                else:
                    return str(sem)
            else:
                return str(sem)
        return recurse(sem)

    def print_grammar(self):
        def all_rules(rule_index):
            return [rule for rules in list(rule_index.values()) for rule in rules]
        def print_rules_sorted(rules):
            for s in sorted([str(rule) for rule in rules]):
                print('  ' + s)
        print('Lexical rules:')
        print_rules_sorted(all_rules(self.lexical_rules))
        print('Unary rules:')
        print_rules_sorted(all_rules(self.unary_rules))
        print('Binary rules:')
        print_rules_sorted(all_rules(self.binary_rules))

    def print_chart(self, nested=False):
        """Print the chart.  Useful for debugging."""
        spans = sorted(list(self.chart.keys()), key=(lambda span: span[0]))
        spans = sorted(spans, key=(lambda span: span[1] - span[0]))
        for span in spans:
            if len(self.chart[span]) > 0:
                print('%-12s' % str(span))
                if nested:
                    for entry in self.chart[span]:
                        print('%-12s' % ' ', entry)
                else:
                    print(' '.join(self.words[span[0]:span[1]]))
                    for entry in self.chart[span]:
                        print('%-12s' % ' ', entry.rule.lhs)