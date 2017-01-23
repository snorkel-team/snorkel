from types import FunctionType
from collections import defaultdict
from itertools import product

class Example:
    def __init__(self, input=None, candidate=None, parse=None, semantics=None, denotation=None):
        self.input = input
        self.candidate = candidate
        self.parse = parse
        self.semantics = semantics
        self.denotation = denotation

    def __str__(self):
        return 'Example(%s)' % (input)


class Rule:
    def __init__(self, lhs, rhs, sem=None):
        self.lhs = lhs
        self.rhs = tuple(rhs.split()) if isinstance(rhs, str) else rhs
        self.sem = sem

    def __str__(self):
        return 'Rule' + str((self.lhs, ' '.join(self.rhs), self.sem))

    def is_cat(self, label):
        return label.startswith('$')

    def is_lexical(self):
        return all([not self.is_cat(rhsi) for rhsi in self.rhs])

    def is_binary(self):
        return len(self.rhs) == 2 and self.is_cat(self.rhs[0]) and self.is_cat(self.rhs[1])


class Grammar:
    def __init__(self, rules=[]):
        self.lexical_rules = defaultdict(list)
        self.binary_rules = defaultdict(list)
        for rule in rules:
            self.add_rule(rule)
        print('Created grammar with %d rules.' % len(rules))

    def add_rule(self, rule):
        if rule.is_lexical():
            self.lexical_rules[rule.rhs].append(rule)
        elif rule.is_binary():
            self.binary_rules[rule.rhs].append(rule)
        else:
            raise Exception('Cannot accept rule: %s' % rule)
        
    def parse_input(self, input):
        """Returns a list of parses for the given input."""
        chart = defaultdict(list)  # map from span (i, j) to list of parses
        tokens = input.split()
        for j in range(1, len(tokens) + 1):
            for i in range(j - 1, -1, -1):
                self.apply_lexical_rules(chart, tokens, i, j)
                self.apply_binary_rules(chart, i, j)
        return chart[(0, len(tokens))]  # return all parses for full span

    def apply_lexical_rules(self, chart, tokens, i, j):
        """Add parses to span (i, j) in chart by applying lexical rules from grammar to tokens."""
        for rule in self.lexical_rules[tuple(tokens[i:j])]:
            chart[(i, j)].append(Parse(rule, tokens[i:j]))

    def apply_binary_rules(self, chart, i, j):
        """Add parses to span (i, j) in chart by applying binary rules from grammar."""
        for k in range(i + 1, j):  # all ways of splitting the span into two subspans
            for parse_1, parse_2 in product(chart[(i, k)], chart[(k, j)]):
                for rule in self.binary_rules[(parse_1.rule.lhs, parse_2.rule.lhs)]:
                    chart[(i, j)].append(Parse(rule, [parse_1, parse_2]))

class Parse:
    def __init__(self, rule, children):
        self.rule = rule
        self.children = tuple(children[:])
        self.semantics = self.compute_semantics()
        self.score = float('NaN')                 # Ditto.
        self.denotation = None                    # Ditto.

    def __str__(self):
        return '(%s %s)' % (self.rule.lhs, ' '.join([str(c) for c in self.children]))
    
    def compute_semantics(self):
        if self.rule.is_lexical() or not isinstance(self.rule.sem, FunctionType):
            return self.rule.sem
        else:
            return self.rule.sem([child.semantics for child in self.children]) 