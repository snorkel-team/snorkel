from lf_helpers import *
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

def is_cat(label):
    return label.startswith('$')

def is_optional(label):
    """
    Returns true iff the given RHS item is optional, i.e., is marked with an
    initial '?'.
    """
    return label.startswith('?') and len(label) > 1

class Rule:
    def __init__(self, lhs, rhs, sem=None):
        self.lhs = lhs
        self.rhs = tuple(rhs.split()) if isinstance(rhs, str) else rhs
        self.sem = sem

    def __str__(self):
        return 'Rule' + str((self.lhs, ' '.join(self.rhs), self.sem))

    def is_lexical(self):
        return all([not is_cat(rhsi) for rhsi in self.rhs])

    def is_unary(self):
        return len(self.rhs) == 1 and is_cat(self.rhs[0])

    def is_binary(self):
        return len(self.rhs) == 2 and is_cat(self.rhs[0]) and is_cat(self.rhs[1])

    def contains_optionals(self):
        """Returns true iff the given Rule contains any optional items on the RHS."""
        return any([is_optional(rhsi) for rhsi in self.rhs])


class Grammar:
    def __init__(self, rules=[], annotators=[], start_symbol='$ROOT'):
        self.categories = set()
        self.lexical_rules = defaultdict(list)
        self.unary_rules = defaultdict(list)
        self.binary_rules = defaultdict(list)
        self.annotators = annotators
        self.start_symbol = start_symbol
        for rule in rules:
            self.add_rule(rule)
        print('Created grammar with %d rules.' % len(rules))

    def add_rule(self, rule):
        if rule.contains_optionals():
            self.add_rule_containing_optional(rule)
        elif rule.is_lexical():
            self.lexical_rules[rule.rhs].append(rule)
        elif rule.is_unary():
            self.unary_rules[rule.rhs].append(rule)
        elif rule.is_binary():
            self.binary_rules[rule.rhs].append(rule)
        elif all([is_cat(rhsi) for rhsi in rule.rhs]):
            self.add_n_ary_rule(rule)
        else:
            # One of the exercises will ask you to handle this case.
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
        first = next((idx for idx, elt in enumerate(rule.rhs) if is_optional(elt)), -1)
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
        def apply_semantics(rule, sems):
            if isinstance(rule.sem, FunctionType):
                return rule.sem(sems)
            else:
                return rule.sem
        
        def add_category(base_name):
            name = base_name
            while name in grammar.categories:
                name = name + '_'
            self.categories.add(name)
            return name

        category = add_category('%s_%s' % (rule.lhs, rule.rhs[0]))
        self.add_rule(Rule(category, rule.rhs[1:], lambda sems: sems))
        self.add_rule(Rule(rule.lhs, (rule.rhs[0], category),
            lambda sems: apply_semantics(rule, [sems[0]] + sems[1])))

    def parse_input(self, input):
        """Returns a list of all parses for input using grammar."""
        tokens = input.split()
        chart = defaultdict(list)
        for j in range(1, len(tokens) + 1):
            for i in range(j - 1, -1, -1):
                self.apply_annotators(chart, tokens, i, j)
                self.apply_lexical_rules(chart, tokens, i, j)
                self.apply_binary_rules(chart, i, j)
                self.apply_unary_rules(chart, i, j)
        parses = chart[(0, len(tokens))]
        import pdb; pdb.set_trace()
        if self.start_symbol:
            parses = [parse for parse in parses if parse.rule.lhs == self.start_symbol]
        return parses

    def apply_annotators(self, chart, tokens, i, j):
        """Add parses to chart cell (i, j) by applying annotators."""
        for annotator in self.annotators:
            for category, semantics in annotator.annotate(tokens[i:j]):
                if not self.check_capacity(chart, i, j):
                    return
                rule = Rule(category, tuple(tokens[i:j]), semantics)
                chart[(i, j)].append(Parse(rule, tokens[i:j]))

    def apply_lexical_rules(self, chart, tokens, i, j):
        """Add parses to span (i, j) in chart by applying lexical rules from grammar to tokens."""
        for rule in self.lexical_rules[tuple(tokens[i:j])]:
            chart[(i, j)].append(Parse(rule, tokens[i:j]))

    def apply_unary_rules(self, chart, i, j):
        """Add parses to chart cell (i, j) by applying unary rules."""
        if self.unary_rules:
            for parse in chart[(i, j)]:
                for rule in self.unary_rules[(parse.rule.lhs,)]:
                    if not self.check_capacity(chart, i, j):
                        return
                    chart[(i, j)].append(Parse(rule, [parse]))

    def apply_binary_rules(self, chart, i, j):
        """Add parses to span (i, j) in chart by applying binary rules from grammar."""
        for k in range(i + 1, j):  # all ways of splitting the span into two subspans
            for parse_1, parse_2 in product(chart[(i, k)], chart[(k, j)]):
                for rule in self.binary_rules[(parse_1.rule.lhs, parse_2.rule.lhs)]:
                    chart[(i, j)].append(Parse(rule, [parse_1, parse_2]))

    MAX_CELL_CAPACITY = 10000
    # Important for catching e.g. unary cycles.
    def check_capacity(self, chart, i, j):
        if len(chart[(i, j)]) >= MAX_CELL_CAPACITY:
            print('Cell (%d, %d) has reached capacity %d' % (
                i, j, MAX_CELL_CAPACITY))
            return False
        return True

class Annotator:
    """A base class for annotators."""
    def annotate(self, tokens):
        """Returns a list of pairs, each a category and a semantic representation."""
        return []

class NumberAnnotator(Annotator):
    def annotate(self, tokens):
        if len(tokens) == 1:
            try:
                value = float(tokens[0])
                if int(value) == value:
                    value = int(value)
                return [('$Number', value)]
            except ValueError:
                pass
        return []

class TokenAnnotator(Annotator):
    def annotate(self, tokens):
        if len(tokens) == 1:
            return [('$Token', tokens[0])]
        else:
            return []

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

class Evaluator:
    def __init__(self, ops):
        self.ops = ops
    
    def convert(self, sem):
        """
        Convert semantic representation into Python string that looks like a function.
        """
        if isinstance(sem, tuple):
            op = self.ops[sem[0]]
            args = [self.convert(arg) for arg in sem[1:]]
            return "(%s)" % op(*args)
        else:
            return sem     

    def evaluate(self, sem):
        """
        Return an executable LF and its corresponding source code from an input 
        semantic representation.
        """
        LF_string = self.convert(sem)
        LF = eval(LF_string)
        return LF, LF_string   