from __future__ import print_function

from collections import Iterable
from six import StringIO

from rule import Rule
import utils

class Parse(object):
    def __init__(self, rule, children, absorbed=0):
        self.rule = rule
        self.children = tuple(children[:])
        self.semantics = self.compute_semantics()
        self.function = None
        self.explanation = None
        self.absorbed = absorbed + sum(child.absorbed for child in self.children if isinstance(child, Parse))
        self.validate_parse()

    def __eq__(self, other):
        return hash(self) == hash(other)
    
    def __ne__(self, other):
        return (not self.__eq__(other))

    def __hash__(self):
        return hash(self.__repr__())

    def __repr__(self):
        if self.function:
            return "Parse({})".format(self.function.__name__)
        else:
            return "Parse(hash={})".format(hash(self.semantics)[:8])
        # child_strings = [str(child) for child in self.children]
        # return '(%s %s)' % (self.rule.lhs, ' '.join(child_strings))

    def validate_parse(self):
        assert isinstance(self.rule, Rule), 'Not a Rule: %s' % self.rule
        assert isinstance(self.children, Iterable)
        assert len(self.children) == len(self.rule.rhs)
        for i in range(len(self.rule.rhs)):
            if utils.is_cat(self.rule.rhs[i]):
                assert self.rule.rhs[i] == self.children[i].rule.lhs
            else:
                assert self.rule.rhs[i] == self.children[i]

    def compute_semantics(self):
        if self.rule.is_lexical():
            return self.rule.sem
        else:
            child_semantics = [child.semantics for child in self.children]
            return self.rule.apply_semantics(child_semantics)

    def display(self, indent=0, show_sem=False):
        def indent_string(level):
            return '  ' * level

        def label(parse):
            if show_sem:
                return '(%s %s)' % (parse.rule.lhs, parse.semantics)
            else:
                return parse.rule.lhs

        def to_oneline_string(parse):
            if isinstance(parse, Parse):
                child_strings = [to_oneline_string(child) for child in parse.children]
                return '[%s %s]' % (label(parse), ' '.join(child_strings))
            else:
                return str(parse)

        def helper(parse, level, output):
            line = indent_string(level) + to_oneline_string(parse)
            if len(line) <= 100:
                print(line, file=output)
            elif isinstance(parse, Parse):
                print(indent_string(level) + '[' + label(parse), file=output)
                for child in parse.children:
                    helper(child, level + 1, output)
                # TODO: Put closing parens to end of previous line, not dangling alone.
                print(indent_string(level) + ']', file=output)
            else:
                print(indent_string(level) + parse, file=output)
        output = StringIO()
        helper(self, indent, output)
        return output.getvalue()[:-1]  # trim final newline


def validate_semantics(self):
    # TODO: write this function.
    assert True