from types import FunctionType

import utils

class Rule(object):
    """Represents a CFG rule with a semantic attachment."""

    def __init__(self, lhs, rhs, sem=None):
        self.lhs = lhs
        self.rhs = tuple(rhs.split()) if isinstance(rhs, str) else rhs
        self.sem = sem
        self.validate_rule()

    def __str__(self):
        """Returns a string representation of this Rule."""
        return 'Rule' + str((self.lhs, ' '.join(self.rhs), self.sem))

    def __eq__(self, other):
        return (self.lhs == other.lhs and self.rhs == other.rhs)
    
    def __ne__(self, other):
        return (self.lhs != other.lhs or self.rhs != other.rhs)

    def __hash__(self):
        return hash((self.lhs, self.rhs))

    def apply_semantics(self, sems):
        # Note that this function would not be needed if we required that semantics
        # always be functions, never bare values.  That is, if instead of
        # Rule('$E', 'one', 1) we required Rule('$E', 'one', lambda sems: 1).
        # But that would be cumbersome.
        if isinstance(self.sem, FunctionType):
            return self.sem(sems)
        else:
            return self.sem

    def is_lexical(self):
        """
        Returns true iff the given Rule is a lexical rule, i.e., contains only
        words (terminals) on the RHS.
        """
        return all([not utils.is_cat(rhsi) for rhsi in self.rhs])

    def is_unary(self):
        """
        Returns true iff the given Rule is a unary compositional rule, i.e.,
        contains only a single category (non-terminal) on the RHS.
        """
        return len(self.rhs) == 1 and utils.is_cat(self.rhs[0])

    def is_binary(self):
        """
        Returns true iff the given Rule is a binary compositional rule, i.e.,
        contains exactly two categories (non-terminals) on the RHS.
        """
        return len(self.rhs) == 2 and utils.is_cat(self.rhs[0]) and utils.is_cat(self.rhs[1])

    def validate_rule(self):
        """Returns true iff the given Rule is well-formed."""
        assert utils.is_cat(self.lhs), 'Not a category: %s' % self.lhs
        assert isinstance(self.rhs, tuple), 'Not a tuple: %s' % self.rhs
        for rhs_i in self.rhs:
            assert isinstance(rhs_i, basestring), 'Not a string: %s' % rhs_i

    def contains_optionals(self):
        """Returns true iff the given Rule contains any optional items on the RHS."""
        return any([utils.is_optional(rhsi) for rhsi in self.rhs])