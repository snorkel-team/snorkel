from ....matchers import Matcher


class FigureMatcher(Matcher):
    """Matcher base class for Figure objects"""
    def _is_subspan(self, c, span):
        """Tests if candidate c does exist"""
        return c.figure.document.id == span[0] and c.figure.position == span[1]

    def _get_span(self, c):
        """Gets a tuple that identifies a figure for the specific candidate class that c belongs to"""
        return (c.figure.document.id, c.figure.position)


class LambdaFunctionFigureMatcher(FigureMatcher):
    """Selects candidate Figures that return True when fed to a function f."""
    def init(self):
        try:
            self.func = self.opts['func']
        except KeyError:
            raise Exception("Please supply a function f as func=f.")
    
    def _f(self, c):
        """The internal (non-composed) version of filter function f"""
        return self.func(c)
