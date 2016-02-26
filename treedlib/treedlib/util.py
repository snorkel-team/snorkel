from collections import namedtuple
import re
import sys


def print_gen(gen):
  """Print the results of a generator one-per-line"""
  for e in gen:
    print e

def print_error(err_string):
  """Function to write to stderr"""
  sys.stderr.write("ERROR[UDF]: " + str(err_string) + "\n")


BOOL_PARSER = {
  't' : True,
  'f' : False,
  'NULL' : None,
  '\\N' : None
}
TYPE_PARSERS = {
  'text' : lambda x : str(x.replace('\n', ' ')),
  'int' : lambda x : int(x.strip()),
  'float' : lambda x : float(x.strip()),
  'boolean' : lambda x : BOOL_PARSER(x.lower().strip())
}

def parse_ptsv_element(s, t, sep='|^|', sep2='|~|'):
  """
  Parse an element in psql-compatible tsv format, i.e. {-format arrays
  based on provided type and type-parser dictionary
  """
  # Interpret null first regardless of type
  if len(s) == 0 or s == '\\N':
    return None

  # Handle lists recursively first
  elif '[]' in t:
    if re.search(r'^\{|\}$', s):
      split = re.split(r'\"?\s*,\s*\"?', re.sub(r'^\{\s*\"?|\"?\s*\}$', '', s))
    else:
      split = s.split(sep)
    return [parse_ptsv_element(ss, t[:-2], sep=sep2) for ss in split]
  
  # Else parse using parser 
  else:
    try:
      parser = TYPE_PARSERS[t]
    except KeyError:
      raise Exception("Unsupported type: %s" % t)
    return parser(s)


class Row:
  def __str__(self):
    return '<Row(' + ', '.join("%s=%s" % x for x in self.__dict__.iteritems()) + ')>'

  def __repr__(self):
    return str(self)

  def _asdict(self):
    return self.__dict__


class PTSVParser:
  """
  Initialized with a list of duples (field_name, field_type)
  Is a factory for simple Row class
  Parsed from Postgres-style TSV input lines
  """
  def __init__(self, fields):
    self.fields = fields
    self.n = len(fields)

  def parse_line(self, line):
    row = Row()
    attribs = line.rstrip().split('\t')
    if len(attribs) != self.n:
      raise ValueError("%s attributes for %s fields:\n%s" % (len(attribs), self.n, line))
    for i,attrib in enumerate(attribs):
      field_name, field_type = self.fields[i]
      setattr(row, field_name, parse_ptsv_element(attrib, field_type))
    return row

  def parse_stdin(self):
    for line in sys.stdin:
      yield self.parse_line(line)


def pg_array_escape(tok):
  """
  Escape a string that's meant to be in a Postgres array.
  We double-quote the string and escape backslashes and double-quotes.
  """
  return '"%s"' % str(tok).replace('\\', '\\\\').replace('"', '\\\\"')

def list_to_pg_array(l):
  """Convert a list to a string that PostgreSQL's COPY FROM understands."""
  return '{%s}' % ','.join(pg_array_escape(x) for x in l)

def print_tsv(out_record):
  """Print a tuple as output of TSV extractor."""
  values = []
  for x in out_record:
    if isinstance(x, list) or isinstance(x, tuple):
      cur_val = list_to_pg_array(x)
    elif x is None:
      cur_val = '\N'
    else:
      cur_val = x
    values.append(cur_val)
  print '\t'.join(str(x) for x in values)
