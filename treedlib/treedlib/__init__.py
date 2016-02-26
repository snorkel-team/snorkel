import os

# Set TREEDLIB_APP env var, for use in libs we load
os.environ["TREEDLIB_LIB"] = os.path.dirname(os.path.realpath(__file__))

# Load treedlib libs
from util import *
from structs import *
from templates import *
from features import *
