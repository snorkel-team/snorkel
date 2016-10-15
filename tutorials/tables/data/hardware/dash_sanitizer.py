# -*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import os
import glob

def unicode_normalize(text):
    '''
    Replaces various unicode chars with ascii equivalent
    '''
    text = text.replace(u'\u2010', '-') # replace hyphen
    text = text.replace(u'\u2011', '-') # replace non-breaking hyphen
    text = text.replace(u'\u2012', '-') # replace figure dash
    text = text.replace(u'\u2013', '-') # replace en dash
    text = text.replace(u'\u2014', '-') # replace em dash
    text = text.replace(u'\u2212', '-') # replace minus sign
    text = text.encode('ascii','replace')
    return text

def get_files(path):
    if os.path.isfile(path):
        fpaths = [path]
    elif os.path.isdir(path):
        fpaths = [os.path.join(path, f) for f in os.listdir(path)]
    else:
        fpaths = glob.glob(path)
    if len(fpaths) > 0:
        return fpaths
    else:
        raise IOError("File or directory not found: %s" % (path,))

for fp in get_files('/Users/bradenhancock/snorkel/tutorials/tables/data/hardware/hardware1000_html/'):
    f = open(fp,'r')
    filedata = f.read()
    f.close()

    newdata = unicode_normalize(filedata)

    f = open(fp,'w')
    f.write(newdata)
    f.close()


