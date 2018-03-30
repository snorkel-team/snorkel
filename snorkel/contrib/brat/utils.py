from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

from six.moves.urllib.request import urlopen
from six.moves.urllib.error import URLError, HTTPError

def download(url, outfname):
    """
    Download target URL

    :param url:
    :param outfname:
    :return:
    """
    try:
        data = urlopen(url)
        with open(outfname, "wb") as f:
            f.write(data.read())
    except HTTPError as e:
        print("HTTP Error: {} {}".format(e.code, url))
    except URLError as e:
        print("URL Error: {} {}".format(e.reason, url))
