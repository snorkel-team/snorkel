
from urllib2 import urlopen, URLError, HTTPError

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
    except HTTPError, e:
        print "HTTP Error:", e.code, url
    except URLError, e:
        print "URL Error:", e.reason, url