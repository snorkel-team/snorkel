'''
@author: henryre
'''

import urllib2, sys
from numpy.random import choice

class PubMedArticle:
    def __init__(self, uid):
        self.uid = uid
        self.url = 'http://ncbi.nlm.nih.gov/pubmed/?term={}'.format(uid)
        self.html = None
    
    def get_html(self):
        response = urllib2.urlopen(urllib2.Request(self.url))
        try:        
            self.html = response.read()
        except:
            print "Could not get article {} from PubMed".format(self.uid)
            print sys.exc_info()[0]
        finally:
            response.close()
    
    def write_html(self, fp='pubmed_article.html'):
        if self.html is None:
            self.get_html()
        if self.html is not None:
            with open(fp, 'w+') as f:
                f.write(self.html)
            print "Wrote article {} to {}".format(uid, fp)

if __name__ == '__main__':
    
    max_size = 150
    with open('data/gene_pmids.tsv', 'rb') as f:
        pmids = [line.strip() for line in f]
    
    if len(pmids) > max_size:
        pmids = choice(pmids, max_size)
    print "Fetching {} pubmed articles".format(len(pmids))
    
    for uid in pmids:
        pma = PubMedArticle(uid)
        pma.write_html('data/ddlite_gene_ex/{}.html'.format(uid))
    
    