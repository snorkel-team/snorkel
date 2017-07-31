
from snorkel import SnorkelSession
from snorkel.models import Document, Sentence
from elasticsearch import helpers, Elasticsearch
import math

session = SnorkelSession()
es = Elasticsearch()

h=500.0
numDocs = session.query(Document).count()
iterations=  int(math.ceil(numDocs/h))

for j in range(0,iterations):	
	print j
	actions = [
	{
	    "_index": j,
	    "_type": p.id,
		"sentence":i.text,

	}
	for p in session.query(Document).limit(h).offset(h*j) for i in p.sentences
	]
	helpers.bulk(es, actions,request_timeout=1000)

searchResult = es.search(index = "_all",body={
	"query": {
		"match_phrase": {
			"sentence": {
				"query":"Tina married Mathew",
				"slop":10
				}
		}
	}
	})

print "Results : "
print searchResult['hits']['total']
for items in searchResult['hits']['hits']:
	print "-------------------"
	print items['_source']['sentence']
