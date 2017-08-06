from snorkel import SnorkelSession
from elasticsearch import helpers, Elasticsearch, client

#intialize snorkel and elasticsearch
session = SnorkelSession()
es = Elasticsearch()


count = 0

for p in session.query(Document):
	for i in p.sentences:
		count+=1
		#index every sentence
		es.index(index="corpus", doc_type="articles", id=count, body = {
		                'lineNum': count,
		                'sentence': i.text,
		            })
#search query 
searchResult = es.search(index = "corpus",body={
	"query": {
		"match": {
			"sentence": {
				"query":"married children" 
				}
		}
	}
	})
#print the sentence that matches query
print "Results : "
print searchResult['hits']['total']
for items in searchResult['hits']['hits']:
	print "-------------------"
	print items['_source']['sentence']

