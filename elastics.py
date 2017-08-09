from elasticsearch import Elasticsearch,client 
import json
es = Elasticsearch()


class elasticSession:
	#define document and index names
	def __init__(self,indexName,docType,):
		self.indexName = indexName
		self.docType = docType

	#get the index mapping
	def getIndexMap(self):
		mapping = es.indices.get_mapping(self.indexName)
		print "Index Mapping"
		print(json.dumps(mapping, indent=2))

	#get all index information
	def getIndices(self):
		print "Index Information: "
		print " "
		print es.cat.indices(v="true")

	#get a document by its id number
	def getDoc(self,iden):
		return es.get(index=self.indexName, doc_type=self.docType, id=iden)

	#Elasticsearch to SQL mapping
	#Index - Database
	#Table - Type
	#Row - Document
	#Values are the data to be added to each document
	def elasticIndex(self,idNum,values):
		es.index(index=self.indexName, doc_type=self.docType, id=idNum, body =values)
		
	#Query a specific field for some string
	#Queries that sentences have an implicit OR in between
	def searchIndex(self,fieldName,query):
		searchResult = es.search(
		index=self.indexName,
		doc_type=self.docType,
		body={
			"query": {
				"match": {
					fieldName: {
						"query":query
						}
				}
			}
		})
		return searchResult

	#Query a specific field where we can about the order
	#position(value1)<position(value2)<position(value3)
	def searchOrder(self,fieldName,distance,value1,value2,value3):
		searchResult = es.search(
		index=self.indexName,
		doc_type=self.docType,
		body={
		    "query": {
		        "span_near" : {
		            "clauses" : [
		                { "span_term" : { fieldName : value1 } },
		                { "span_term" : { fieldName : value2 } },
		                { "span_term" : { fieldName : value3 } }
		            ],
		            "slop" : distance,
		            "in_order" : "true"
	        	}
	    	}
		})

		return searchResult

	#Query two fields in parallel respective of order
	#the mask searches tagField for tags then switches to the worField to search for value
	#before switching back to the tagField to search for the tags again
	def searchBetweenCandidates(self,tagField,tags,worField,value,distance):
		searchResult = es.search(
		index=self.indexName,
		doc_type=self.docType,
		body={
		  "query": {
		    "span_near": {
		      "clauses": [
		        {
		          "span_term": {
		            tagField: tags
		          }
		        },
		        {
		         "field_masking_span": {
		            "query": {
		            	"span_term": {
		           			worField: value
		          		}
		            },
		            "field": tagField
		          }
		        },
		        {
		          "span_term": {
		            tagField: tags
		          }
		        },
		      ],
		      "slop": distance,
		      "in_order": "true"
		    }
		  }
		})

		return searchResult
#print the results of an elasticsearch query
def printResults(searchResult,*args):
	hitCount=searchResult['hits']['total']
	if hitCount>0:
		print "Number of hits "
		print hitCount
		i=1
		for items in searchResult['hits']['hits']:
			print "Result %d" %(i)
			i+=1
			print "-------------------"
			for hold,fields in enumerate(args):
				print fields
				print items['_source'][fields]
	else:
		print "No hits"
#deletes an elasticsearch index taking the index name as a parameter 
#the _all flag will delete all indecies
def deleteIndex(indexName):
	print es.indices.delete(index=indexName,ignore=404)