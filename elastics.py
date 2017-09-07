from elasticsearch import Elasticsearch,client 
from snorkel import SnorkelSession
from snorkel.models import Document, Sentence,Span
from snorkel.viewer import SentenceNgramViewer
import os 
import json
es = Elasticsearch()
session = SnorkelSession()

class ElasticSession:
	#define document and index names
	def __init__(self,**keyword_parameters):
		self.indexName = "corpus"
		self.docType = "articles"
		self.fieldName = "sentence"
		self.elastic_index()
		if "cands" in keyword_parameters:
			self.generate_tags(keyword_parameters['cands'])

	def set_cand(self, Cands): 
		self.cands = Cands
	#get the index mapping
	def get_map(self):
		mapping = es.indices.get_mapping(self.indexName)
		print 'Index Mapping'
		print(json.dumps(mapping, indent=2))

	#get all index information
	def get_index(self):
		print 'Index Information: '
		print ' '
		print es.cat.indices(v='true')

	#get a document by its id number
	def get_doc(self,iden):
		return es.get(index=self.indexName, doc_type=self.docType, id=iden)

	#Elasticsearch to SQL mapping
	#Index - Database
	#Table - Type
	#Row - Document
	#Values are the data to be added to each document
	def elastic_index(self):
		#Define our index mapping
		request_body = {
			'settings' : {
				'number_of_shards': 5,
				'number_of_replicas': 1,
				'analysis':{
					'char_filter': { 
						'quotes': {
							#Standardize apostrophes
							'type': 'mapping',
							'mappings': [ 
								'\u0091=>\u0027',
								'\u0092=>\u0027',
								'\u2018=>\u0027',
								'\u2019=>\u0027',
								'\u201B=>\u0027'
							]
						}
					},
					'analyzer':{
						'my_analyzer':{
							'type':'custom',
							'tokenizer':'standard',
							'char_filter': ['quotes'],
							#Remove apostrophes and perform asciifolding
							'filter':['apostrophe','asciifolding']
						},
						#used to remove the unicode marker
						'my_stop': {
							'type':'stop',
							'stopwords': ['u']
						}
					}
				}
			},
			#define field properties
			'mappings': {
				self.docType: {
					'properties': {
						'lineNum':{'type':'integer'},
						self.fieldName: {'type': 'text','analyzer':'my_analyzer'},
						'tagged':{'type':'text','analyzer':'my_stop'},
						'fillCand':{'type':'text','analyzer':'my_stop','search_analyzer':'my_stop'}
					}}}}
		#create the index
		es.indices.create(index = self.indexName, body = request_body)
		print 'Begin indexing'
		docCount=0
		for p in session.query(Document):
			docCount+=1
			for i in p.sentences:
				#analyze the string and create an array of that length of o's
				#this will be used for the candidate layer
				value=len((es.indices.analyze(index=self.indexName,body={'analyzer':'standard','text':i.text}))['tokens'])
				es.index(index=self.indexName, doc_type=self.docType, id=i.id,
					body = {
						'lineNum': i.id,
						self.fieldName: i.text,
						'fillCand':['o']*value
					})
		self.get_index()
		print '%d items indexed'%docCount
		print ""
		

	def generate_tags(self,Cands):
		self.set_cand(Cands)

		print "Begin generating tags"
		unique=[]
		total=0
		#Get all the sentences in our candidate set
		for c in session.query(Cands).all():
			total+=1
			unique.append(c[0].sentence_id)
		#Turn it into a set to get only the unique sentences
		unique = set(unique)
		#Used to keep tracking of the candidates that could not be tagged
		flagNum=0
		flagged=[]
		for sent in unique:
			#Get all candidates that correspond to a particular sentence
			q = session.query(Cands)\
			.join(Span, getattr(Cands, Cands.__argnames__[0] + '_id') == Span.id)\
			.join(Span.sentence).filter(Sentence.id == sent).all()

			#Get the term vector of the sentence. We will use this to determine
			#where the candidate is in the sentence
			vector=es.termvectors(
				index=self.indexName,
				doc_type=self.docType,
				id=sent,
				body ={
				'fields' : [self.fieldName],
				'positions' : 'true'
			})
			temp = []
			for p in q:
				for num in range(0,2):
					candidate= p[num].get_span()
					#Candidates can be more the one word so we asciifold and split the candidates
					#on the spaces
					value=es.indices.analyze(index=self.indexName,body={'analyzer':'my_analyzer','text':candidate})['tokens']
					for vectorized in value:
						temp.append(vectorized['token'])
			#Get the candidate array that we will modify
			hold=es.get(index=self.indexName, doc_type='articles', id=sent)['_source']['fillCand']
			for tagObj in temp:
				try:
					#Candidates can appear multiple times in a sentence so we get the
					#total number of occurances
					limit = vector['term_vectors'][self.fieldName]['terms'][tagObj]['term_freq']
					for i in range(0,limit):
						#Find the candidate position and tag that index
						index=vector['term_vectors'][self.fieldName]['terms'][tagObj]['tokens'][i]['position']
						hold[index]='OBJECT'
				#Used to handle candidates that could not be found
				except KeyError:
					flagNum+=1
					flagged.append([sent,tagObj])
			#Arrays have an implicit 100 positional gap between indices which 
			#make the search queries behave weirdly. To compensate we change 
			#the array to a string and add it to a new field.
			turnAr = ' '.join((e).decode('utf-8') for e in hold)
			es.update(index=self.indexName, doc_type=self.docType, id=sent,
				body={'doc':{'fillCand':hold,'tagged':turnAr}})

		#Most candidates that can not be tagged are ones that correspond to punctuation and spaces
		#those are automatically stripped when the string is tokenized
		print '%d candidates of %d tagged'%((total-flagNum),(total))

	def search_index(self,keyWord,*args,**keyword_parameters):
		check = 0

		if keyWord == 'match':
			for hold,query in enumerate(args):
				#Match phrase if there is a slop value
				if 'slop' in keyword_parameters:
					sQuery={
						'match_phrase':{
							self.fieldName:{
								'query':query,
								'slop':keyword_parameters['slop']
							}
						}
					}
				else:
					#Match query if no slop is defined
					sQuery={
						'match': {
							self.fieldName: {
								'query':query
								}
						}
					}
		#Query a specific field where we can about the order
		#position(value1)<position(value2)<position(value3) etc
		elif keyWord=='position':
			holdVal=[]
			if 'slop' in keyword_parameters:
				dist = keyword_parameters['slop']
			else:
				dist=0

			for hold,values in enumerate(args):
				holdVal.append({ 'span_term' : { self.fieldName : values } })
			sQuery={
				'span_near' : {
					'clauses' : holdVal,
					'slop' : dist,
					'in_order' : 'true'
			}
			}
		#Query two fields in parallel respective of order
	 	#the mask searches the tagged for object then switches to the fieldName to search for value
	 	#before switching back to tagged to search for object again
		elif keyWord=='between_cand':
			check=1
			if 'slop' in keyword_parameters:
				dist = keyword_parameters['slop']
			else:
				dist=0
			for hold,value in enumerate(args):
				sQuery={
					'span_near': {
						'clauses': [
							{'span_term': {'tagged': 'object'}},
							{'field_masking_span': {
								'query': {
									'span_term': {
										self.fieldName: value
									}
								},
							'field': 'tagged'}
							},
							{'span_term': {'tagged': 'object'}},
						],
					'slop': dist,
					'in_order': 'true'
					}
				}
		#Query two fields in parallel respective of order
		#Searches the fieldName first for the value then switches to the tagged 
		#field for the OBJECT tag	
		elif keyWord == 'before_cand':
			check=1
			holdVal=[]
			if 'slop' in keyword_parameters:
				dist = keyword_parameters['slop']
			else:
				dist=0

			for hold,values in enumerate(args):	
				sQuery={
					'span_near': {
						'clauses': [
							{'span_term': 
								{ self.fieldName : values }},
							{'field_masking_span': {
								'query': {
									'span_term': {
										'tagged': 'object'
									}
								},
							'field': self.fieldName}
							}
						],
					'slop': dist,
					'in_order': 'true'
					}
				}
		#Query two fields in parallel respective of order
		#Searches the tagged field first for object then switches to the fieldName
		#for the value
		elif keyWord == 'after_cand':
			check=1
			if 'slop' in keyword_parameters:
				dist = keyword_parameters['slop']
			else:
				dist=0

			for hold,values in enumerate(args):
				sQuery={
					'span_near': {
						'clauses': [
							{'span_term':
								{'tagged': 'object'}},
							{'field_masking_span': {
								'query': {
									'span_term': {
										 self.fieldName : values
									}
								},
							'field': 'tagged'}
							}
						],
					'slop': dist,
					'in_order': 'true'
					}
				}

		else:
			print 'QUERY TYPE NOT FOUND'
			return

		#Size indicates how many search results to return		
		if 'size' in keyword_parameters:
			numRes = keyword_parameters['size']
		else:
			numRes=5

		#Perform the query
		searchResult = es.search(
			size =numRes,
			index=self.indexName,
			doc_type=self.docType,
			body={
				'query': sQuery
			})

		
		temp=[]
		print "Number of hits: %d" %searchResult['hits']['total']
		#get sentence numbers from the search results
		for i in searchResult['hits']['hits']:
		    temp.append(i['_source']['lineNum'])
		holdCands=[]
		if check ==1:
			for i in temp:
				#query the candidate set for all spans with the sentence number
			    q = session.query(self.cands)\
			        .join(Span, getattr(self.cands, self.cands.__argnames__[0] + '_id') == Span.id)\
			        .join(Span.sentence).filter(Sentence.id == i).all()
			    for span in q:
			    	holdCands.append(span)
		else:
			for i in temp:
				#get sentence using sentence number
			    q=session.query(Sentence).filter(Sentence.id ==i).all()
			    holdCands.append(q[0])

		#returns candidate object
		return holdCands
		

#deletes an elasticsearch index taking the index name as a parameter 
#the _all flag will delete all indecies
def delete_index(indexName):
	print es.indices.delete(index=indexName,ignore=404)
