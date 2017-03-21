import csv
import collections
#extract data for documents in test and dev gold new
'''
test_doc = set()
with open("test/hardware_test_gold.csv", "r") as csvinput:
	reader = csv.reader(csvinput)
	for line in reader:
		test_doc.add(line[0].split("_")[1])
#need to add dev documents - combined with old dev documents
print test_doc
with open("train_digikey/digikey_documents_matching_with_gold.csv", "w") as csvoutput:
	writer = csv.writer(csvoutput, lineterminator="\n")
	with open("train_digikey/hardware_digikey_gold.csv", "r") as csvinput:
		reader = csv.reader(csvinput)
		for line in reader:
			#print line[0]
			if(line[0].upper() in test_doc):
				writer.writerow(line)
'''

#code to see how many part numbers match exactly
digikey_part_num = set()
digikey_part_doc = {}
with open("train_digikey/hardware_digikey_gold.csv", "r") as csvinput:
	reader = csv.reader(csvinput)
	for line in reader:
		digikey_part_num.add(line[1].lower())

test_part_num = set()
with open("gold_parsed/new_total_gold.csv", "r") as csvinput:
	reader = csv.reader(csvinput)
	for line in reader:
		test_part_num.add(line[1].lower())

parts_matched = digikey_part_num.intersection(test_part_num)
print "Part Numbers in Test Data:", len(test_part_num), "Part Numbers exactly overlapping Digikey Data:", len(parts_matched)

test_part_num_man_name = {}
with open("gold_parsed/new_total_gold.csv", "r") as csvinput:
	reader = csv.reader(csvinput)
	for line in reader:
		man_name = line[0].split("_")[0].lower().strip()
		part_num = line[1].lower()
		if(part_num in parts_matched):
			test_part_num_man_name[part_num] = man_name

digikey_part_num_doc_name = {}
digikey_doc_name = set()
parts_doc_matched = set()
with open("train_digikey/hardware_digikey_gold.csv", "r") as csvinput:
	reader = csv.reader(csvinput)
	for line in reader:
		part_num_2 = line[1].lower()
		if(part_num_2 in parts_matched):
			if(line[0][0:5] != "//med"):
				man_name_2 = ((line[0].replace("www.","").replace("https","http")).split("http://")[1]).split(".")[0]
				if((test_part_num_man_name[part_num_2] in man_name_2) or (man_name_2 in test_part_num_man_name[part_num_2])):
					if(part_num_2 not in digikey_part_num_doc_name):
						digikey_part_num_doc_name[part_num_2] = set()
					digikey_part_num_doc_name[part_num_2].add(line[0])
					digikey_doc_name.add(line[0])
					parts_doc_matched.add(part_num_2)

print len(digikey_part_num_doc_name), "distinct Documents found for", len(parts_doc_matched), "part numbers after filtering for Manufacturer Names."

#get all part numbers in the documents!
digikey_part_num = set()
with open("train_digikey/hardware_digikey_gold.csv", "r") as csvinput:
	reader = csv.reader(csvinput)
	for line in reader:
		part_num_2 = line[1].lower()
		if(line[0] in digikey_doc_name):
			digikey_part_num.add(part_num_2)

print "Additional Part Numbers in Matching Digikey Documents:", len(digikey_part_num.difference(test_part_num))


#code to compare numbers for parts which match completely
test_parts_data = {}
with open("gold_parsed/new_total_gold.csv", "r") as csvinput:
	reader = csv.reader(csvinput)
	for line in reader:
		part_num = line[1].lower()
		if(part_num in parts_doc_matched):
			relation = line[2]
			value = line[3]
			if(part_num not in test_parts_data):
				test_parts_data[part_num] = {}
			if(relation not in test_parts_data[part_num]):
				test_parts_data[part_num][relation] = set()
			test_parts_data[part_num][relation].add(value)

digikey_parts_data = {}			
with open("train_digikey/hardware_digikey_gold.csv", "r") as csvinput:
	reader = csv.reader(csvinput)
	for line in reader:
		part_num = line[1].lower()
		if(part_num in parts_doc_matched):
			relation = line[2]
			value = line[3]
			if(part_num not in digikey_parts_data):
				digikey_parts_data[part_num] = {}
			if(relation not in digikey_parts_data[part_num]):
				digikey_parts_data[part_num][relation] = set()
			digikey_parts_data[part_num][relation].add(value)

relations_not_in_digikey = collections.defaultdict(int)
relations_match = collections.defaultdict(int)
relations_partial_match = collections.defaultdict(int)
relations_no_match = collections.defaultdict(int)
unique_relations = set()

for part_num in test_parts_data:
	test_data = test_parts_data[part_num]
	digikey_data = digikey_parts_data[part_num]
	for relation in test_data:
		unique_relations.add(relation)
		#code to check problem with Current
		'''
		if(relation == "c_current_max"):
			if(relation in digikey_data):
				if(set(test_data[relation]) != set(digikey_data[relation])):
					print set(test_data[relation]), set(digikey_data[relation])	
		'''
		if relation not in digikey_data:
			relations_not_in_digikey[relation] = relations_not_in_digikey[relation] + 1
		elif(set(test_data[relation]) == set(digikey_data[relation])):
			relations_match[relation] = relations_match[relation] + 1
		elif(len(set(test_data[relation]).intersection(set(digikey_data[relation])))>0):
			relations_partial_match[relation] = relations_partial_match[relation] + 1
		else:
			relations_no_match[relation] = relations_no_match[relation] + 1
			
for relation in unique_relations:
	sum_relation = relations_not_in_digikey[relation] + relations_match[relation] + relations_partial_match[relation] + relations_no_match[relation]
	print relation, "Relations from Test Data Not in Digikey:", round(float(relations_not_in_digikey[relation])/sum_relation,2), "Exact Matches:", round(float(relations_match[relation])/sum_relation,2), "Partial Matches:", round(float(relations_partial_match[relation])/sum_relation,2), "No Matches:", round(float(relations_no_match[relation])/sum_relation,2)


'''
test_part_num_doc_name = {}
gold_digikey_matches = {}
with open("gold_parsed/new_total_gold.csv", "r") as csvinput:
	reader = csv.reader(csvinput)
	for line in reader:
		part_num = line[1].lower()
		if(part_num not in gold_digikey_matches):
			gold_digikey_matches[part_num] = set()
			test_part_num_doc_name[part_num] = [line[0], False, "", float("Inf")]

#print len(test_part_num_doc_name), len(gold_digikey_matches)

with open("train_digikey/hardware_digikey_gold.csv", "r") as csvinput:
	reader = csv.reader(csvinput)
	for line in reader:
		part_num_2 = line[1].lower()
		for part_num in test_part_num_doc_name:
			part_num_iter = test_part_num_doc_name[part_num]
			if(part_num == part_num_2):
				part_num_iter[1] = True
				part_num_iter[2] = part_num_2
				gold_digikey_matches[part_num].add(("Exact:", part_num_2, line[0]))
			elif((part_num in part_num_2)):# or (part_num.split("-")[0] in part_num_2)):
				if(part_num_iter[1] == False):
					num_extra_2 = len(part_num_2.replace(part_num,""))
					if(num_extra_2 < part_num_iter[3]):
						part_num_iter[2] = part_num_2
						part_num_iter[3] = num_extra_2
				gold_digikey_matches[part_num].add(("Partial in Digikey:", part_num_2, line[0]))
			test_part_num_doc_name[part_num] = part_num_iter
				
no_match = 0
for part_num in gold_digikey_matches:
	if(len(gold_digikey_matches[part_num])) == 0:
		no_match = no_match + 1
	print part_num, test_part_num_doc_name[part_num][0], gold_digikey_matches[part_num]
print "No match: ", no_match
'''