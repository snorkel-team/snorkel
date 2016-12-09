import csv

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
with open("test/hardware_test_gold.csv", "r") as csvinput:
	reader = csv.reader(csvinput)
	for line in reader:
		test_part_num.add(line[1].lower())

parts_matched = digikey_part_num.intersection(test_part_num)
print "Part Numbers in Test Data:", len(test_part_num), "Part Numbers exactly overlapping Digikey Data:", len(parts_matched)
#print test_part_num.difference(digikey_part_num)
#code to compare numbers for parts which match completely

test_digikey_map = {}
test_digikey_array = []
bool_partial_match = 0
for part_num in test_part_num:
	part_match = False
	matching_part_num = ""
	num_extra_2 = 0
	min_extra = float("Inf")
	for part_num_2 in digikey_part_num:
		if(part_num == part_num_2):
			part_match = True
			matching_part_num = part_num_2
		elif(part_num in part_num_2):
			if(part_match == False):
				num_extra_2 = len(part_num_2.replace(part_num,""))
				if(num_extra_2 < min_extra):
					matching_part_num = part_num_2
	if(matching_part_num != ""):
		#print part_num, matching_part_num
		test_digikey_map[part_num] = matching_part_num
		test_digikey_array.append(matching_part_num)
		bool_partial_match = bool_partial_match + 1
		
print bool_partial_match

test_parts_data = {}
with open("test/hardware_test_gold.csv", "r") as csvinput:
	reader = csv.reader(csvinput)
	for line in reader:
		part_num = line[1].lower()
		#if(part_num in parts_matched):
		if(part_num in test_digikey_map):
			relation = line[2]
			value = line[3]
			if(part_num not in test_parts_data):
				test_parts_data[part_num] = {}
			if(relation not in test_parts_data[part_num]):
				test_parts_data[part_num][relation] = []
			test_parts_data[part_num][relation].append(value)

digikey_parts_data = {}			
with open("train_digikey/hardware_digikey_gold.csv", "r") as csvinput:
	reader = csv.reader(csvinput)
	for line in reader:
		part_num = line[1].lower()
		#if(part_num in parts_matched):
		if(part_num in test_digikey_array):
			relation = line[2]
			value = line[3]
			if(part_num not in digikey_parts_data):
				digikey_parts_data[part_num] = {}
			if(relation not in digikey_parts_data[part_num]):
				digikey_parts_data[part_num][relation] = []
			digikey_parts_data[part_num][relation].append(value)

relations_not_in_digikey = {}
relations_match = {}
relations_partial_match = {}
relations_no_match = {}
for part_num in test_parts_data:
	test_data = test_parts_data[part_num]
	#digikey_data = digikey_parts_data[part_num]
	digikey_data = digikey_parts_data[test_digikey_map[part_num]]
	for relation in test_data:
		if relation not in digikey_data:
			if(relation not in relations_not_in_digikey):
				relations_not_in_digikey[relation] = 0
			relations_not_in_digikey[relation] = relations_not_in_digikey[relation] + 1
		elif(set(test_data[relation]) == set(digikey_data[relation])):
			#print test_data[relation], digikey_data[relation], set(sorted(test_data[relation])), set(sorted(digikey_data[relation]))
			if(relation not in relations_match):
				relations_match[relation] = 0
			relations_match[relation] = relations_match[relation] + 1
		elif(len(set(test_data[relation]).intersection(set(digikey_data[relation])))>0):
			if(relation not in relations_partial_match):
				relations_partial_match[relation] = 0
			relations_partial_match[relation] = relations_partial_match[relation] + 1
			#print relation, set(sorted(test_data[relation])), set(sorted(digikey_data[relation]))
		else:
			if(relation not in relations_no_match):
				relations_no_match[relation] = 0
			relations_no_match[relation] = relations_no_match[relation] + 1
			#print relation, set(sorted(test_data[relation])), set(sorted(digikey_data[relation]))
print "Relations from Test Data Not in Digikey:", relations_not_in_digikey
print "Exact Matches:", relations_match
print "Partial Matches:", relations_partial_match
print "No Matches:", relations_no_match