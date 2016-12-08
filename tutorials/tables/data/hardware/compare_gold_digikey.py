import csv

#code to see how many part numbers match exactly
digikey_part_num = set()
with open("train_digikey/hardware_digikey_gold.csv", "r") as csvinput:
	reader = csv.reader(csvinput)
	for line in reader:
		digikey_part_num.add(line[1].lower())

test_part_num = set()
with open("test/hardware_test_gold.csv", "r") as csvinput:
	reader = csv.reader(csvinput)
	for line in reader:
		test_part_num.add(line[1].lower())

print len(test_part_num), len(digikey_part_num.intersection(test_part_num))

#code to compare numbers for parts which match completely
parts_matched = digikey_part_num.intersection(test_part_num)

test_parts_data = {}
with open("test/hardware_test_gold.csv", "r") as csvinput:
	reader = csv.reader(csvinput)
	for line in reader:
		part_num = line[1].lower()
		if(part_num in parts_matched):
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
		if(part_num in parts_matched):
			relation = line[2]
			value = line[3]
			if(part_num not in digikey_parts_data):
				digikey_parts_data[part_num] = {}
			if(relation not in digikey_parts_data[part_num]):
				digikey_parts_data[part_num][relation] = []
			digikey_parts_data[part_num][relation].append(value)

relation_not_in_digikey = 0
exact_match_with_digikey = 0
total_relations = 0
total_values = 0
for part_num in test_parts_data:
	test_data = test_parts_data[part_num]
	digikey_data = digikey_parts_data[part_num]
	for relation in test_data:
		total_relations = total_relations + 1
		if relation not in digikey_data:
			relation_not_in_digikey = relation_not_in_digikey + 1
			#print relation, digikey_data
			continue
		if(test_data[relation].sort() == digikey_data[relation].sort()):
			exact_match_with_digikey = exact_match_with_digikey + len(test_data[relation])
		total_values = total_values + len(test_data[relation])
print relation_not_in_digikey, total_relations, exact_match_with_digikey, total_values