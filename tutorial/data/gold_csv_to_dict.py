#!/usr/bin/python

import csv
with open('gold_stg_temp_min.csv', 'rb') as csvfile:
    gold_reader = csv.reader(csvfile)
    gold_dict = {}
    for row in gold_reader:
        (doc, part, temp, label) = row
        if label=='stg_temp_min':
            gold_dict[(doc,temp)] = 1
print len(gold_dict)

