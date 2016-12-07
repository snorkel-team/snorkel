import csv
import os
import itertools
from hardware_normalizers import *

polarity_values = ["NPN", "PNP"]
with open("DigiKey_Hardware_Relations.csv", "w") as csvoutput:
    writer = csv.writer(csvoutput, lineterminator="\n")
    for root, subFolders, files in os.walk('single-bjt'):
        for file in files:
            print os.path.join(root, file)
            with open(os.path.join(root, file), "r") as csvinput:
                reader = csv.reader(csvinput)
                next(reader, None)
                for line in reader:
                	if(len(line) == 27):
                		print "error"
                	if(len(line) == 28):
                		add_index = 2
                	else:
                		add_index = 0
                	if len(line) == 26:
	                	pdf_name = line[0].split("/")[-1]
	                	part_num = line[3]
	                	polarity = line[14]
	                	new_polarity = ""
	                	for val in polarity_values:
	                		if(val in polarity):
	                			new_polarity = new_polarity+val+";"
	                	polarity = new_polarity
	                	c_current_max = line[15]
	                	ce_v_max = line[16]
	                	ce_v_max = "".join(itertools.takewhile(str.isdigit, ce_v_max))
	                	dc_gain_min = line[19 + add_index]
	                	dev_dissipation = line[20 + add_index]
	                	stg_temp = line[22 + add_index]
	                	print stg_temp
	                	if('~' in stg_temp):
	                		stg_temp_min = stg_temp.split("~")[0].strip()[1:]
	                		stg_temp_min = stg_temp.split("~")[0].strip()[0] + "".join(itertools.takewhile(str.isdigit, stg_temp_min)) + " "
	                		stg_temp_max = stg_temp.split("~")[1].strip()
	                		stg_temp_max = "".join(itertools.takewhile(str.isdigit, stg_temp_max)) + " " 
	                	else:
	                		stg_temp_min = " "
	                		stg_temp_max = stg_temp.strip()
	                		stg_temp_max = "".join(itertools.takewhile(str.isdigit, stg_temp_max)) + " "
	                	name_attr_norm = [
                        	('polarity', polarity, polarity_normalizer),
                        	('ce_v_max', ce_v_max, voltage_normalizer),
                        	('c_current_max', c_current_max, current_normalizer),
                        	('dev_dissipation', dev_dissipation, dissipation_normalizer),
                        	('stg_temp_min', stg_temp_min, temperature_normalizer),
                        	('stg_temp_max', stg_temp_max, temperature_normalizer),
                        	('dc_gain_min', dc_gain_min, gain_normalizer)]
                        for name, attr, normalizer in name_attr_norm:
							if 'N/A' not in attr and attr.strip() not in  ["-","*"]:
								print part_num, attr
								for a in attr.split(';'):
									if len(a.strip())>0:
										if(name == "c_current_max" or name == "dev_dissipation"):
											a = "".join(itertools.takewhile(str.isdigit, a))
										writer.writerow([pdf_name, part_num, name, normalizer(a)])

	                	#break
        '''
        with open(os.path.join(root, file):

		with open("Hardware Gold - Combined Annotations.csv", "r") as csvinput:
		reader = csv.reader(csvinput)
		next(reader, None)
		for line in reader:
			doc_name = line[0]
			part_num = line[2]
			polarity = line[4]
			v_ceo = line[5]
			v_cbo = line[6]
			v_ebo = line[7]
			current = line[8]
			dev_dissipation = line[9]
			min_temp = line[10]
			max_temp = line[11]
			dev_gain = line[12]
			if "N/A" not in polarity:
				writer.writerow([doc_name, part_num, "Polarity", polarity_normalizer(polarity)])
			if "N/A" not in v_ceo:
				writer.writerow([doc_name, part_num, "Collector-Emitter Voltage Max", voltage_normalizer(v_ceo)])
			if "N/A" not in v_cbo:
				writer.writerow([doc_name, part_num, "Collector-Base Voltage Max", voltage_normalizer(v_cbo)])
			if "N/A" not in v_ebo:
				writer.writerow([doc_name, part_num, "Emitter-Base Voltage Max", voltage_normalizer(v_ebo)])
			if "N/A" not in current:
				writer.writerow([doc_name, part_num, "Collector Current Continuous Max", current_normalizer(current)])
			if "N/A" not in dev_dissipation:
				for dissipation in dev_dissipation.split(";"):
					writer.writerow([doc_name, part_num, "Total Power Dissipation", dissipation_normalizer(dissipation)])
			if "N/A" not in min_temp:
				writer.writerow([doc_name, part_num, "Storage Temperature Min", temperature_normalizer(min_temp)])
			if "N/A" not in max_temp:
				writer.writerow([doc_name, part_num, "Storage Temperature Max", temperature_normalizer(max_temp)])
			if "N/A" not in dev_gain:
				for gain in dev_gain.split(";"):
					if("@" in gain):
						writer.writerow([doc_name, part_num, "DC Current Gain Min", gain_normalizer(gain)])
		'''


