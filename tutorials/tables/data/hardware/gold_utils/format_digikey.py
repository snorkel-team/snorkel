import csv
import os
import itertools
from hardware_normalizers import *

def format_digikey_gold(raw_dir, with_name, without_name):
	with_pdf_name = []
	without_pdf_name = []
	polarity_values = ["NPN", "PNP"]
	for root, subFolders, files in os.walk(raw_dir):
		for file in files:
			print os.path.join(root, file)
			with open(os.path.join(root, file), "r") as csvinput:
				reader = csv.reader(csvinput)
				next(reader, None)
				for line in reader:
					if(len(line) == 27):
						add_index = 1
					if(len(line) == 28):	#Handling Data With Populated Resistor Entries
						add_index = 2
					else:
						add_index = 0
					if len(line) >= 26:		#Most of the entries have length 26, some have 28; entries with less than that are discarded
						pdf_name = line[0]
						if("=" in pdf_name):
							pdf_name = line[0].split("=")[-1]
						else:
							pdf_name = line[0].split("/")[-1]
						part_num = line[3]
						polarity = line[14]
						new_polarity = ""	#Restricting Values to NPN and PNP only to avoid errors in polarity normalizer function
						for val in polarity_values:
							if(val in polarity):
								new_polarity = new_polarity+val+";"
						polarity = new_polarity
						c_current_max = line[15]
						ce_v_max = line[16]
						ce_v_max = "".join(itertools.takewhile(str.isdigit, ce_v_max))	#Remove Units
						dc_gain_min = line[19 + add_index]
						dc_gain_min = "".join(itertools.takewhile(str.isdigit, ce_v_max))
						dev_dissipation = line[20 + add_index]
						stg_temp = line[22 + add_index]
						if('~' in stg_temp):
							stg_temp_min = stg_temp.split("~")[0].strip()[1:]
							stg_temp_min = stg_temp.split("~")[0].strip()[0] + "".join(itertools.takewhile(str.isdigit, stg_temp_min)) + " "
							stg_temp_max = stg_temp.split("~")[1].strip()
							stg_temp_max = "".join(itertools.takewhile(str.isdigit, stg_temp_max)) + " " 
						else:		#Just Max Value present
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
								# print part_num, attr
								for a in attr.split(';'):
									if len(a.strip())>0:
										if(name == "c_current_max" or name == "dev_dissipation"):
											a = "".join(itertools.takewhile(str.isdigit, a))	#Remove Units
										if(pdf_name != "-"):
											with_pdf_name.append([pdf_name, part_num, name, normalizer(a)])
										else:
											without_pdf_name.append([pdf_name, part_num, name, normalizer(a)])					
	print len(without_pdf_name), len(with_pdf_name)
	if with_name:
		with open(with_name, "w") as csvoutput:
			writer = csv.writer(csvoutput, lineterminator="\n")
			for line in with_pdf_name:
				writer.writerow(line)
	
	if without_name:
		with open(without_name, "w") as csvoutput:
			writer = csv.writer(csvoutput, lineterminator="\n")
			for line in without_pdf_name:
				writer.writerow(line)


def main():
    # Transform the test set
	raw_gold_csv_directory = os.environ['SNORKELHOME']+ '/tutorials/tables/data/hardware/gold_raw/digikey_raw/'
	formatted_gold_with_name = os.environ['SNORKELHOME']+'/tutorials/tables/data/hardware/train_digikey/hardware_digikey_gold.csv'
	formatted_gold_without_name = None
	format_digikey_gold(raw_gold_csv_directory, formatted_gold_with_name, formatted_gold_without_name)

if __name__=='__main__':
	main()