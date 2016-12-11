import csv
import os
import itertools
from hardware_normalizers import *

#with_pdf_name = []
#without_pdf_name = []
polarity_values = ["NPN", "PNP"]

with open("hardware_digikey_gold.csv", "w") as csvoutput:
    writer = csv.writer(csvoutput, lineterminator="\n")
    for root, subFolders, files in os.walk('single-bjt'):
        for file in files:
            print os.path.join(root, file)
            with open(os.path.join(root, file), "r") as csvinput:
                reader = csv.reader(csvinput)
                next(reader, None)
                for line in reader:
                    add_index = 0
                    if(len(line) == 27):
                        add_index = 1
                    if(len(line) == 28):	#Handling Data With Populated Resistor Entries
                        add_index = 2
                    if len(line) >= 26:		#Most of the entries have length 26, some have 28; entries with less than that are discarded
                        pdf_name = line[0]
                        '''
                        if("=" in pdf_name):
                            pdf_name = pdf_name.split("=")[-1]
                        elif("/" in pdf_name):
                            pdf_name = pdf_name.split("/")[-1]
                        '''
                        part_num = line[3]
                        polarity = line[14]
                        new_polarity = ""	#Restricting Values to NPN and PNP only to avoid errors in polarity normalizer function
                        for val in polarity_values:
                            if(val in polarity):
                                new_polarity = new_polarity+val+";"
                        polarity = new_polarity
                        c_current_max = line[15]
                        ce_v_max = line[16]
                        dc_gain_min = line[19 + add_index]
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
                            if("Manufacturer" in part_num):
                                continue
                            if ('N/A' not in attr) and (attr.strip() not in ["-","*"]):
                                print part_num, attr
                                for a in attr.split(';'):
                                    if len(a.strip())>0:
                                        if(name in ["c_current_max", "dev_dissipation", "dc_gain_min", "ce_v_max"]):
                                            a = "".join(itertools.takewhile(str.isdigit, a))	#Remove Units
                                        writer.writerow([pdf_name, part_num, name, normalizer(a)])
