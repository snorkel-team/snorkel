from hardware_normalizers import *
import csv

def format_gold(raw_gold_file, formatted_gold_file)
	with open(formatted_gold_file, "w") as csvoutput:
		writer = csv.writer(csvoutput, lineterminator="\n")
		with open(raw_gold_file, "r") as csvinput:
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
					writer.writerow([doc_name, part_num, "polarity", polarity_normalizer(polarity)])
				if "N/A" not in v_ceo:
					writer.writerow([doc_name, part_num, "ce_v_max", voltage_normalizer(v_ceo)])
				if "N/A" not in v_cbo:
					writer.writerow([doc_name, part_num, "cb_v_max", voltage_normalizer(v_cbo)])
				if "N/A" not in v_ebo:
					writer.writerow([doc_name, part_num, "eb_v_max", voltage_normalizer(v_ebo)])
				if "N/A" not in current:
					writer.writerow([doc_name, part_num, "c_current_max", current_normalizer(current)])
				if "N/A" not in dev_dissipation:
					for dissipation in dev_dissipation.split(";"):
						writer.writerow([doc_name, part_num, "dev_dissipation", dissipation_normalizer(dissipation)])
				if "N/A" not in min_temp:
					writer.writerow([doc_name, part_num, "stg_temp_min", temperature_normalizer(min_temp)])
				if "N/A" not in max_temp:
					writer.writerow([doc_name, part_num, "stg_temp_max", temperature_normalizer(max_temp)])
				if "N/A" not in dev_gain:
					for gain in dev_gain.split(";"):
						if("@" in gain):
							writer.writerow([doc_name, part_num, "dc_gain_min", gain_normalizer(gain)])


