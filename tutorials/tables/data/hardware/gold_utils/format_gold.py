from hardware_normalizers import *
import csv

def format_gold(raw_gold_file, formatted_gold_file):
	delim = ';'
	with open(raw_gold_file, "r") as csvinput, open(formatted_gold_file, "w") as csvoutput:
		writer = csv.writer(csvoutput, lineterminator="\n")
		reader = csv.reader(csvinput)
		next(reader, None) # Skip header row
		for line in reader:
			(pdf_name, part_family, part_num, manufacturer, 
			polarity, ce_v_max, cb_v_max, eb_v_max, c_current_max, 
			dev_dissipation, stg_temp_min, stg_temp_max, dc_gain_min, 
			notes, annotator) = line
			(doc_name, extension) = pdf_name.rsplit('.', 1)
			part_num = part_num.replace(' ','').upper()
			name_attr_norm = [
				('polarity', polarity, polarity_normalizer),
				('ce_v_max', ce_v_max, voltage_normalizer),
				('cb_v_max', cb_v_max, voltage_normalizer),
				('eb_v_max', eb_v_max, voltage_normalizer),
				('c_current_max', c_current_max, current_normalizer),
				('dev_dissipation', dev_dissipation, dissipation_normalizer),
				('stg_temp_min', stg_temp_min, temperature_normalizer),
				('stg_temp_max', stg_temp_max, temperature_normalizer),
				('dc_gain_min', dc_gain_min, gain_normalizer)]
			for name, attr, normalizer in name_attr_norm:
				if 'N/A' not in attr:
					print part_num, attr
					for a in attr.split(';'):
						if len(a.strip())>0:
							writer.writerow([doc_name, part_num, name, normalizer(attr)])

			# if polarity != 'N/A':
			# 	for p in polarity.split(delim):
			# 		writer.writerow([doc_name, part_num, "polarity", polarity_normalizer(polarity)])
			# if ce_v_max != 'N/A':
			# 	for v in ce_v_max.split(delim):
			# 		writer.writerow([doc_name, part_num, "ce_v_max", voltage_normalizer(v_ceo)])
			# if cb_v_max != 'N/A':
			# 	for v in cb_v_max.split(delim):
			# 		writer.writerow([doc_name, part_num, "cb_v_max", voltage_normalizer(v_cbo)])
			# if eb_v_max != 'N/A':
			# 	for v in eb_v_max.split(delim):
			# 	writer.writerow([doc_name, part_num, "eb_v_max", voltage_normalizer(v_ebo)])
			# if "N/A" not in current:
			# 	writer.writerow([doc_name, part_num, "c_current_max", current_normalizer(current)])
			# if "N/A" not in dev_dissipation:
			# 	for dissipation in dev_dissipation.split(";"):
			# 		writer.writerow([doc_name, part_num, "dev_dissipation", dissipation_normalizer(dissipation)])
			# if "N/A" not in min_temp:
			# 	writer.writerow([doc_name, part_num, "stg_temp_min", temperature_normalizer(min_temp)])
			# if "N/A" not in max_temp:
			# 	writer.writerow([doc_name, part_num, "stg_temp_max", temperature_normalizer(max_temp)])
			# if "N/A" not in dev_gain:
			# 	for gain in dev_gain.split(";"):
			# 		if("@" in gain):
			# 			writer.writerow([doc_name, part_num, "dc_gain_min", gain_normalizer(gain)])
'''
def test_normalizer(raw_gold_file):
	with open(raw_gold_file, "r") as csvinput:
		reader = csv.reader(csvinput)
		next(reader, None) # Skip header row
		for line in reader:
			(pdf_name, part_family, part_num, manufacturer, 
			 polarity, ce_v_max, cb_v_max, eb_v_max, c_current_max, 
			 dev_dissipation, stg_temp_min, stg_temp_max, dc_gain_min, 
			 notes, annotator) = line

			if stg_temp_min != 'N/A':
				print polarity_normalizer(polarity)
				# print temperature_normalizer(stg_temp_min)
				# print temperature_normalizer(stg_temp_max)
'''

def main():
	raw_gold = os.environ['SNORKELHOME']+ '/tutorials/tables/data/hardware/gold_raw/test_gold_raw.csv'
	formatted_gold = os.environ['SNORKELHOME']+'/tutorials/tables/data/hardware/gold_raw/test_gold_formatted.csv'
	format_gold(raw_gold, formatted_gold)

if __name__=='__main__':
	main()