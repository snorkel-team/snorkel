from hardware_normalizers import *
import csv
import os

def format_old_dev_gold(raw_gold_file, formatted_gold_file):
	with open (raw_gold_file, 'r') as csvinput, open(formatted_gold_file, 'w') as csvoutput:
		writer = csv.writer(csvoutput, lineterminator='\n')
		reader = csv.reader(csvinput)
		next(reader, None) # Skip header row
		for line in reader:
			(pdf_name, part_num, manufacterur, polarity, pin_count,
			ce_v_max, cb_v_max, eb_v_max, c_current_max, dev_dissipation,
			stg_temp_min, stg_temp_max, stg_temp_unit, dc_gain_min) = line
			(doc_name, extension) = pdf_name.rsplit('.', 1) # split on right-most period
			# Add unit to stg_temp_min and max
			stg_temp_min = stg_temp_min + ' ' + stg_temp_unit
			stg_temp_max = stg_temp_max + ' ' + stg_temp_unit
			part_num = part_num.replace(' ', '').upper()
			name_attr_norm = [
				('polarity', polarity, polarity_normalizer),
				('ce_v_max', ce_v_max, voltage_normalizer),
				('cb_v_max', cb_v_max, voltage_normalizer),
				('eb_v_max', eb_v_max, voltage_normalizer),
				('c_current_max', c_current_max, current_normalizer),
				('dev_dissipation', dev_dissipation, dissipation_normalizer),
				('stg_temp_min', stg_temp_min, temperature_normalizer),
				('stg_temp_max', stg_temp_max, temperature_normalizer),
				('dc_gain_min', dc_gain_min, old_dev_gain_normalizer)]
			for name, attr, normalizer in name_attr_norm:
				if attr:
					print part_num, attr
					if name == "dc_gain_min":
						for a in attr.split(' '):
							writer.writerow([doc_name, part_num, name, normalizer(a)])
					else:
						writer.writerow([doc_name, part_num, name, normalizer(attr)])

def format_new_dev_gold(raw_gold_file, formatted_gold_file):
	# NOTE: this is APPENDING to the gold file, and is meant to be called AFTER
	# the format_gold function.
	delim = ';'
	with open(raw_gold_file, "r") as csvinput, open(formatted_gold_file, "a") as csvoutput:
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
							writer.writerow([doc_name, part_num, name, normalizer(a)])

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
					for a in attr.split(';'):
						if len(a.strip())>0:
							writer.writerow([doc_name, part_num, name, normalizer(a)])


def main():
    # Transform the test set
	raw_gold = os.environ['SNORKELHOME']+ '/tutorials/tables/data/hardware/gold_raw/test_gold_raw.csv'
	formatted_gold = os.environ['SNORKELHOME']+'/tutorials/tables/data/hardware/test/hardware_test_gold.csv'
	format_gold(raw_gold, formatted_gold)

    # Transform the dev set
	raw_gold = os.environ['SNORKELHOME']+ '/tutorials/tables/data/hardware/gold_raw/old_dev_gold_raw.csv'
	formatted_gold = os.environ['SNORKELHOME']+'/tutorials/tables/data/hardware/dev/hardware_dev_gold.csv'
	format_old_dev_gold(raw_gold, formatted_gold)
    # NOTE: This MUST come after format_old_dev_gold so it can append to the previous file.
	raw_gold = os.environ['SNORKELHOME']+ '/tutorials/tables/data/hardware/gold_raw/new_dev_gold_raw.csv'
	formatted_gold = os.environ['SNORKELHOME']+'/tutorials/tables/data/hardware/dev/hardware_dev_gold.csv'
	format_new_dev_gold(raw_gold, formatted_gold)

	#Total Gold Data
	#raw_gold = os.environ['SNORKELHOME']+ '/tutorials/tables/data/hardware/gold_raw/test_gold_raw.csv'
	#formatted_gold = os.environ['SNORKELHOME']+'/tutorials/tables/data/hardware/gold_parsed/new_total_gold.csv'
	#format_gold(raw_gold, formatted_gold)
	#raw_gold = os.environ['SNORKELHOME']+ '/tutorials/tables/data/hardware/gold_raw/new_dev_gold_raw.csv'
	#formatted_gold = os.environ['SNORKELHOME']+'/tutorials/tables/data/hardware/gold_parsed/new_total_gold.csv'
	#format_new_dev_gold(raw_gold, formatted_gold)

if __name__=='__main__':
	main()
