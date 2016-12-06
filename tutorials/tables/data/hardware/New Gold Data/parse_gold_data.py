import csv

def voltage_normalizer(voltage):
	voltage = voltage.replace("K","000")
	voltage = voltage.replace("k","000")
	return voltage.split(" ")[0].replace("-","")

def current_normalizer(current):
	return str(abs(round(float(current.split(" ")[0]),1)))

def gain_normalizer(gain):
	while(gain[0] == " "):
		gain = gain[1:]
	gain = gain.replace(",","")
	gain = gain.replace("K","000")
	gain = gain.replace("k","000")
	return str(abs(round(float(gain.split(" ")[0]),1)))

def dissipation_normalizer(dissipation):
	if(dissipation[0] == " "):
		dissipation = dissipation[1:]
	return str(abs(round(float(dissipation.split(" ")[0]),1)))

def polarity_normalizer(polarity):
	try:
		if(polarity in ["NPN", "PNP"]):
			return polarity
	except:
		print "Incorrect Polarity Value"
def temperature_normalizer(temperature):
	try:
		if(temperature.split(" ")[0].lstrip('-').replace('.','',1).isdigit()):
			return temperature.split(" ")[0]
	except:
		print "Incorrect Temperature Value"

with open("Hardware_Relations.csv", "w") as csvoutput:
	writer = csv.writer(csvoutput, lineterminator="\n")
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


