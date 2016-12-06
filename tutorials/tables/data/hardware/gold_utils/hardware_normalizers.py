def temperature_normalizer(temperature):
	try:
		if(temperature.split(" ")[0].lstrip('-').replace('.','',1).isdigit()):
			return temperature.split(" ")[0]
	except:
		print "Incorrect Temperature Value"

def polarity_normalizer(polarity):
	try:
		if(polarity in ["NPN", "PNP"]):
			return polarity
	except:
		print "Incorrect Polarity Value"

def dissipation_normalizer(dissipation):
	if(dissipation[0] == " "):
		dissipation = dissipation[1:]
	return str(abs(round(float(dissipation.split(" ")[0]),1)))

def current_normalizer(current):
	return str(abs(round(float(current.split(" ")[0]),1)))

def voltage_normalizer(voltage):
	voltage = voltage.replace("K","000")
	voltage = voltage.replace("k","000")
	return voltage.split(" ")[0].replace("-","")

def gain_normalizer(gain):
	while(gain[0] == " "):
		gain = gain[1:]
	gain = gain.replace(",","")
	gain = gain.replace("K","000")
	gain = gain.replace("k","000")
	return str(abs(round(float(gain.split(" ")[0]),1)))




