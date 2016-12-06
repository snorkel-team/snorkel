def temperature_normalizer(temperature):
	try:
		(temp, unit) = temperature.rsplit(' ', 1)
		return int(temp)
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
	if(current[0] == " "):
		current = current[1:]
	return str(abs(round(float(current.split(" ")[0]),1)))

def voltage_normalizer(voltage):
	voltage = voltage.replace("K","000")
	voltage = voltage.replace("k","000")
	return voltage.split(" ")[0].replace("-","")

def gain_normalizer(gain):
	gain= gain.split('@')[0]
	gain = gain.strip()
	gain = gain.replace(",","")
	gain = gain.replace("K","000")
	gain = gain.replace("k","000")
	return str(abs(round(float(gain.split(" ")[0]),1)))

def old_dev_gain_normalizer(gain):
    return str(abs(round(float(gain),1)))
