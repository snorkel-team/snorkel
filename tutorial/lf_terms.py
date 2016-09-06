vague = ['abnormality', 'absence', 'activity', 'adult', 'adult-onset', 'age of onset', 'aggressive', 'all age',
         'ashkenazi jewish', 'asymptomatic', 'atm', 'atypical', 'best', 'black', 'blood', 'blue', 'brazilian',
         'carrier', 'case', 'cause by a', 'center', 'child', 'close', 'combination', 'common', 'completely',
         'condition', 'contraction', 'correlation', 'critical', 'damage', 'death', 'defect', 'deficiency',
         'deficitanimal model', 'detect', 'diagnose', 'diagnosis', 'disease model', 'disorder', 'distortion',
         'does', 'donor', 'early onset', 'early-onset', 'employ', 'ethnic', 'false negative', 'false positive',
         'family', 'family history', 'family study', 'fatal', 'founder', 'germline', 'grade', 'greater than',
         'heterozygous', 'high risk', 'high-grade', 'history', 'homozygous', 'human model', 'identify',
         'immunosuppression', 'incidence', 'independent', 'indicate', 'individual', 'inheritance', 'instability',
         'intensity', 'intermediate', 'late-onset', 'lesion', 'less than', 'lethal', 'location', 'long arm',
         'low risk', 'mass', 'max', 'mild', 'missense', 'mobility', 'model', 'moderate', 'moderate risk',
         'mouse model', 'negative', 'newborn', 'novel', 'one family', 'onset', 'outcome', 'parameter',
         'pathogenesis', 'patient die', 'physical', 'platelet', 'positive', 'prevalence', 'progression',
         'proliferation', 'protection', 'radiation', 'rank', 'reading', 'red', 'reduce', 'reflex', 'remission',
         'response', 'responsible', 'result', 'same family', 'sensitivity', 'severity', 'sex', 'single', 'sit',
         'skip', 'southern blot', 'staining', 'strain', 'study', 'suffering', 'suppression', 'syndrome', 'task',
         'test', 'therapy', 'transmission', 'transplant', 'unrelated', 'unstable', 'usually fatal', 'very close',
         'very high', 'very large', 'very rare', 'vibration', 'white', 'withdrawal', 'zeta']

vague += ['most case','examine for','increase activity','european origin','clinical sign']
vague += ['colorectal']
vague += ["alpha", "beta", "gamma"]
vague += ["type i", "type ii", "type iii", "class i", "class ii", "class iii", "stage i", "stage ii", "stage iii"]
vague += ['rflp', 'ashkenazi jewish', 'dmt1', 'c2', 'sscp', 'pcr', 'mefv', 'phenylalanine hydroxylase', 'fas',
          'g6pd', 'autosomal dominant', 'wt1', 'vlcad', 'missense', 'plp', 'wasp', 'does not', 'fmrp', 'ph',
          'gaa', 'dtdst', 'galt', 'hgo', 'pah', 'nervous system', 'normal', 'an2', 'arsa', 'aga', 'mentally alert',
          'absent', 'galc', 'lpp', 'cga', 'esr', 'cm', 'rr', 'dn', 'gt']

vague += ["type i", "type ii", "type iii", "class i", "class ii", "class iii", "stage i", "stage ii", "stage iii"]
vague += ['rflp', 'ashkenazi jewish', 'dmt1', 'c2', 'sscp', 'pcr', 'mefv', 'phenylalanine hydroxylase', 'fas',
          'g6pd', 'autosomal dominant', 'wt1', 'vlcad', 'missense', 'plp', 'wasp', 'does not', 'fmrp', 'ph',
          'gaa', 'dtdst', 'galt', 'hgo', 'pah', 'nervous system', 'normal', 'an2', 'arsa', 'aga', 'mentally alert',
          'absent', 'galc', 'lpp', 'cga', 'esr', 'cm', 'rr', 'dn', 'gt']
vague = set(vague)

organs = set(["brain","heart","lung","muscle","tissue","stomach",'skin','spinal cord'])

intensity_terms = set(["acute","progressive","recurrent","mild","severe",
                       "extreme","low","normal","premature","classic"])

left_terms = set(['maternal', 'adolescent', 'neonatal', 'adult', 'childhood',
                  'familial',"hereditary","autosomal", "autoimmune","inherited"])

right_terms = set(['deficienty', 'syndrome', 'disorders', 'deficient', 'defects', 'tumors',
                   'disease', 'tumor', 'deficiency', 'deficiencies', 'dysfunction', 'infection'])

indicators = set(['cancer', 'cancers', 'tumor', 'tumors', 'tumour', 'tumours', 'dm'])

bodypart = ["abdomen", "adam's apple", "adenoids", "adrenal gland", "anatomy", "ankle", "anus",
            "appendix", "arch", "arm", "artery", "back", "ball of the foot", "belly", "belly button",
            "big toe", "bladder", "blood", "blood vessels", "body", "bone", "brain", "breast", "buttocks",
            "calf", "capillary", "carpal", "cartilage", "cell", "cervical vertebrae", "cheek", "chest",
            "chin", "circulatory system", "clavicle", "coccyx", "collar bone", "diaphragm", "digestive system",
            "ear", "ear lobe", "elbow", "endocrine system", "esophagus", "eye", "eyebrow", "eyelashes",
            "eyelid", "face", "fallopian tubes", "feet", "femur", "fibula", "filling", "finger",
            "fingernail", "follicle", "foot", "forehead", "gallbladder", "glands", "groin", "gums",
            "hair", "hand", "head", "heart", "heel", "hip", "humerus", "immune system", "instep",
            "index finger", "intestines", "iris", "jaw", "kidney", "knee", "larynx", "leg", "ligament",
            "lip", "liver", "lobe", "lumbar vertebrae", "lungs", "lymph node", "mandible", "metacarpal",
            "metatarsal", "molar", "mouth", "muscle", "nail", "navel", "neck", "nerves", "nipple", "nose",
            "nostril", "organs", "ovary", "palm", "pancreas", "sperm" ,"patella", "pelvis", "phalanges", "pharynx",
            "pinky", "pituitary", "pore", "pupil", "radius", "rectum", "red blood cells", "respiratory system",
            "ribs", "sacrum", "scalp", "scapula", "senses", "shin", "shoulder", "shoulder blade", "skeleton",
            "skin", "skull", "sole", "spinal column", "spinal cord", "spine", "spleen", "sternum", "stomach",
            "tarsal", "teeth", "tendon", "testes", "thigh", "thorax", "throat", "thumb", "thyroid", "tibia",
            "tissue", "toe", "toenail", "tongue", "tonsils", "tooth", "torso", "trachea", "ulna", "ureter",
            "urethra", "urinary system", "uterus", "uvula", "vein", "vertebra", "waist", "white blood cells", "wrist"]

symptom = (["symptoms","symptom", "disequilibrium"])

bodysym = bodypart + symptom

common_disease = ["abuse", "acute liver failure", "acute renal failure", "adenomas", "akathisia", "amnesia", "anemia",
                  "anxiety", "apnea", "arrhythmia", "asystole", "bleeding", "blindness", "bradycardia", "cardiomyopathy",
                  "cardiotoxicity", "catalepsy", "cataract", "cataracts", "cleft lip", "cleft palate",
                  "cognitive impairment", "confusion", "convulsions", "convulsive", "delirium", "depression", "diabetes",
                  "diabetes insipidus", "diabetes mellitus", "diabetic", "dyskinesia", "dyskinesias", "dystonia",
                  "encephalopathy", "epilepsy", "epileptic", "fever", "headache", "heart failure", "hemolysis",
                  "hemolytic anemia", "hemorrhage", "hepatitis", "hepatotoxicity", "hiccups", "hydrocephalus",
                  "hypertension", "hypertensive", "hyperthermia", "hypokalemia", "hypotension", "hypotensive", "hypotonia",
                  "immunodeficiency", "infections", "infertility", "leukemia", "liver damage", "malaria", "mania",
                  "melanoma", "mental retardation", "migraine", "myalgia", "myocardial damage", "myocardial infarction",
                  "myoclonus", "myopathy", "myotonia", "nausea", "nephropathy", "nephrotoxic", "nephrotoxicity",
                  "neurotoxicity", "obese", "obesity", "ototoxicity", "overdose", "pain", "paralysis", "parkinsonian",
                  "pneumonia", "polyuria", "proteinuria", "psoriasis", "psychiatric", "psychosis", "qt prolongation",
                  "renal failure", "rhabdomyolysis", "rigidity", "rubella", "seizure", "seizures", "status epilepticus",
                  "stroke", "sudden cardiac death", "sudden death", "systemic lupus erythematosus", "tachycardia",
                  "thrombocytopenia", "torsades de pointes", "toxicities",  "toxicity", "tremor", "tumor",
                  "ventricular tachycardia", "visual loss", "vomiting", "exencephalic", "anuria", "anaemia",
                  "platelet aggregation", "chorea", "aggressiveness", "leukocyturia", "neurotoxic", "atrial fibrillation",
                  "ventricular fibrillation", "HIV-infected", "poisoning", "bruising", "schizophrenic", "fasciculation",
                  "hemorrhagic", "dyskinetic", "weight gain", "infection", "burn", "bleeding", "arrhythmias", "dizziness",
                  "malignancies", "malignancy"]

non_common_disease = ["attack", "block", "light", "strains", "prophylactic", "serum creatinine", "three patients",
                      "blocked", "microscopy", "pan", "pregnancy", "recurrence", "reflex", "tolerance", "ecg",
                      "pathogenesis", "adverse effect", "oxygen", "physical", "transplant", "lesion", "map",
                      "pressure", "unchanged", "died", "fatal", "adverse events", "diagnosed", "indicated",
                      "controlled", "nervous", "side effect", "anaesthesia", "resolved", "absence", "production",
                      "discontinued", "injury", "anesthesia", "hbv", "impairment", "management", "protection",
                      "completely", "discontinuation", "parameters", "prospective", "sensitivity", "abnormalities",
                      "medical", "adverse effects", "bone", "secondary", "withdrawal", "bladder",
                      "elderly", "moderate", "stress", "adult", "syndrome", "toxic", "complication", "complications",
                      "sedation", "surgery", "ci", "lesions", "memory", "exposure", "blood pressure", "chemotherapy",
                      "damage", "examined", "long-term", "heart", "combination", "symptoms", "cardiac", "injection",
                      "response", "reduced", "severe", "therapy", "green", "creatinine clearance", "resonance",
                      "behaviors", "emergency", "evaluable", "angiogenesis", "fall", "body weight", "oxidative stress",
                      "regression"]

common_disease_acronyms = ['AAPC', 'ACS', 'AD', 'ADD', 'ADD-RT', 'ADEM', 'ADHD', 'AF', 'AGC', 'AGS', 'AHC', 'AIDS',
                           'AIP', 'ALA DD', 'ALI', 'ALS', 'AMD', 'AOS', 'APA', 'APS', 'ARBD', 'ARD', 'ARDS', 'ARND',
                           'AS', 'ASD', 'ASDs', 'AVMs', 'B-NHL', 'BBS', 'BD', 'BEB', 'BEH', 'BFIC', 'BH', 'BPD', 'BPH',
                           'BSE', 'BSS', 'BV', 'C1D', 'C3D', 'C4D', 'C5D', 'C6D', 'C7D', 'CACH', 'CAD', 'CADSIL', 'CAPD',
                           'CCALD', 'CCD', 'CCHF', 'CCHS', 'CCM', 'CDG', 'CDGS', 'CEP', 'CES', 'CESD', 'CF', 'CFIDS', 'CFS',
                           'CGBD', 'CHD', 'CHF', 'CHSCIDP', 'CIN', 'CIPA', 'CJD', 'CL/P', 'CLD', 'COFS', 'COPD', 'CP/CPPS',
                           'CPM', 'CPPS', 'CRF', 'CRKP', 'CRPS', 'CSD', 'CVD', 'DAS', 'DBA', 'DBMD', 'DD', 'DEF', 'DF',
                           'DH', 'DHF', 'DLB', 'DM', 'DMD', 'DP', 'DRSP disease', 'DSPS', 'DTs', 'DVD', 'DVT', 'DiG', 'EDS',
                           'EEE', 'EHK', 'EMH', 'EMR', 'ENS', 'EPP', 'ESRD', 'ESS', 'EVAFAE', 'FASDs', 'FFI', 'FMA', 'FMD',
                           'FMF', 'FMS', 'FNDI', 'FSP', 'FTD', 'FVS', 'FXS', 'GAN', 'GAS disease', 'GBS', 'GBS disease',
                           'GCE', 'GERD', 'GIB', 'GN', 'GRMD', 'GSS disease', 'GT/LD', 'GVHD', 'GWD', 'HAS', 'HBL', 'HCP',
                           'HD', 'HDL2', 'HFA', 'HFMD', 'HFRS', 'HI', 'HIT', 'HL', 'HMS', 'HMSN Type III', 'HOH', 'HPS',
                           'HSP', 'HTN', 'IBD', 'IBM', 'IBS', 'IC/PBS', 'IDMS', 'IED', 'IHA', 'INAD', 'IRD', 'ITP', 'JAS', 'JE',
                           'JHD', 'JT', 'KS', 'KSS', 'LCM', 'LEMS', 'LFA', 'LGV', 'LID', 'LIDs', 'LKS', 'LNS', 'MBD', 'MCS',
                           'MD', 'MDD', 'MDP', 'MDR TB', 'MEF', 'MHP', 'MID', 'MJD', 'ML', 'MLD', 'MMA', 'MMR', 'MMRV', 'MND',
                           'MOH', 'MPD', 'MPS I', 'MPS II', 'MPS III', 'MPS IV', 'MPS VI', 'MPS VII', 'MR/DD', 'MSA', 'MSDD',
                           'NAS', 'NBIA', 'NCL', 'NF1', 'NF2', 'NKH', 'NLD', 'NMDs', 'NMO', 'NMS', 'NPC1', 'NPH', 'NSCLC', 'NTD',
                           'NTDs', 'OA', 'OAB', 'OCD', 'ODD', 'OIH', 'OMA', 'ON', 'OPC', 'OPCA', 'OSA', 'PBC', 'PBD', 'PCOS',
                           'PCT', 'PD', 'PDD', 'PDD-NOS', 'PDD/NOS', 'PKAN', 'PLMD', 'PLS', 'PMD', 'PML', 'PMS',
                           'POTS', 'PPMA', 'PPS', 'PSC', 'PSP', 'PVL', 'PW', 'Q fever', 'RA', 'RAD', 'RIHA', 'RIND', 'RLF',
                           'RLS', 'RMDs', 'ROP', 'RS', 'RSD', 'RTI', 'SARS', 'SB', 'SBS', 'SCA1', 'SCA2', 'SIADH', 'SIAT',
                           'SIDS', 'SIS', 'SJS type 2', 'SLE', 'SMEI', 'SMS', 'SPS', 'SSPE', 'STD', 'STEMI', 'SUNCT', 'SWS',
                           'TB', 'TBI', 'TCD', 'TCS', 'TDP', 'TEF', 'TIA', 'TMH', 'TMJ/TMD', 'TMR', 'TN', 'TOS', 'TS', 'TSEs',
                           'TSP', 'TTH', 'TTP', 'UC', 'UCPPS', 'UDA', 'URI', 'UTIs', 'VCFS', 'VD', 'VHF', 'VSD', 'VVC', 'WD',
                           'WEE', 'WFS', 'WS', 'XDR TB', 'XLDCM', 'XLSA', 'XLT', 'XP', 'YSS', 'YVS', 'ZBLS', 'vCJD']

non_disease_acronyms = ["CPDD", "GALT", "PAH", "SSCP", "MEFV", "DTDST", "WASP", "VLCAD", "HGO", "MCC", "CT", "TGA", "AVP", "TGA", "ARSA",
                        "IDDM4", "AGA", "TfR", "CI", "PTT", "IC", "ACTH", "IVA", "FH", "GALC", "TAM", "SD", "BN", "CC", "DX", "ECG",
                        "MAP", "NTG", "ADR", "FS", "AAP", "CSA", "HIV", "BMD", "MIC", "PR", "GM", "IP", "DS", "ED", "CP", "GAD"]

positive_indicator = ["nephrotoxic", "delirium", "hallucinations", "epilepticus", "epilepsy", "amnesia", "rheumatoid",
                      "epileptic", "hepatotoxicity", "ischemia", "cardiotoxicity", "cardiomyopathy", "esrd", "nephrotoxicity", "parkinson",
                      "'s", "injury", "loss", "pain", "depression", "failure", "disease", "diseases", "neutropenia", "toxicity",
                      "schizophrenia", "renal", "pd", "hemorrhage", "myocardial", "abuse", "rubella"]

negative_indicator = ["indicated", "patients", "controlled", "died", "fatal", "nervous", "oxygen", "events", "day",
                      "anaesthesia", "resolved", "side", "production", "discontinued", "management", "protection",
                      "completely", "discontinuation", "parameters", "prospective", "sensitivity", "anesthesia",
                      "grade", "effects", "medical", "elderly", "moderate", "adult", "serum", "complication",
                      "sedation", "effect", "surgery", "creatinine", "exposure", "chemotherapy", "examined",
                      "long-term", "combination", "reduced", "injection", "response", "severe", "case", "therapy", "pressure",
                      "pressures", "use", "used"]

adj_diseases = ['acromegalic', 'akinetic', 'allergic', 'arrhythmic', 'arteriopathic', 'asthmatic', 
                'atherosclerotic', 'bradycardic', 'cardiotoxic', 'cataleptic', 'cholestatic', 
                'cirrhotic', 'diabetic', 'dyskinetic', 'dystonic', 'eosinophilic', 'epileptic', 
                'exencephalic', 'haemorrhagic', 'hemolytic', 'hemorrhagic', 'hemosiderotic', 'hepatotoxic'
                'hyperalgesic', 'hyperammonemic', 'hypercalcemic', 'hypercapnic', 'hyperemic', 
                'hyperkinetic', 'hypertrophic', 'hypomanic', 'hypothermic', 'ischaemic', 'ischemic', 
                'leukemic', 'myelodysplastic', 'myopathic', 'necrotic', 'nephrotic', 'nephrotoxic', 
                'neuropathic', 'neurotoxic', 'neutropenic', 'ototoxic', 'polyuric', 'proteinuric', 
                'psoriatic', 'psychiatric', 'psychotic', 'quadriplegic', 'schizophrenic', 'teratogenic', 
                'thromboembolic', 'thrombotic', 'traumatic', 'vasculitic']
