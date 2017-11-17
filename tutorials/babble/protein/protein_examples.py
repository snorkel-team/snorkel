import csv
import os
import re

from snorkel.contrib.babble import Explanation, link_explanation_candidates

DATA_ROOT = os.environ['SNORKELHOME'] + '/tutorials/babble/protein/data/'

def get_user_lists():
    return {}


def extract_explanations(fpath):
    explanations = []
    with open(DATA_ROOT + fpath, 'rU') as csvfile:
        csvreader = csv.reader(csvfile)
        csvreader.next()
        for i, row in enumerate(csvreader):
            row = [unicode(cell, 'utf-8') for cell in row]
            try:
                doc_id, entities, span1, span2, direction, description = row[:6]
                if doc_id == 'Pubmed ID':
                    continue

                entity1, entity2 = direction.split('-->')
                
                span1_start, span1_end = span1.split(',')
                span2_start, span2_end = span2.split(',')
                span1_stable_id = "{}::span:{}:{}".format(doc_id, span1_start.strip(), span1_end.strip())
                span2_stable_id = "{}::span:{}:{}".format(doc_id, span2_start.strip(), span2_end.strip())
                protein_stable_id = span1_stable_id if entity1 == 'protein' else span2_stable_id
                kinase_stable_id = span1_stable_id if entity1 == 'kinase' else span2_stable_id
                candidate_stable_id = '~~'.join([protein_stable_id, kinase_stable_id])                

                label_str = re.match(r'(true|false)\,?\s*', description,
                    flags=re.UNICODE | re.IGNORECASE)
                label = label_str.group(1) in ['True', 'true']

                condition = description[len(label_str.group(0)):]
                # Only one of these will fire
                condition = re.sub(r"\"entities:[^\"]+\"", 'them', condition, 
                    flags=re.UNICODE | re.IGNORECASE)
                condition = re.sub(r"\"entity:\s(protein|kinase)_[^\"]+\"", 'the \g<1>', condition, 
                    flags=re.UNICODE | re.IGNORECASE)

                explanation = Explanation(condition, label, candidate_stable_id)
                explanations.append(explanation)
            except ValueError:
                if all(cell == u'' for cell in row):
                    break
                print("Skipping malformed or header row {}...".format(i + 2))
                continue
    return explanations


def get_explanations(fpath='razor_explanations.csv'):
    return extract_explanations(fpath)
    # return link_explanation_candidates(explanations, candidates)
            