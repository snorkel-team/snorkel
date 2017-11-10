import csv

from snorkel.models import Candidate

from explanation import Explanation


class ExplanationIO(object):

    def write(self, explanations, fpath):
        with open(fpath, 'w') as tsvfile:
            tsvwriter = csv.writer(tsvfile, delimiter='\t')
            for exp in explanations:
                tsvwriter.writerow([exp.candidate.get_stable_id(), 
                                    exp.label, 
                                    exp.condition.encode('utf-8'), 
                                    exp.semantics])
        fpath = fpath if len(fpath) < 50 else fpath[:20] + '...' + fpath[-30:]
        print("Wrote {} explanations to {}".format(len(explanations), fpath))

    def read(self, fpath):
        with open(fpath, 'r') as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter='\t')
            num_read = 0
            explanations = []
            for (candidate, label, condition, semantics) in tsvreader:
                explanations.append(
                    Explanation(
                        condition=condition.strip(),
                        label=True if label=='True' else False,
                        candidate=None if candidate == 'None' else candidate,
                        semantics=semantics))
                num_read += 1
        fpath = fpath if len(fpath) < 50 else fpath[:20] + '...' + fpath[-30:]
        print("Read {} explanations from {}".format(num_read, fpath))
        return explanations


def link_explanation_candidates(explanations, candidates):
    """Doc string goes here."""

    target_candidate_ids = set()
    linked = 0
    print("Building list of target candidate ids...")
    for e in explanations:
        if not isinstance(e.candidate, Candidate):
            target_candidate_ids.add(e.candidate)
        elif e.candidate:
            linked += 1
    if linked == len(explanations):
        print("All {} explanations are already linked to candidates.".format(
            len(explanations)))
        return explanations
    else:
        print("Collected {} unique target candidate ids from {} explanations.".format(
            len(target_candidate_ids), len(explanations)))
    if not target_candidate_ids:
        print("No candidate hashes were provided. Skipping linking.")
        return explanations

    candidate_map = {}
    print("Gathering desired candidates...")
    for candidate in candidates:
        if candidate.get_stable_id() in target_candidate_ids:
            candidate_map[candidate.get_stable_id()] = candidate
    if len(candidate_map) < len(target_candidate_ids):
        num_missing = len(target_candidate_ids) - len(candidate_map)
        print("Could not find {} target candidates with the following stable_ids (first 5):".format(
            num_missing))
        num_reported = 0
        for i, c_hash in enumerate(target_candidate_ids):
            if c_hash not in candidate_map:
                print(c_hash)
                num_reported += 1
                if num_reported >= 5:
                    break
        # raise Exception("Could not find {} target candidates.".format(num_missing))

    print("Found {}/{} desired candidates".format(
        len(candidate_map), len(target_candidate_ids)))

    print("Linking explanations to candidates...")
    for e in explanations:
        if not isinstance(e.candidate, Candidate):
            try:
                e.candidate = candidate_map[e.candidate]
                linked += 1
            except KeyError:
                pass
                # raise Exception("Expected candidate with hash {} could not be found.".format(
                #     e.candidate))

    print("Linked {}/{} explanations".format(linked, len(explanations)))

    return explanations
