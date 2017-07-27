def link_explanation_candidates(explanations, candidates):
    """Doc string goes here."""

    candidate_hash_set = set()
    linked = 0
    print("Building list of target candidate hashes...")
    for e in explanations:
        if isinstance(e.candidate, int):
            candidate_hash_set.add(e.candidate)
        elif e.candidate:
            linked += 1
    if linked == len(explanations):
        print("All {} explanations are already linked to candidates.".format(
            len(explanations)))
        return explanations
    else:
        print("Collected {} target candidate hashes from {} explanations.".format(
            len(candidate_hash_set), len(explanations)))
    if not candidate_hash_set:
        print("No candidate hashes were provided. Skipping linking.")
        return explanations

    candidate_hash_map = {}
    print("Gathering desired candidates...")
    for candidate in candidates:
        if hash(candidate) in candidate_hash_set:
            candidate_hash_map[hash(candidate)] = candidate
    print("Found {}/{} desired candidates".format(
        len(candidate_hash_set), len(candidate_hash_map)))
    
    print("Linking explanations to candidates...")
    for e in explanations:
        if isinstance(e.candidate, int):
            try:
                e.candidate = candidate_hash_map[e.candidate]
            except KeyError:
                raise Exception("Expected candidate with hash {} could not be found.".format(
                    e.candidate))
            linked += 1

    print("Linked {}/{} explanations".format(linked, len(explanations)))

    return explanations