def link_example_candidates(examples, candidates):
    """Doc string goes here."""

    candidate_hash_set = set()
    linked = 0
    print("Building list of target candidate hashes...")
    for e in examples:
        if isinstance(e.candidate, int):
            candidate_hash_set.add(e.candidate)
        elif e.candidate:
            linked += 1
    if linked == len(examples):
        print("All {} examples are already linked to candidates.".format(
            len(examples)))
        return examples
    else:
        print("Collected {} target candidate hashes from {} examples.".format(
            len(candidate_hash_set), len(examples)))
    
    candidate_hash_map = {}
    print("Gathering desired candidates...")
    for candidate in candidates:
        if hash(candidate) in candidate_hash_set:
            candidate_hash_map[hash(candidate)] = candidate
    print("Found {}/{} desired candidates".format(
        len(candidate_hash_set), len(candidate_hash_map)))
    
    print("Linking examples to candidates...")
    for e in examples:
        if isinstance(e.candidate, int):
            try:
                e.candidate = candidate_hash_map[e.candidate]
            except KeyError:
                raise Exception("Expected candidate with hash {} could not be found.".format(
                    e.candidate))
            linked += 1

    print("Linked {}/{} examples".format(linked, len(examples)))

    return examples