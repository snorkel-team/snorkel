def pubtator_doc_generator(fp):
    with open(fp, 'rb') as f:
        lines = []
        for line in f:
            if len(line.rstrip()) == 0:
                if len(lines) > 0:
                    yield ''.join(lines)
                    lines = []
            else:
                lines.append(line)