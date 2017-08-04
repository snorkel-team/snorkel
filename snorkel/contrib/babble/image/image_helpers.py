def extract_edge(bbox, side):
    return getattr(bbox, side)

def is_below(b1, b2):
    return b1 > b2

helpers = {
    'extract_edge': extract_edge,
    'is_below': is_below,
}