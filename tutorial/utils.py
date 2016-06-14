from lxml import etree as et
from collections import defaultdict, namedtuple

PubtatorMention = namedtuple('PubtatorMention', 'mesh_id, text, char_offset, char_length')

PubtatorRelation = namedtuple('PubtatorRelation', 'types, mesh_ids')

def collect_pubtator_annotations(doc):
    """
    Given a list of ddlite Documents with PubTator/CDR annotations,
    extract a dictionary of annotations by type.
    """
    annotations = defaultdict(list)
    for a in doc.attribs['root'].xpath('.//annotation'):

        # Relation annotations
        if len(a.xpath('./infon[@key="relation"]')) > 0:
            type = a.xpath('./infon[@key="relation"]/text()')[0]
            types = a.xpath('./infon[@key != "relation"]/@key')
            mesh_ids = a.xpath('./infon[@key != "relation"]/text()')
            annotations[type].append(PubtatorRelation(types=types, mesh_ids=mesh_ids))

        # Mention annotations
        else:
            txt = a.xpath('./text/text()')[0]
            offset = int(a.xpath('./location/@offset')[0])
            length = int(a.xpath('./location/@length')[0])
            type = a.xpath('./infon[@key="type"]/text()')[0]
            mesh = a.xpath('./infon[@key="MESH"]/text()')[0]
            annotations[type].append(PubtatorMention(mesh_id=mesh, text=txt,
                                                     char_offset=offset, char_length=length))
    return annotations
