import pandas as pd
from snorkel.models import StableLabel
from snorkel.db_helpers import reload_annotator_labels

FPATH = 'data/gold_labels.tsv'

def number_of_people(sentence):
    active_sequence = False
    count = 0
    for tag in sentence.ner_tags:
        if tag == 'PERSON' and not active_sequence:
            active_sequence = True
            count += 1
        elif tag != 'PERSON' and active_sequence:
            active_sequence = False
    return count


def load_external_labels(session, candidate_class, annotator_name='gold'):
    gold_labels = pd.read_csv(FPATH, sep="\t")
    for index, row in gold_labels.iterrows():    

        # We check if the label already exists, in case this cell was already executed
        context_stable_ids = "~~".join([row['person1'], row['person2']])
        query = session.query(StableLabel).filter(StableLabel.context_stable_ids == context_stable_ids)
        query = query.filter(StableLabel.annotator_name == annotator_name)
        if query.count() == 0:
            session.add(StableLabel(
                context_stable_ids=context_stable_ids,
                annotator_name=annotator_name,
                value=row['label']))
                    
        # Because it's a symmetric relation, load both directions...
        context_stable_ids = "~~".join([row['person2'], row['person1']])
        query = session.query(StableLabel).filter(StableLabel.context_stable_ids == context_stable_ids)
        query = query.filter(StableLabel.annotator_name == annotator_name)
        if query.count() == 0:
            session.add(StableLabel(
                context_stable_ids=context_stable_ids,
                annotator_name=annotator_name,
                value=row['label']))

    # Commit session
    session.commit()

    # Reload annotator labels
    reload_annotator_labels(session, candidate_class, annotator_name, split=1, filter_label_split=False)
    reload_annotator_labels(session, candidate_class, annotator_name, split=2, filter_label_split=False)



# create distant superivsion subset for workshop
    # from lib.viz import display_candidate

    # known = []
    # dev_cands = session.query(Candidate).filter(Candidate.split == 1).order_by(Candidate.id).all()
    # for i in range(L_gold_dev.shape[0]):
    #     if L_gold_dev[i,0] == 1:
    #         p1,p2 = dev_cands[i][0].get_span(), dev_cands[i][1].get_span()
    #         if re.search("(Dr|Mr|Mrs|Sir)",p1 + " "+ p2):
    #             continue
    #         if len(p1.split()) > 1 and len(p2.split()) > 1:
    #             #display_candidate(dev_cands[i])
    #             known.append( (p1,p2) )


    # print len(set(known))
    # for c in sorted(set(known)):
    #     print ",".join(c)



# exercises


def check_exercise_1(subclass):
    """
    Check if type is Person
    :param subclass:
    :return:
    """
    v = subclass.__mapper_args__['polymorphic_identity'] == "person"
    v &= len(subclass.__argnames__) == 1 and 'person' in subclass.__argnames__
    print 'Correct!' if v else 'Sorry, try again!'


def check_exercise_2(c):
    s1 = c[0].get_span()
    s2 = c[1].get_span()
    print 'Correct!' if "{} {}".format(s1, s2) == "Katrina Dawson Paul Smith" else 'Sorry, try again!'