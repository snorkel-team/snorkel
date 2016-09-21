from snorkel.models import Span, Label
from sqlalchemy.orm.exc import NoResultFound


def add_spouse_label(session, key, cls, person1, person2, value):
    try:
        person1 = session.query(Span).filter(Span.stable_id == person1).one()
        person2 = session.query(Span).filter(Span.stable_id == person2).one()
    except NoResultFound as e:
        if int(value) == -1:
            ### Due to variations in the NER output of CoreNLP, some of the included annotations for
            ### false candidates might cover slightly different text spans when run on some systems,
            ### so we must skip them.
            return
        else:
            raise e
    candidate = session.query(cls).filter(cls.person1 == person1).filter(cls.person2 == person2).first()
    if candidate is None:
        candidate = session.query(cls).filter(cls.person1 == person2).filter(cls.person2 == person1).one()

    label = session.query(Label).filter(Label.candidate == candidate).one_or_none()
    if label is None:
        label = Label(candidate=candidate, key=key, value=value)
        session.add(label)
    else:
        label.value = int(value)
    session.commit()
