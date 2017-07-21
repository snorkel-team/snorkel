
def create_serialized_candidate_view(session, C, verbose=False):
    """Creates a view in the database for a Candidate sub-class C defined over
    Span contexts, which are direct children of a single sentence.

    Creates VIEW with schema:
        candidate.id, candidate.split, span0.*, ..., spanK.*, sentence.*

    NOTE: This limited functionality should be expanded for arbitrary context
    trees. Also this should be made more dialect-independent.
    """
    selects, froms, joins = [], [], []
    for i, arg in enumerate(C.__argnames__):
        # Select the context CID
        selects.append("{0}.{1}_cid".format(C.__tablename__, arg))
        # Then all of its span columns
        selects.append("span{0}.*".format(i))
        froms.append("span AS span{0}".format(i))
        joins.append("{0}.{1}_id = span{2}.id".format(C.__tablename__, arg, i))

    sql = """
    CREATE VIEW IF NOT EXISTS {0}_serialized AS
        SELECT
            candidate.id,
            candidate.split,
            {1},
            sentence.*
        FROM
            candidate,
            {0},
            {2},
            sentence
        WHERE
            candidate.id = {0}.id
            AND sentence.id = span0.sentence_id
            AND {3}
    """.format(
        C.__tablename__,
        ", ".join(selects),
        ", ".join(froms),
        " AND ".join(joins)
    )
    if verbose:
        print("Creating view...")
        print(sql)
    session.execute(sql)