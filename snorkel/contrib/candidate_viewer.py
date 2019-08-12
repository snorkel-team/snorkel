"""
Viewing/Printing information of extracted candidates in Snorkel
"""

import sqlite3

__author__ = "Pedram Hosseini"
__email__ = 'pdr.hosseini@gmail.com'

"""
-------------------------------------------
============== Example usage ==============
-------------------------------------------
from candidate_viewer import CandidateViewer

# creating an object of CandidateViewer class
cv = CandidateViewer("snorkel.db")

# printing extracted candidates
# where 'bio_rel' in the name of relation defined using Snorkel's candidate_subclass and 'entity1' and 'entity2'
# are arguments of the relation
cv.print_candidates(['bio_rel', ['entity1', 'entity2']], "article-1", max_output=10, keywords=["macrophages", "iNOS"])

# getting a document's text and sentences
doc_text, doc_sents, doc_sents_start_idx = cv.get_doc_text("article-1")
print("==Document Text\n" + doc_text, "\n")
print("==Document Sentences (count: {0})\n".format(len(doc_sents)), doc_sents, "\n")
print("==Document Sentences Start Indexes\n", doc_sents_start_idx, "\n")
"""


class CandidateViewer:
    def __init__(self, db_path):
        self.conn = self._create_connection(db_path)

    @staticmethod
    def _create_connection(db_path):
        """ create a database connection to snorkel SQLite database
            :param db_path: path to snorkel database, by default it's snorkel.db
            :return: Connection object or None
            """
        try:
            conn = sqlite3.connect(db_path)
            return conn
        except sqlite3.Error as e:
            print("[log-CandidateViewer] error in creating connection to snorkel database. [Details]: " + str(e))
            return None

    def _db_select(self, query):
        try:
            cur = self.conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
            return rows
        except Exception as e:
            print("[log-CandidateViewer] failed to execute select query. [Details]: " + str(e))
            return None

    def _get_span_text(self, span_id):
        """
        getting actual text string of a span
        :param span_id: span id
        :return:
        """
        try:
            span_result = self._db_select(
                "SELECT sentence_id, char_start, char_end FROM span WHERE id = " + str(span_id) + "")
            sentence_id = span_result[0][0]
            char_start = span_result[0][1]
            char_end = span_result[0][2]

            sentence_result = self._db_select("SELECT id, text FROM sentence WHERE id = " + str(sentence_id) + "")
            sentence_text = sentence_result[0][1]

            return sentence_text[char_start:char_end + 1]
        except Exception as e:
            print("[log-CandidateViewer] failed to get span string from database. [Details]: " + str(e))
            return None

    def print_candidates(self, rel_info, doc_id, max_output=20, keywords=[]):
        """
        printing extracted candidates in a document
        :param rel_info: [x, [y, z]] where x is the table name and y, and z are the relation arguments (take a look at
        the candidate class you defined using candidate_subclass to find these two names)
        :param doc_id: id of document in input file
        :param max_output: maximum number of candidates to show in the output
        :param keywords: list of keyword strings to be used in searching arg1 and arg1 in a relation
        :return:
        """

        try:
            # getting list of extracted candidates
            candidates_query = "SELECT id, " + rel_info[1][0] + "_id, " + rel_info[1][1] + "_id FROM " + rel_info[0] + ""
            # [id, entity1_id, entity2_id]
            candidates_results = self._db_select(candidates_query)

            # printing
            for i in range(len(candidates_results)):
                arg1 = candidates_results[i][1]
                arg2 = candidates_results[i][2]
                arg1_text = self._get_span_text(arg1)
                arg2_text = self._get_span_text(arg2)

                if not len(keywords) or (any(k in arg1_text for k in keywords) or any(k in arg2_text for k in keywords)):
                    print("[" + doc_id + "] " + arg1_text + " <-----> " + arg2_text)
                    # check the max print for the output
                    if i >= max_output - 1:
                        break
        except Exception as e:
            print("[log-CandidateViewer] error in printing candidates. [Details]: " + str(e))

    def get_doc_text(self, doc_name):
        """
        getting text string of a document along with list of sentences and their starting indexes
        :param doc_name: name of the document
        :return: doc_text, doc_sents, doc_sents_start_idx
        """
        doc_sents = []
        doc_sents_start_idx = []

        tmp = self._db_select("SELECT id FROM document WHERE name = \"" + doc_name + "\"")
        if tmp:
            doc_id = tmp[0][0]
            doc_sent_query = "SELECT stable_id FROM context WHERE type = \"sentence\" and stable_id like \"" + doc_name + "::%\""
            doc_sent_info = self._db_select(doc_sent_query)
            doc_sent_query = "SELECT text FROM sentence WHERE document_id = " + str(doc_id) + ""
            doc_sent_text = self._db_select(doc_sent_query)

            if doc_sent_info and doc_sent_text:
                doc_text = ""
                for i in range(len(doc_sent_info)):
                    if i < len(doc_sent_info) - 1:

                        doc_text = doc_text + doc_sent_text[i][0]
                        doc_sents.append(doc_sent_text[i][0])

                        id_info = doc_sent_info[i][0].split(":")
                        doc_sents_start_idx.append(id_info[3])
                        end_index = int(id_info[4])

                        id_info = doc_sent_info[i+1][0].split(":")
                        start_index = int(id_info[3])

                        diff = start_index - end_index
                        if diff > 0:
                            for j in range(diff):
                                doc_text = doc_text + " "

                doc_text = doc_text + doc_sent_text[len(doc_sent_text)-1][0]

                # adding information of the last sentence
                id_info = doc_sent_info[len(doc_sent_info) - 1][0].split(":")
                doc_sents.append(doc_sent_text[len(doc_sent_text)-1][0])
                doc_sents_start_idx.append(int(id_info[3]))

                return doc_text, doc_sents, doc_sents_start_idx
