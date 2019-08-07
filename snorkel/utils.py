from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

import re
import sys
import sqlite3
import numpy as np
import scipy.sparse as sparse


def get_ORM_instance(ORM_class, session, instance):
    """
    Given an ORM class and *either an instance of this class, or the name attribute of an instance
    of this class*, return the instance
    """
    if isinstance(instance, str):
        return session.query(ORM_class).filter(ORM_class.name == instance).one()
    else:
        return instance


def camel_to_under(name):
    """
    Converts camel-case string to lowercase string separated by underscores.

    Written by epost
    (http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case).

    :param name: String to be converted
    :return: new String with camel-case converted to lowercase, underscored
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def sparse_nonzero(X):
    """Sparse matrix with value 1 for i,jth entry !=0"""
    X_nonzero = X.copy()
    if not sparse.issparse(X):
        X_nonzero[X_nonzero != 0] = 1
        return X_nonzero
    if sparse.isspmatrix_csr(X) or sparse.isspmatrix_csc(X):
        X_nonzero.data[X_nonzero.data != 0] = 1
    elif sparse.isspmatrix_lil(X):
        X_nonzero.data = [np.ones(len(L)) for L in X_nonzero.data]
    else:
        raise ValueError("Only supports CSR/CSC and LIL matrices")
    return X_nonzero

def sparse_abs(X):
    """Element-wise absolute value of sparse matrix- avoids casting to dense matrix!"""
    X_abs = X.copy()
    if not sparse.issparse(X):
        return abs(X_abs)
    if sparse.isspmatrix_csr(X) or sparse.isspmatrix_csc(X):
        X_abs.data = np.abs(X_abs.data)
    elif sparse.isspmatrix_lil(X):
        X_abs.data = np.array([np.abs(L) for L in X_abs.data])
    else:
        raise ValueError("Only supports CSR/CSC and LIL matrices")
    return X_abs


def matrix_coverage(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates that each LF labels.**
    """
    return np.ravel(sparse_nonzero(L).sum(axis=0) / float(L.shape[0]))


def matrix_overlaps(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates that each LF _overlaps with other LFs on_.**
    """
    L_nonzero = sparse_nonzero(L)
    return np.ravel(np.where(L_nonzero.sum(axis=1) > 1, 1, 0).T * L_nonzero / float(L.shape[0]))

def matrix_conflicts(L):
    """
    Given an N x M matrix where L_{i,j} is the label given by the jth LF to the ith candidate:
    Return the **fraction of candidates that each LF _conflicts with other LFs on_.**
    """
    B = L.copy()
    if not sparse.issparse(B):
        for row in range(B.shape[0]):
            if np.unique(np.array(B[row][np.nonzero(B[row])])).size == 1:
                B[row] = 0
        return matrix_coverage(sparse_nonzero(B))
    if not (sparse.isspmatrix_csc(B) or sparse.isspmatrix_lil(B) or sparse.isspmatrix_csr(B)):
        raise ValueError("Only supports CSR/CSC and LIL matrices")
    if sparse.isspmatrix_csc(B) or sparse.isspmatrix_lil(B):
        B = B.tocsr()
    for row in range(B.shape[0]):
        if np.unique(B.getrow(row).data).size == 1:
            B.data[B.indptr[row]:B.indptr[row+1]] = 0
    return matrix_coverage(sparse_nonzero(B))


def matrix_tp(L, labels):
    return np.ravel([
        np.sum(np.ravel((L[:, j] == 1).todense()) * (labels == 1)) for j in range(L.shape[1])
    ])

def matrix_fp(L, labels):
    return np.ravel([
        np.sum(np.ravel((L[:, j] == 1).todense()) * (labels == -1)) for j in range(L.shape[1])
    ])

def matrix_tn(L, labels):
    return np.ravel([
        np.sum(np.ravel((L[:, j] == -1).todense()) * (labels == -1)) for j in range(L.shape[1])
    ])

def matrix_fn(L, labels):
    return np.ravel([
        np.sum(np.ravel((L[:, j] == -1).todense()) * (labels == 1)) for j in range(L.shape[1])
    ])

def get_as_dict(x):
    """Return an object as a dictionary of its attributes"""
    if isinstance(x, dict):
        return x
    else:
        try:
            return x._asdict()
        except AttributeError:
            return x.__dict__


def sort_X_on_Y(X, Y):
    return [x for (y,x) in sorted(zip(Y,X), key=lambda t : t[0])]


def corenlp_cleaner(words):
  d = {'-RRB-': ')', '-LRB-': '(', '-RCB-': '}', '-LCB-': '{',
       '-RSB-': ']', '-LSB-': '['}
  return [d[w] if w in d else w for w in words]


def tokens_to_ngrams(tokens, n_max=3, delim=' '):
    N = len(tokens)
    for root in range(N):
        for n in range(min(n_max, N - root)):
            yield delim.join(tokens[root:root+n+1])

class SnorkelPrint:
    def __init__(self, db_path):
        self.conn = self.__create_connection(db_path)

    @staticmethod
    def __create_connection(db_path):
        """ create a database connection to snorkel SQLite database
            :param db_path: path to snorkel database, by default it's snorkel.db
            :return: Connection object or None
            """
        try:
            conn = sqlite3.connect(db_path)
            return conn
        except sqlite3.Error as e:
            print("[log-SnorkelPrint] error in creating connection to snorkel database. [Details]: " + str(e))
            return None

    def __db_select(self, query):
        try:
            cur = self.conn.cursor()
            cur.execute(query)
            rows = cur.fetchall()
            return rows
        except Exception as e:
            print("[log-SnorkelPrint] failed to execute select query. [Details]: " + str(e))
            return None

    def __get_span_text(self, span_id):
        """
        getting actual text string of a span
        :param span_id: span id
        :return:
        """
        try:
            span_result = self.__db_select(
                "SELECT sentence_id, char_start, char_end FROM span WHERE id = " + str(span_id) + "")
            sentence_id = span_result[0][0]
            char_start = span_result[0][1]
            char_end = span_result[0][2]

            sentence_result = self.__db_select("SELECT id, text FROM sentence WHERE id = " + str(sentence_id) + "")
            sentence_text = sentence_result[0][1]

            return sentence_text[char_start:char_end + 1]
        except Exception as e:
            print("[log-SnorkelPrint] failed to get span string from database. [Details]: " + str(e))
            return None

    def print_candidates(self, rel_info, doc_id, max_output=20):
        """
        printing extracted candidates in a document
        :param rel_info: [x, [y, z]] where x is the table name and y, and z are the relation arguments (take a look at
        the candidate class you defined using candidate_subclass to find these two names)
        :param doc_id: id of document in input file
        :param max_output: maximum number of candidates to show in the output
        :return:
        """

        try:
            # getting list of extracted candidates
            candidates_query = "SELECT id, " + rel_info[1][0] + "_id, " + rel_info[1][1] + "_id FROM " + rel_info[0] + ""
            # [id, entity1_id, entity2_id]
            candidates_results = self.__db_select(candidates_query)

            # printing
            for i in range(len(candidates_results)):
                arg1 = candidates_results[i][1]
                arg2 = candidates_results[i][2]
                print("[" + doc_id + "] " + self.__get_span_text(arg1) + " <-----> " + self.__get_span_text(arg2))
                # check the max print for the output
                if i >= max_output - 1:
                    break
        except Exception as e:
            print("[log-SnorkelPrint] error in printing candidates. [Details]: " + str(e))

    def get_doc_text(self, doc_name):
        """
        getting text string of a document along with list of sentences and their starting indexes
        :param doc_name: name of the document
        :return: doc_text, doc_sents, doc_sents_start_idx
        """
        doc_sents = []
        doc_sents_start_idx = []

        tmp = self.__db_select("SELECT id FROM document WHERE name = \"" + doc_name + "\"")
        if tmp is not None:
            doc_id = tmp[0][0]
            doc_sent_query = "SELECT stable_id FROM context WHERE type = \"sentence\" and stable_id like \"" + doc_name + "::%\""
            doc_sent_info = self.__db_select(doc_sent_query)
            doc_sent_query = "SELECT text FROM sentence WHERE document_id = " + str(doc_id) + ""
            doc_sent_text = self.__db_select(doc_sent_query)

            if doc_sent_info is not None and doc_sent_text is not None:
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
