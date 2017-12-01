import csv

from scipy.sparse import csr_matrix

from snorkel.annotations import csr_LabelMatrix

NUM_SPLITS = 3

class QalfConverter(object):
    """Converts a matrix_*.tsv qalf file into csrAnnotationMatrix's."""
    def __init__(self, session, candidate_class):
        self.session = session
        self.candidate_class = candidate_class
    
    def convert(self, matrix_tsv_path):
        candidate_map = {}
        split_sizes = [0] * NUM_SPLITS
        for split in [0, 1, 2]:
            candidates = self.session.query(self.candidate_class).filter(
                self.candidate_class.split == split).order_by(
                    self.candidate_class.id).all()
            split_sizes[split] = len(candidates)
            for c in candidates:
                candidate_map[c.get_stable_id()] = (c.id, split)
                
        label_matrices = self.tsv_to_matrix(matrix_tsv_path, candidate_map)

        for i, label_matrix in enumerate(label_matrices):
            assert(label_matrix.shape[0] == split_sizes[i])
            
        return label_matrices

    def tsv_to_matrix(self, matrix_tsv_path, candidate_map):
        """
        Args:
            matrix_tsv_path: path to tsv where first column is candidate_ids
                and all remaining columns contains labels from qa queries.
            candidate_map: dict mapping candidate_ids to their split
        Returns:
            L_train, L_dev, L_test: a csrAnnotationMatrix for each split
        """
        rows = [[], [], []]
        cols = [[], [], []]
        data = [[], [], []]
        row_ids = [[], [], []]
        col_ids = [[], [], []]
        candidate_count = [0] * 3

        misses = 0
        with open(matrix_tsv_path, 'rb') as tsv:
            tsv_reader = csv.reader(tsv, delimiter='\t')
            for row in tsv_reader:
                candidate_id = row[0]

                orm_id, split = candidate_map[candidate_id]
                i = candidate_count[split]
                candidate_count[split] += 1

                row_ids[split].append(orm_id)
                
                labels = row[1:]
                if not col_ids[split]:
                    # Just use indices as column ids
                    col_ids[split] = range(len(labels))

                for j, label in enumerate(labels):
                    label = int(label)
                    if label:
                        rows[split].append(i)
                        cols[split].append(j)
                        data[split].append(label)

        label_matrices = [None] * NUM_SPLITS
        for split in [0, 1, 2]:
            csr = csr_matrix((data[split], (rows[split], cols[split])), 
                shape=(len(row_ids[split]), len(col_ids[split]))).tocsr()
            label_matrices[split] = self.csr_to_labelmatrix(
                csr, row_ids[split], col_ids[split])
        return label_matrices

    def csr_to_labelmatrix(self, csr, row_ids, col_ids):
        # NOTE: col_ids currently goes unused
        candidate_index = {candidate_id: i for i, candidate_id in enumerate(row_ids)}
        row_index = {v: k for k, v in candidate_index.items()}
        return csr_LabelMatrix(csr, 
                               candidate_index=candidate_index,
                               row_index=row_index)

# NOTE:
# This is not yet a completely valid LabelMatrix, as we do not create the
# col_index or key_index. Once you do, you should be able to run the following
# lines to see a printout of LF performance:

# from snorkel.annotations import load_gold_labels
# L_gold_dev = load_gold_labels(session, annotator_name='gold', split=1)
# print(L_dev.lf_stats(session, labels=L_gold_dev))