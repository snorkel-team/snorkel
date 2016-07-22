from collections import defaultdict
import scipy.sparse as sparse

class Featurizer(object):
    def __init__(self, candidates, corpus):
        self.num_cand = candidates.num_candidates()
        self._features_by_id = defaultdict(list)
        print "Extracting features..."
        self.feats = self.extract_features(candidates, corpus)
        print "Extracted {} features for each of {} mentions".format(self.num_features(), self.num_candidates())

    def num_candidates(self):
        return self.num_cand

    def num_features(self):
        return self.feats.shape[1]

    def extract_features(self, candidates, corpus):
        f_index = self._get_feature_index(candidates, corpus)
        f_matrix = self._get_feature_matrix(f_index)
        return f_matrix

    def _get_feature_index(self, candidates, corpus):
        f_index = defaultdict(list)
        for j,cand in enumerate(candidates):
            for feat in self._featurize(cand, corpus.get_context(cand.context_id)):
                self._features_by_id[cand.id].append(feat)
                f_index[feat].append(j)
        return f_index

    def _featurize(self):
        raise NotImplementedError

    def _get_feature_matrix(self, f_index):
        # Apply the feature generator, constructing a sparse matrix incrementally
        # Note that lil_matrix should be relatively efficient as we proceed row-wise
        self.feats = sparse.lil_matrix((self.num_candidates(), len(f_index)))
        for j,feat in enumerate(f_index.keys()):
            for i in f_index[feat]:
                self.feats[i,j] = 1
        return self.feats

    def get_features_by_id(self, id):
        features = self._features_by_id[id]
        return features if features is not None else None

class NgramFeaturizer(Featurizer):
    def _featurize(self, sent, context):
        yield "Ngram_features_to_come"

class CellNgramFeaturizer(NgramFeaturizer):
    def _featurize(self, cell, context):
        for feat in super(CellNgramFeaturizer, self)._featurize(cell, context):
            yield feat
        for feat in self.get_cell_feats(cell, context):
            yield ''.join(["TABLE_",feat])

    def get_cell_feats(self, cell, context):
        yield "ROW_NUM_%s" % cell.row_num
        yield "COL_NUM_%s" % cell.col_num
        yield "HTML_TAG_" + cell.html_tag
        for attr in cell.html_attrs:
            yield "HTML_ATTR_" + attr
        for tag in cell.html_anc_tags:
            yield "HTML_ANC_TAG_" + tag
        for attr in cell.html_anc_attrs:
            yield "HTML_ANC_ATTR_" + attr
        # TODO: add cell span features here
