from snorkel.learning import NoiseAwareModel


class SymbolTable:
    """Wrapper for dict to encode unknown symbols"""
    def __init__(self, starting_symbol=2): 
        self.s = starting_symbol
        self.d = dict()

    def get(self, w):
        if w not in self.d:
            self.d[w] = self.s
            self.s += 1
        return self.d[w]

    def lookup(self, w):
        return self.d.get(w, 1)


def generate_marks( (l,h,idx)): 
    return [(l,"{}{}".format('[[', idx)),(h+1,"{}{}".format(idx, ']]'))]
    

def mark_sentence(s,mids):
    all_marks = sorted([ y for m in mids for y in generate_marks(m) ], reverse=True)
    x         = list(s) # new copy each time.
    for (idx,v) in all_marks:
        x.insert(idx,v)
    #return ' '.join(x)
    return x


class reLSTM(NoiseAwareModel):
    """LSTM for relation extraction"""

    def _preprocess_data(self):
        

    def train(self, X, training_marginals, **hyperparams):
        """Trains the model; also must set self.X_train and self.w"""
        raise NotImplementedError()

    def marginals(self, X, **kwargs):
        raise NotImplementedError()

    def save(self, session, version=0):
        """Save the Parameter (weight) values, i.e. the model"""
        # Check for X_train and w
        if not hasattr(self, 'X_train') or self.X_train is None or not hasattr(self, 'w') or self.w is None:
            name = self.__class__.__name__
            raise Exception("{0}.train() must be run, and must set {0}.X_train and {0}.w".format(name))

        # Create and save a new set of Parameters- note that PK of params is (feature_key_id, param_set_id)
        # Note: We can switch to using bulk insert if this is too slow...
        for j, v in enumerate(self.w):
            session.add(Parameter(feature_key_id=self.X_train.col_index[j], value=v, version=version))
        session.commit()

    def load(self, session, version=0):
        """Load the Parameters into self.w, given parameter version"""
        q = session.query(Parameter.value).filter(Parameter.version == version)
        q = q.order_by(Parameter.feature_key_id)
        self.w = np.array([res[0] for res in q.all()])