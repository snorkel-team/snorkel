# TODO: move this to tutorials

class CDRModel(SnorkelModel):
    """
    A class specifically intended for use with the CDR task/tutorial/dataset
    """
    def parse(self, file_path=(os.environ['SNORKELHOME'] + '/tutorials/cdr/data/CDR.BioC.xml'), clear=True):
        doc_preprocessor = XMLMultiDocPreprocessor(
            path=file_path,
            doc='.//document',
            text='.//passage/text/text()',
            id='.//id/text()',
            max_docs=self.config['max_docs']
        )
        # tagger_one = TaggerOneTagger()
        # fn=tagger_one.tag
        fn = None
        SnorkelModel.parse(self, doc_preprocessor, fn=fn, clear=clear)


    def extract(self, clear=True):
        with open(os.environ['SNORKELHOME'] + '/tutorials/cdr/data/doc_ids.pkl', 'rb') as f:
            train_ids, dev_ids, test_ids = cPickle.load(f)
        train_ids, dev_ids, test_ids = set(train_ids), set(dev_ids), set(test_ids)

        train_sents, dev_sents, test_sents = set(), set(), set()
        docs = self.session.query(Document).order_by(Document.name).all()
        for i, doc in enumerate(docs):
            for s in doc.sentences:
                if doc.name in train_ids:
                    train_sents.add(s)
                elif doc.name in dev_ids:
                    dev_sents.add(s)
                elif doc.name in test_ids:
                    test_sents.add(s)
                else:
                    raise Exception('ID <{0}> not found in any id set'.format(doc.name))

        candidate_extractor = PretaggedCandidateExtractor(self.candidate_class, ['Chemical', 'Disease'])
        for split, sents in enumerate([train_sents, dev_sents, test_sents]):
            if len(sents) > 0 and split in self.config['splits']:
                SnorkelModel.extract(self, candidate_extractor, sents, split=split, clear=clear)

    def load_gold(self, split=None):
        if not split:
            splits = self.config['splits']
        else:
            splits = [split] if not isinstance(split, list) else split
        for split in splits:
            nCandidates = self.session.query(self.candidate_class).filter(self.candidate_class.split == split).count()
            if nCandidates > 0:
                print("Split {}:".format(split))
                # load_external_labels(self.session, self.candidate_class, split=split, annotator='gold')

    def featurize(self, split=None, config=None):
        if config:
            self.config = config
        if not split:
            splits = self.config['splits']
        else:
            splits = [split] if not isinstance(split, list) else split         
        featurizer = FeatureAnnotator()
        for split in splits:
            nCandidates = self.session.query(self.candidate_class).filter(self.candidate_class.split == split).count()
            if nCandidates > 0:
                F = SnorkelModel.featurize(self, featurizer, split)
        self.featurizer = featurizer

    def generate_lfs(self, config=None):
        if config:
            self.config = config
        if self.config['source'] == 'py':
            from cdr_lfs import get_cdr_lfs
            LFs = get_cdr_lfs()
            if not self.config['include_py_only_lfs']:
                for lf in list(LFs):
                    if lf.__name__ in ['LF_closer_chem', 'LF_closer_dis', 'LF_ctd_marker_induce', 'LF_ctd_unspecified_induce']:
                        LFs.remove(lf)
                print("Removed 4 'py only' LFs...")
        elif self.config['source'] == 'nl':
            with bz2.BZ2File(os.environ['SNORKELHOME'] + '/tutorials/cdr/data/ctd.pkl.bz2', 'rb') as ctd_f:
                ctd_unspecified, ctd_therapy, ctd_marker = cPickle.load(ctd_f)
            user_lists = {
                'uncertain': ['combin', 'possible', 'unlikely'],
                'causal': ['causes', 'caused', 'induce', 'induces', 'induced', 'associated with'],
                'treat': ['treat', 'effective', 'prevent', 'resistant', 'slow', 'promise', 'therap'],
                'procedure': ['inject', 'administrat'],
                'patient': ['in a patient with', 'in patients with'],
                'weak': ['none', 'although', 'was carried out', 'was conducted', 'seems', 
                        'suggests', 'risk', 'implicated', 'the aim', 'to investigate',
                        'to assess', 'to study'],
                'ctd_unspecified': ctd_unspecified,
                'ctd_therapy': ctd_therapy,
                'ctd_marker': ctd_marker,
            }
            train_cands = self.session.query(self.candidate_class).filter(self.candidate_class.split == 0).all()
            # examples = get_examples('semparse_cdr', train_cands)
            examples = None
            sp = SemanticParser(
                self.candidate_class, 
                user_lists, 
                beam_width=self.config['beam_width'], 
                top_k=self.config['top_k'])
            print("Generating LFs with beam_width={0}, top_k={1}".format(
                self.config['beam_width'], self.config['top_k']))
            sp.evaluate(examples,
                        show_everything=False,
                        show_explanation=False,
                        show_candidate=False,
                        show_sentence=False,
                        show_parse=False,
                        show_passing=False,
                        show_correct=False,
                        pseudo_python=False,
                        remove_paren=self.config['remove_paren'],
                        paraphrases=self.config['paraphrases'],
                        only=[])
            (correct, passing, failing, redundant, erroring, unknown) = sp.LFs
            LFs = []
            for (name, lf_group) in [('correct', correct),
                                     ('passing', passing),
                                     ('failing', failing),
                                     ('redundant', redundant),
                                     ('erroring', erroring),
                                     ('unknown', unknown)]:
                if name in self.config['include']:
                    LFs += lf_group
                    print("Keeping {0} {1} LFs...".format(len(lf_group), name))
                else:
                    if len(lf_group) > 0:
                        print("Discarding {0} {1} LFs...".format(len(lf_group), name))
            if self.config['include_py_only_lfs']:
                from cdr_lfs import LF_closer_chem, LF_closer_dis, LF_ctd_marker_induce, LF_ctd_unspecified_induce
                LFs = sorted(LFs + [LF_closer_chem, LF_closer_dis, LF_ctd_marker_induce, LF_ctd_unspecified_induce], key=lambda x: x.__name__)
                print("Added 4 'py only' LFs...")
        else:
            raise Exception("Parameter 'source' must be in {'py', 'nl'}")
        
        if self.config['max_lfs']:
            if self.config['seed']:
                np.random.seed(self.config['seed'])
            np.random.shuffle(LFs)
            LFs = LFs[:self.config['max_lfs']]
        self.LFs = LFs
        print("Using {0} LFs".format(len(self.LFs)))

    def label(self, config=None):
        if config:
            self.config = config
        
        if self.LFs is None:
            print("Running generate_lfs() first...")
            self.generate_lfs()

        while True:
            labeler = LabelAnnotator(f=self.LFs)
            for split in self.config['splits']:
                if split==TEST:
                    continue
                nCandidates = self.session.query(self.candidate_class).filter(self.candidate_class.split == split).count()
                if nCandidates > 0:
                    L = SnorkelModel.label(self, labeler, split)
                    if split==TRAIN:
                        L_train = L
                    nCandidates, nLabels = L.shape
                    if self.config['verbose']:
                        print("\nLabeled split {}: ({},{}) sparse (nnz = {})".format(split, nCandidates, nLabels, L.nnz))
                        training_set_summary_stats(L, return_vals=False, verbose=True)
            self.labeler = labeler

            lf_useless = set()
            if self.config['filter_uniform_labels']:
                for i in range(L_train.shape[1]):
                    if abs(np.sum(L_train[:,i])) == L_train.shape[0]:
                        lf_useless.add(self.LFs[i])
                
            lf_twins = set()
            if self.config['filter_redundant_signatures']:
                signatures = set()
                L_train_coo = coo_matrix(L_train)
                row = L_train_coo.row
                col = L_train_coo.col
                data = L_train_coo.data
                for i in range(L_train.shape[1]):
                    signature = hash((hash(tuple(row[col==i])),hash(tuple(data[col==i]))))
                    if signature in signatures:
                        lf_twins.add(self.LFs[i])
                    else:
                        signatures.add(signature)
                lf_twins = lf_twins.difference(lf_useless)
            
            """
            NOTE: This method of removal is a total hack. Far better would be
            to create a sane slice method for the csr_AnnotationMatrix class and
            use that to simply slice only the relevant LFs.
            """
            lf_remove = lf_useless.union(lf_twins)
            if len(lf_remove) == 0:
                break
            else:
                print("Uniform labels filter found {} LFs".format(len(lf_useless)))
                print("Redundant signature filter found {} LFs".format(len(lf_twins)))
                self.LFs = [lf for lf in self.LFs if lf not in lf_remove]
                print("Filters removed a total of {} LFs".format(len(lf_remove)))
                print("Running label step again with {} LFs...\n".format(len(self.LFs)))

