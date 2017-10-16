import os
import numpy as np
import csv

from snorkel.parser import ImageCorpusExtractor, CocoPreprocessor
from snorkel.models import StableLabel
from snorkel.db_helpers import reload_annotator_labels
from snorkel.annotations import load_marginals, load_gold_labels

from snorkel.contrib.babble import Babbler
from snorkel.contrib.babble.pipelines import BabblePipeline

from tutorials.babble import MTurkHelper


class BikePipeline(BabblePipeline):

    def parse(self, anns_path=os.environ['SNORKELHOME'] + '/tutorials/babble/bike/data/'):
        self.anns_path = anns_path
        train_path = anns_path + 'train_anns.npy'
        val_path = anns_path + 'val_anns.npy'

        corpus_extractor = ImageCorpusExtractor(candidate_class=self.candidate_class)

        coco_preprocessor = CocoPreprocessor(train_path, source=0)
        corpus_extractor.apply(coco_preprocessor)

        coco_preprocessor = CocoPreprocessor(val_path, source=1)
        corpus_extractor.apply(coco_preprocessor, clear=False)


    def extract(self):
        print("Extraction was performed during parse stage.")
        for split in self.config['splits']:
            num_candidates = self.session.query(self.candidate_class).filter(
                self.candidate_class.split == split).count()
            print("Candidates [Split {}]: {}".format(split, num_candidates))

    def load_gold(self, anns_path=None, annotator_name='gold'):
        if anns_path:
            self.anns_path = anns_path
            
        def load_labels(set_name, output_csv_path):
            helper = MTurkHelper(candidates=[], labels=[], num_hits=None, domain='vg', workers_per_hit=3)
            labels_by_candidate = helper.postprocess_visual(output_csv_path, 
                                                            is_gold=True, set_name=set_name, 
                                                            candidates=[], verbose=False)
            return labels_by_candidate
            
        
        validation_labels_by_candidate = load_labels('val', self.anns_path+
                                                     'Labels_for_Visual_Genome_all_out.csv')
        train_labels_by_candidate = load_labels('train', self.anns_path+
                                                'Train_Labels_for_Visual_Genome_out.csv')

        def assign_gold_labels(labels_by_candidate):
            for candidate_hash, label in labels_by_candidate.items():
                set_name, image_idx, bbox1_idx, bbox2_idx = candidate_hash.split(':')
                source = {'train': 0, 'val': 1}[set_name]
                stable_id_1 = "{}:{}::bbox:{}".format(source, image_idx, bbox1_idx)
                stable_id_2 = "{}:{}::bbox:{}".format(source, image_idx, bbox2_idx)
                context_stable_ids = "~~".join([stable_id_1, stable_id_2])
                query = self.session.query(StableLabel).filter(StableLabel.context_stable_ids == context_stable_ids)
                query = query.filter(StableLabel.annotator_name == annotator_name)
                label = 1 if label else -1
                if query.count() == 0:
                    self.session.add(StableLabel(
                        context_stable_ids=context_stable_ids,
                        annotator_name=annotator_name,
                        value=label))

            self.session.commit()
            reload_annotator_labels(self.session, self.candidate_class, 
                annotator_name, split=source, filter_label_split=False)
            
            
        assign_gold_labels(validation_labels_by_candidate)
        assign_gold_labels(train_labels_by_candidate)

    def collect(self):
        helper = MTurkHelper()
        output_csv_path = (os.environ['SNORKELHOME'] + 
                        '/tutorials/babble/bike/data/VisualGenome_all_out.csv')
        explanations = helper.postprocess_visual(output_csv_path, set_name='train', verbose=False)
        
        from snorkel.contrib.babble import link_explanation_candidates
        candidates = self.session.query(self.candidate_class).filter(self.candidate_class.split == self.config['babbler_candidate_split']).all()
        explanations = link_explanation_candidates(explanations, candidates)
        user_lists = {}
        super(BikePipeline, self).babble('image', explanations, user_lists, self.config)


    def classify(self, model_path = '/dfs/scratch0/paroma/slim_ws/', opt_b = 0.5):
        config = self.config

        def get_candidates(self, split):
            return self.session.query(self.candidate_class).filter(
                self.candidate_class.split == split)

        def create_csv(coco_ids, labels, filename, setname=None):
            csv_name = model_path+'datasets/mscoco/'+filename
            with open(csv_name, 'w') as csvfile:
                csvwriter = csv.writer(csvfile)
                
                for idx in range(len(coco_ids)):
                    if coco_ids[idx] == 0:
                        continue
                    else:
                        url = 'http://images.cocodataset.org/{}/{:012}.jpg'.format(setname,int(coco_ids[idx]))
                        csvwriter.writerow([url,labels[idx]])

        def link_images_candidates(anns, candidates, mscoco, marginals):
            coco_ids =  np.zeros(len(anns))
            labels = np.zeros(len(anns))
            num_candidates = len(candidates)

            for idx in range(num_candidates):
                cand = candidates[idx]
                image_id = int(cand.bike.stable_id.split(":")[1])
                mscoco_id = mscoco[image_id]

                coco_ids[image_id] = int(mscoco_id)
                try:
                    labels[image_id] = max(labels[image_id], max(marginals[idx], 0))
                except:
                    import pdb; pdb.set_trace()

            return coco_ids, labels

        if config:
            self.config = config

        if self.config['seed']:
            np.random.seed(self.config['seed'])

        X_train = self.get_candidates(0)
        Y_train = self.train_marginals
        Y_train_gold = np.array(load_gold_labels(self.session, annotator_name='gold', split=0).todense()).ravel()
        X_val = self.get_candidates(1)
        Y_val = np.array(load_gold_labels(self.session, annotator_name='gold', split=1).todense()).ravel()

        #Save out Validation Images and Labels
        val_anns = np.load(self.anns_path + 'val_anns.npy').tolist()
        val_mscoco = np.load(self.anns_path+'val_mscoco.npy')
        val_coco_ids, val_labels = link_images_candidates(val_anns, X_val, val_mscoco, Y_val)
        create_csv(val_coco_ids, val_labels, 'validation_images.csv', 'val2017')

        train_anns = np.load(self.anns_path + 'train_anns.npy').tolist()
        train_mscoco = np.load(self.anns_path+'train_mscoco.npy')

        #Depending on value of self.config['traditional'], create train marginals
        if self.config['supervision'] == 'traditional':
            train_size = self.config['traditional']
            train_coco_ids, train_labels = link_images_candidates(train_anns, X_train, train_mscoco, Y_train_gold)
            create_csv(train_coco_ids[:train_size], train_labels[:train_size], 'train_marginal_images.csv', 'train2017') #download script reads from one train file
        else:
            train_coco_ids, train_labels = link_images_candidates(train_anns, X_train, train_mscoco, Y_train)
            create_csv(train_coco_ids, train_labels, 'train_marginal_images.csv', 'train2017')

        #Convert to TFRecords Format
        #TODO: We are loading and converting images every time classify is called!!!
        print ('Loading Images...')
        os.system('python '+ model_path +'download_and_convert_data.py --dataset_name mscoco --dataset_dir ' + model_path+ '/datasets/mscoco')
        
        #Call TFSlim Model
        print ('Calling TFSlim...')
        train_dir = model_path+ '/datasets/mscoco'
        dataset_dir = train_dir
        os.system('python '+ model_path +'train_image_classifier.py --train_dir=' + train_dir + \
            ' --dataset_name=mscoco --dataset_split_name=train --dataset_dir=' + dataset_dir + \
            ' --model_name=' + self.config['disc_model_class'] + ' --num_clones=' + str(self.config['parallelism'])
            + ' --learning_rate=50.0 --max_number_of_steps 150') #don't know the setup of config file for lr etc params

        # scores = {}
        # with PrintTimer("[7.3] Evaluate discriminative model (opt_b={})".format(opt_b)):
        #     # Score discriminative model trained on generative model predictions
        #     np.random.seed(self.config['seed'])
        #     scores['Disc'] = score_marginals(val_marginals, Y_val, b=opt_b)

        # final_report(self.config, scores)
