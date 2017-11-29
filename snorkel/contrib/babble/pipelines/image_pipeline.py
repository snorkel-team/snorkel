import csv
from itertools import product
import os
import random
import re
import shutil

import numpy as np
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

from snorkel.parser import ImageCorpusExtractor, CocoPreprocessor
from snorkel.models import StableLabel
from snorkel.db_helpers import reload_annotator_labels
from snorkel.annotations import load_marginals, load_gold_labels

from snorkel.contrib.babble.pipelines import BabblePipeline, final_report

from tutorials.babble import MTurkHelper

TRAIN = 0
DEV = 1
TEST = 2


class ImagePipeline(BabblePipeline):

    def extract(self):
        print("Extraction was performed during parse stage.")
        for split in self.config['splits']:
            num_candidates = self.session.query(self.candidate_class).filter(
                self.candidate_class.split == split).count()
            print("Candidates [Split {}]: {}".format(split, num_candidates))
    
    def classify(self, config=None, slim_ws_path=None):
        if config:
            self.config = config
        if not slim_ws_path:
            slim_ws_path = self.config['slim_ws_path']

        def get_candidates(self, split):
            return self.session.query(self.candidate_class).filter(
                self.candidate_class.split == split)

        def create_csv(dataset_dir, filename, coco_ids, labels, setname=None):
            csv_name = os.path.join(dataset_dir, filename)
            with open(csv_name, 'w') as csvfile:
                csvwriter = csv.writer(csvfile)
                num_images = 0

                for idx in range(len(coco_ids)):
                    if coco_ids[idx] == 0:
                        continue
                    else:
                        num_images += 1
                        url = 'http://images.cocodataset.org/{}/{:012}.jpg'.format(setname,int(coco_ids[idx]))
                        csvwriter.writerow([url,labels[idx]])
            return num_images

        def link_images_candidates(anns, candidates, mscoco, marginals):
            """
            Stores a max-pooled label per image based on bbox-level annotations.
            :param anns: np.array (of what?)
            :param candidates: list of candidates.
            :param mscoco: np.array (of what?)
            :param marginals: np.array of marginal probababilities per candidate.
            """
            coco_ids =  np.zeros(len(anns))
            labels = np.zeros(len(anns))
            num_candidates = len(candidates)

            for idx in range(num_candidates):
                cand = candidates[idx]
                image_id = int(cand[1].stable_id.split(":")[1])
                mscoco_id = mscoco[image_id]

                coco_ids[image_id] = int(mscoco_id)
                
                #HACK: sometimes marginals[idx] is a float, sometimes a matrix...
                try:
                    labels[image_id] = max(labels[image_id], max(marginals[idx], 0))
                except:
                    try:
                        labels[image_id] = max(labels[image_id], max(marginals[idx].todense(), 0))
                    except:
                        import pdb; pdb.set_trace()

            return coco_ids, labels

        def print_settings(settings):
            for k, v in sorted(settings.items()):
                print("{}: {}".format(k, v))

        def scrape_output(output_file):
            with open(output_file, mode='rb') as output:
                value_rgx = r'eval/\w+\[([\d\.]+)\]'
                for row in output:
                    if 'eval/Accuracy' in row:
                        accuracy = float(re.search(value_rgx, row).group(1))
                    elif 'eval/Precision' in row:
                        precision = float(re.search(value_rgx, row).group(1))
                    elif 'eval/Recall' in row:
                        recall = float(re.search(value_rgx, row).group(1))
                    else:
                        continue
                return accuracy, precision, recall

        if self.config['seed']:
            np.random.seed(self.config['seed'])

        dataset_dir = os.path.join(slim_ws_path, 'datasets/mscoco/', self.config['domain'])
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

        X_train = self.get_candidates(TRAIN)
        X_val = self.get_candidates(DEV)
        Y_val = np.array(load_gold_labels(self.session, annotator_name='gold', split=1).todense()).ravel()

        # Save out Validation Images and Labels
        if not getattr(self, 'anns_path', False):
            self.anns_path = self.config['anns_path']
        val_anns = np.load(self.anns_path + self.config['domain'] + '_val_anns.npy').tolist()
        val_mscoco = np.load(self.anns_path + self.config['domain'] + '_val_mscoco.npy')
        val_coco_ids, val_labels = link_images_candidates(val_anns, X_val, val_mscoco, Y_val)

        # Split validation set 50/50 into val/test
        num_labeled = len(val_labels)
        assignments = np.random.permutation(num_labeled)
        val_assignments = assignments[:num_labeled*2/5]
        test_assignments = assignments[num_labeled*2/5:]
        test_coco_ids, test_labels = val_coco_ids[test_assignments], val_labels[test_assignments]
        val_coco_ids, val_labels = val_coco_ids[val_assignments], val_labels[val_assignments]

        num_dev = create_csv(dataset_dir, 'validation_images.csv', val_coco_ids, val_labels, 'val2017')
        num_test = create_csv(dataset_dir, 'test_images.csv', test_coco_ids, test_labels, 'val2017')

        train_anns = np.load(self.anns_path + self.config['domain'] + '_train_anns.npy').tolist()
        train_mscoco = np.load(self.anns_path + self.config['domain'] + '_train_mscoco.npy')

        # If we're in traditional supervision mode, use hard marginals from the train set
        if self.config['supervision'] == 'traditional':
            print("In 'traditional' supervision mode...grabbing candidate and gold label subsets.")  
            candidates_train = self.get_candidates(TRAIN)
            Y_train = load_gold_labels(self.session, annotator_name='gold', split=TRAIN)
            #Deleted call to traditional_supervision, which was pruning by number of non-zeros
            if self.config['display_marginals'] and not self.config['no_plots']:
                plt.hist(Y_train, bins=20)
                plt.show()
        else:
            Y_train = (self.train_marginals if getattr(self, 'train_marginals', None) 
                is not None else load_marginals(self.session, split=TRAIN))

        train_coco_ids, train_labels = link_images_candidates(train_anns, X_train, train_mscoco, Y_train)
        num_train = create_csv(dataset_dir, 'train_images.csv', train_coco_ids, train_labels, 'train2017')

        print("Train size: {}".format(num_train))
        print("Dev size: {}".format(num_dev))
        print("Test size: {}".format(num_test))

        # Convert to TFRecords Format
        if self.config.get('download_data', False):
            print ('Downloading and converting images...')
            os.system('python ' + os.path.join(slim_ws_path, 'download_and_convert_data.py') + ' --dataset_name mscoco ' + ' --dataset_dir ' + dataset_dir)
        else:
            print("Assuming MSCOCO data is already downloaded and converted (download_data = False).")
        
        # Call TFSlim Model
        train_root = os.path.join(dataset_dir, 'train/')
        eval_root = os.path.join(dataset_dir, 'eval/')

        # Run homemade hacky random search
        # First, make random assignments in space of possible configurations
        param_names = self.config['disc_params_range'].keys()
        param_assignments = list(product(*[self.config['disc_params_range'][pn] for pn in param_names]))
        disc_params_list = [{k: v for k, v in zip(param_names, param_assignments[i])} for i in range(len(param_assignments))]
        # Randomnly select a small number of these to try
        random.shuffle(disc_params_list)
        disc_params_options = disc_params_list[:self.config['disc_model_search_space']]

        print("Starting training over space of {} configurations".format(
            min(self.config['disc_model_search_space'], len(disc_params_options))))

        accuracies, precisions, recalls = [], [], []
        lrs, weight_decays, max_stepses = [], [], []
        for i, disc_params in enumerate(disc_params_options):
            train_dir = os.path.join(train_root, "config_{}".format(i))
            eval_dir = os.path.join(eval_root, "config_{}".format(i))
            print("\nConfiguration {}.".format(i, eval_dir))
            print("Running the following configuration:".format(i))
            print_settings(disc_params)

            print('Calling TFSlim train...')
            # TODO: launch these in parallel
            # Remove the train_dir so no checkpoints are kept
            if os.path.exists(train_dir):
                shutil.rmtree(train_dir)
            os.makedirs(train_dir)
            
            train_cmd = 'python ' + slim_ws_path + 'train_image_classifier.py ' + \
                ' --train_dir=' + train_dir + \
                ' --dataset_name=mscoco' + \
                ' --dataset_split_name=train' + \
                ' --dataset_dir=' + dataset_dir + \
                ' --model_name=' + self.config['disc_model_class'] + \
                ' --optimizer=' + str(self.config['optimizer']) + \
                ' --opt_epsilon=' + str(self.config['opt_epsilon']) + \
                ' --num_clones=' + str(self.config['parallelism']) + \
                ' --log_every_n_steps=' + str(self.config['print_freq']) + \
                ' --learning_rate=' + str(disc_params['lr']) + \
                ' --weight_decay=' + str(disc_params['weight_decay']) + \
                ' --max_number_of_steps=' + str(disc_params['max_steps'])
            os.system(train_cmd)

            print('Calling TFSlim eval on validation...')
            output_file = os.path.join(eval_dir, 'output.txt')
            if not os.path.exists(eval_dir):
                os.makedirs(eval_dir)
            eval_cmd = 'python '+ slim_ws_path + 'eval_image_classifier.py ' + \
                  ' --dataset_name=mscoco ' + \
                  ' --dataset_split_name=validation' + \
                  ' --dataset_dir=' + dataset_dir + \
                  ' --checkpoint_path=' + train_dir + \
                  ' --eval_dir=' + eval_dir + \
                  ' --dataset_split_name=validation ' + \
                  ' --model_name=' + self.config['disc_model_class'] + \
                  ' --batch_size=' + self.config['disc_eval_batch_size'] + \
                  ' | tee -a ' + output_file
            ### TEMP ###
            # You added the batch_size parameter above
            ### TEMP ###
            os.system(eval_cmd)

            # Scrape results from output.txt 
            import pdb; pdb.set_trace()
            accuracy, precision, recall = scrape_output(output_file)
            print("Accuracy: {}".format(accuracy))
            print("Precision: {}".format(precision))
            print("Recall: {}".format(recall))
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            lrs.append(disc_params['lr'])
            weight_decays.append(disc_params['weight_decay'])
            max_stepses.append(disc_params['max_steps'])
        
        # Calculate F1 scores
        f1s = [float(2 * p * r)/(p + r) if p and r else 0 for p, r in zip(precisions, recalls)]
        dev_results = {
            'accuracy':     pd.Series(accuracies),
            'precision':    pd.Series(precisions),
            'recall':       pd.Series(recalls),
            'f1':           pd.Series(f1s),
            'lrs':          pd.Series(lrs),
            'weight_decays':pd.Series(weight_decays),
            'max_stepses':  pd.Series(max_stepses)
        }
        dev_df = pd.DataFrame(dev_results)
        print("\nDev Results: {}")
        print(dev_df)
        best_config_idx = dev_df['f1'].idxmax()

        # Identify best configuration and run on test
        print("\nBest configuration ({}):".format(best_config_idx))
        print_settings(disc_params_options[best_config_idx])
        checkpoint_path = os.path.join(train_root, "config_{}".format(best_config_idx))
        eval_dir = os.path.join(eval_root, "config_{}".format(best_config_idx))
        test_file = os.path.join(eval_dir, 'test_output.txt')

        print('\nCalling TFSlim eval on test...')
        os.system('python '+ slim_ws_path + 'eval_image_classifier.py ' + \
                 ' --dataset_name=mscoco '
                 ' --dataset_split_name=test' + \
                 ' --dataset_dir=' + dataset_dir + \
                 ' --checkpoint_path=' + checkpoint_path + \
                 ' --eval_dir=' + eval_dir + \
                 ' --dataset_split_name=test ' + \
                 ' --model_name=' + str(self.config['disc_model_class']) + \
                 ' | tee -a ' + test_file)
                
        
        accuracy, precision, recall = scrape_output(test_file)
        p, r = precision, recall
        f1 = float(2 * p * r)/(p + r) if p and r else 0

        test_results = {
            # 'accuracy':     pd.Series([accuracy]),
            'precision':    pd.Series([precision]),
            'recall':       pd.Series([recall]),
            'f1':           pd.Series([f1])
        }
        test_df = pd.DataFrame(test_results)
        print("\nTest Results: {}")
        print(test_df)

        if not getattr(self, 'scores', False):
            self.scores = {}
        self.scores['Disc'] = [precision, recall, f1]
        print("\nWriting final report to {}".format(self.config['log_dir']))
        final_report(self.config, self.scores)