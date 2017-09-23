import collections
import csv
import itertools
import random
import re

import matplotlib.pyplot as plt
import numpy as np

from snorkel.contrib.babble import Explanation

def shuffle_lists(a, b):
    combined = zip(a, b)
    random.shuffle(combined)
    a, b = zip(*combined)
    return a, b


class MTurkHelper(object):
    def __init__(self, candidates=[], labels=[], num_hits=None, candidates_per_hit=4, 
        workers_per_hit=3, shuffle=True, pct_positive=None, seed=1234, domain=None):
        random.seed(seed)
        if pct_positive:
            assert(0 < pct_positive and pct_positive < 1)
        if shuffle:
            if labels:
                candidates, labels = shuffle_lists(candidates, labels)
            else:
                random.shuffle(candidates)
        if pct_positive:
            if not labels:
                raise Exception("Must provide labels to obtain target pct_positive.")
            total_candidates = num_hits * candidates_per_hit
            candidates, labels = self.balance_labels(candidates, labels, 
                total_candidates, pct_positive)
        self.candidates = candidates
        self.labels = labels
        self.num_hits = num_hits
        self.candidates_per_hit = candidates_per_hit
        self.workers_per_hit = workers_per_hit

    def balance_labels(self, candidates, labels, num_candidates, pct_positive):
        target_positive = int(num_candidates * pct_positive)
        target_negative = num_candidates - target_positive
        positive_candidates = []
        negative_candidates = []
        unknown_candidates = []
        for i, c in enumerate(candidates):
            if labels[i] == 1:
                positive_candidates.append(c)
            elif labels[i] == -1:
                negative_candidates.append(c)
            else:
                unknown_candidates.append(c)
        print("Found {} positive, {} negative, {} unknown candidates.".format(
            len(positive_candidates), len(negative_candidates), len(unknown_candidates)))
        if len(positive_candidates) < target_positive:
            raise Exception('Not enough positive candidates ({}) to satisfy '
                'target number of {} ({}%)'.format(len(positive_candidates),
                target_positive, pct_positive * 100))
        if len(negative_candidates) < target_negative:
            raise Exception('Not enough negative candidates ({}) to satisfy '
                'target number of {} ({}%)'.format(len(negative_candidates),
                target_negative, (1 - pct_positive) * 100))
        balanced_candidates = (positive_candidates[:target_positive] + 
                               negative_candidates[:target_negative])
        balanced_labels = [1] * target_positive + [-1] * target_negative
        print("Using {} positive, {} negative candidates.".format(
            target_positive, target_negative))
        return shuffle_lists(balanced_candidates, balanced_labels)

    def preprocess(self, csvpath):
        """
        Converts candidates into a csv file input for MTurk.
        """
        def batch_iter(iterable, batch_size):
            n = len(iterable)
            for i in range(0, n, batch_size):
                yield iterable[i:min(i + batch_size, n)]
        
        def highlighted(text):
            return '<span style="background-color: rgb(255, 255, 0);">' + text + '</span>'

        with open(csvpath, 'wb') as csvfile:
            csvwriter = csv.writer(csvfile)
            # write header row
            header = []
            contents = []
            spans = []
            labels = []
            for i in range(1, self.candidates_per_hit + 1):
                contents.append('content{}'.format(i))
                spans.append('span1_{}'.format(i))
                spans.append('span2_{}'.format(i))
                labels.append('goldlabel_{}'.format(i))
            header = spans + contents + labels
            csvwriter.writerow(header)

            # write data rows
            batcher = batch_iter(self.candidates, self.candidates_per_hit)
            i_candidate = 0
            for i_hit in range(self.num_hits):
                hit = []
                contents = []
                spans = []
                labels = []
                batch = batcher.next()
                for candidate in batch:
                    content = candidate.get_parent().text.strip()
                    for span in sorted(candidate.get_contexts(), key=lambda x: x.char_start, reverse=True):
                        content = content[:span.char_start] + highlighted(span.get_span()) + content[span.char_end + 1:]
                    contents.append(content.encode('utf-8'))
                    for span in candidate.get_contexts():
                        spans.append(span.stable_id)
                    if self.labels:
                        labels.append(self.labels[i_candidate])
                    i_candidate += 1
                hit = spans + contents + labels
                csvwriter.writerow(hit)
        print("Wrote {} HITs with {} candidates per HIT".format(self.num_hits, self.candidates_per_hit))

    def preprocess_visual(self, csvpath):
        """
        Converts candidates into a csv file input for MTurk for non candidate tasks
        """
        def batch_iter(iterable, batch_size):
            n = len(iterable)
            for i in range(0, n, batch_size):
                yield iterable[i:min(i + batch_size, n)]

        def image_template(id):
            return '<img class="img-responsive center-block" src="http://stanford.edu/~paroma/' + id + '" />'

        with open(csvpath, 'wb') as csvfile:
            csvwriter = csv.writer(csvfile)
            # write header row
            header = []
            contents = []
            for i in range(1, self.candidates_per_hit + 1):
                contents.append('content{}'.format(i))
            header = contents
            csvwriter.writerow(header)

            # write data rows
            batcher = batch_iter(self.candidates, self.candidates_per_hit)
            i_candidate = 0
            for i_hit in range(self.num_hits):
                hit = []
                contents = []
                labels = []
                batch = batcher.next()
                for candidate in batch:
                    content = image_template(candidate)
                    contents.append(content.encode('utf-8'))
                    i_candidate += 1
                hit = contents
                csvwriter.writerow(hit)
        print("Wrote {} HITs with {} candidates per HIT".format(self.num_hits, self.candidates_per_hit))
            
        
    def postprocess_visual(self, csvpath, is_gold=False, set_name=None, candidates=None, verbose=False):
        """
        Assumptions:
        HITs are sorted by HITId
        Don't need to pass original candidate list, parsed from $content
        """
        
        if set_name not in ['train','val']:
            raise ValueError('A valid set name was not passed')
        else:
            print 'Reading ' + set_name + ' CSV!'
      
            
        with open(csvpath, 'rb') as csvfile:
            csvreader = csv.reader(csvfile)
            
            # prep data structures
            explanations_by_candidate = collections.defaultdict(list)
            labels_by_candidate = collections.defaultdict(list)
            label_converter = {
                'true': True,
                'false': False,
                'not_person': None,
            }
            hits = collections.Counter()
            times = []
            workers = []

            # read data
            header = csvreader.next()
            for row in csvreader:
                img_indices = []             
                p_indices = []
                b_indices = []
                explanations = []
                labels = []
                for i, field in enumerate(row):
                    if header[i] == 'HITId':
                        hits.update([field])
                    elif header[i] == 'WorkTimeInSeconds':
                        times.append(int(field))
                    elif header[i] == 'WorkerId':
                        workers.append(field)
                    elif header[i].startswith('Input.content'):
                        #The HTML Parsing!
                        parsed_url = field.split('_')
                        if len(parsed_url) == 5:
                            img_indices.append(int(parsed_url[2]))
                            p_indices.append(int(parsed_url[3]))
                            b_indices.append(int(parsed_url[4].split('.')[0]))
                        else:
                            img_indices.append(int(parsed_url[1]))
                            p_indices.append(int(parsed_url[2]))
                            b_indices.append(int(parsed_url[3].split('.')[0]))
                        
                    elif header[i].startswith('Answer.explanation'):
                        explanations.append(field)
                    elif header[i].startswith('Answer.label'):
                        labels.append(field)
                    if is_gold:
                            explanations.append('temp exp')

                #import pdb; pdb.set_trace()                
                source = {'train': 0, 'val': 1}[set_name]
                for (img_idx, p_idx, b_idx, explanation, label) in zip(img_indices, p_indices, b_indices, explanations, labels):
                    p_bbox_stable_id = "{}:{}::bbox:{}".format(source, img_idx, p_idx)
                    b_bbox_stable_id = "{}:{}::bbox:{}".format(source, img_idx, b_idx)
                    candidate_stable_id = '~~'.join([p_bbox_stable_id, b_bbox_stable_id])
                    label = label_converter[label.lower()]
                    if label is None:
                        exp = None
                    else:
                        exp = Explanation(explanation, label, candidate=candidate_stable_id)
                    explanations_by_candidate[candidate_stable_id].append(exp)

   
            # Sanity check
            print("Num HITs unique: {}".format(len(hits)))
            print("Num HITs total: {}".format(sum(hits.values())))
            #assert(all([n == self.workers_per_hit for n in hits.values()]))

            # Analyze worker distribution
            if verbose:
                responses_by_worker = collections.Counter(workers)
                plt.hist(responses_by_worker.values(), bins='auto')
                plt.title('# Responses Per Worker')
                plt.show()

            # Analyze time distribution
            if verbose:
                median_time = int(np.median(times))
                print("Median # seconds/HIT: {:d} ({:.1f} s/explanation)".format(
                    median_time, median_time/self.candidates_per_hit))
                plt.hist(times, bins='auto')
                plt.title('Seconds per HIT')
                plt.show()

            # Link candidates
            if candidates:
                self.candidates = candidates
            if self.candidates:
                self.link_candidates(explanations_by_candidate)

            # Filter and merge data
            num_unanimous = 0
            num_majority = 0
            num_bad = 0
            valid_explanations = []
            for cand, explanations in explanations_by_candidate.items():
                consensus = None
                if None in explanations:
                    consensus = None
                    num_bad += 1
                    continue
                    
                labels = [exp.label for exp in explanations]
                for option in [True, False]:
                    if labels.count(option) == self.workers_per_hit:
                        consensus = option
                        num_unanimous += 1
                    #Temp Hack to remove 2700 Bad from Train Gold in Drink
                    elif labels.count(option) >= np.floor(self.workers_per_hit/2.0):
                        consensus = option
                        num_majority += 1
                if consensus == None:
                    print "Bad candidate detected", cand

                
                #Hack to maintain label by candidate format for bike_model
                cand_split = cand.split('~~')
                img_idx = cand_split[0].split(':')[1]
                p_idx = cand_split[0].split(':')[4]
                b_idx = cand_split[1].split(':')[4]
                labels_by_candidate['%s:%s:%s:%s'%(set_name,img_idx,p_idx,b_idx)] = consensus
                
                
                valid_explanations.extend([exp for exp in explanations if exp.label == consensus])
            for candidate_stable_id, responses in explanations_by_candidate.items():
                if (len(responses) != self.workers_per_hit) and (verbose):
                    print("Warning: for candidate {}, expected {} responses, found {}.".format(
                        candidate_stable_id, self.workers_per_hit, len(responses)))
            if self.num_hits:
                num_actual_explanations = num_unanimous + num_majority + num_bad
                num_expected_explanations = self.num_hits * self.candidates_per_hit
                if (num_actual_explanations != num_expected_explanations) and (verbose):
                    print("Warning: expected {} total explanations, found {}.".format(
                        num_expected_explanations, num_actual_explanations))
            print("Unanimous: {}".format(num_unanimous))
            print("Majority: {}".format(num_majority))
            print("Bad: {}".format(num_bad))

            if is_gold:
                return labels_by_candidate
            else:
                return valid_explanations
    

    def postprocess(self, csvpath, candidates=None, verbose=False):
        """
        Assumptions:
        HITs are sorted by HITId
        For v0.2 only, pass original candidate list again; otherwise, no need.
        """
        with open(csvpath, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            
            # prep data structures
            explanations_by_candidate = collections.defaultdict(list)
            label_converter = {
                'true': True,
                'false': False,
                'not_person': None,
            }
            hits = collections.Counter()
            times = []
            workers = []

            # read data
            header = csvreader.next()
            for row in csvreader:
                span1s = []
                span2s = []
                goldlabels = []
                explanations = []
                labels = []
                for i, field in enumerate(row):
                    if header[i] == 'HITId':
                        hits.update([field])
                    elif header[i] == 'WorkTimeInSeconds':
                        times.append(int(field))
                    elif header[i] == 'WorkerId':
                        workers.append(field)
                    elif header[i].startswith('Input.span1'):
                        span1s.append(field)
                    elif header[i].startswith('Input.span2'):
                        span2s.append(field)
                    elif header[i].startswith('Input.goldlabel'):
                        goldlabels.append(field)
                    elif header[i].startswith('Answer.explanation'):
                        explanations.append(re.sub(r'[\x80-\xff]+', "", field))
                    elif header[i].startswith('Answer.label'):
                        labels.append(field)

                for (span1, span2, explanation, label) in zip(span1s, span2s, explanations, labels):
                    candidate_stable_id = u'~~'.join([span1, span2])
                    label = label_converter[label.lower()]
                    if label is None:
                        exp = None
                    else:
                        exp = Explanation(explanation, label, candidate=candidate_stable_id)
                    explanations_by_candidate[candidate_stable_id].append(exp)
            
            # Sanity check
            print("Num HITs unique: {}".format(len(hits)))
            print("Num HITs total: {}".format(sum(hits.values())))
            assert(all([n == self.workers_per_hit for n in hits.values()]))

            # Analyze worker distribution
            if verbose:
                responses_by_worker = collections.Counter(workers)
                plt.hist(responses_by_worker.values(), bins='auto')
                plt.title('# Responses Per Worker')
                plt.show()

            # Analyze time distribution
            if verbose:
                median_time = int(np.median(times))
                print("Median # seconds/HIT: {:d} ({:.1f} s/explanation)".format(
                    median_time, median_time/self.candidates_per_hit))
                plt.hist(times, bins='auto')
                plt.title('Seconds per HIT')
                plt.show()

            # Link candidates
            if candidates:
                self.candidates = candidates
            if self.candidates:
                self.link_candidates(explanations_by_candidate)

            # Filter and merge data
            num_unanimous = 0
            num_majority = 0
            num_bad = 0
            valid_explanations = []
            for _, explanations in explanations_by_candidate.items():
                if None in explanations:
                    consensus = None
                    num_bad += 1
                    continue
                labels = [exp.label for exp in explanations]
                for option in [True, False]:
                    if labels.count(option) == self.workers_per_hit:
                        consensus = option
                        num_unanimous += 1
                    elif labels.count(option) >= np.floor(self.workers_per_hit/2.0 + 1):
                        consensus = option
                        num_majority += 1
                assert(consensus is not None)
                valid_explanations.extend([exp for exp in explanations if exp.label == consensus])
            # assert(all([len(responses) == self.workers_per_hit 
            #     for responses in explanations_by_candidate.values()]))
            if self.num_hits:
                num_actual_explanations = num_unanimous + num_majority + num_bad
                num_expected_explanations = self.num_hits * self.candidates_per_hit
                if num_actual_explanations != num_expected_explanations:
                    print("Warning: expected {} total explanations, found {}.".format(
                        num_expected_explanations, num_actual_explanations))
            print("Unanimous: {}".format(num_unanimous))
            print("Majority: {}".format(num_majority))
            print("Bad: {}".format(num_bad))


            # Link candidates
            print("Total explanations: {}".format(len(valid_explanations)))
            return valid_explanations


    def link_candidates(self, explanations_by_candidate):
        for candidate in self.candidates:
            stable_id = candidate.get_stable_id()
            if stable_id in explanations_by_candidate:
                for exp in explanations_by_candidate[stable_id]:
                    if exp is not None:
                        exp.candidate = candidate
        unlinked = 0
        total = 0
        for exp_list in itertools.chain(explanations_by_candidate.values()):
            for exp in exp_list:
                if exp is None:
                    continue
                total +=1
                if isinstance(exp.candidate, basestring):
                    unlinked += 1
                    # import pdb; pdb.set_trace()
        if unlinked:
            print("Warning: {} out of {} non-None explanations could not be "
                "linked to candidates.".format(unlinked, total))
            if unlinked == total:
                raise Exception("Candidate linking failed.")
