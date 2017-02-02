import csv

def split_data(labels_file, labeled_docs_file, train, dev, test):
    # get list of labeled/unlabeled cos
    labeled_docs = set()
    with open(labels_file, 'rb') as tsvin, open(labeled_docs_file, 'wb') as csvout:
        reader = csv.reader(tsvin, delimiter='\t')
        reader.next()
        writer = csv.writer(csvout)
        for row in reader:
            (person1, person2, label) = row
            doc = person1[:person1.index(':')]
            labeled_docs.add(doc)
        for doc in labeled_docs:
            writer.writerow([doc])

def main():
    labels_file='gold_labels.tsv'
    labeled_docs_file = 'labeled_docs.csv'
    split_data(labels_file, labeled_docs_file, train=0.40, dev=0.40, test=0.20)


if __name__ == '__main__':
    main()