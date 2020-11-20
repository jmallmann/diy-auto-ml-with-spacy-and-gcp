#!/usr/bin/env python
# coding: utf8
"""Train a convolutional neural network text classifier on a
 dataset, using the TextCategorizer component. The dataset will be loaded
automatically via Thinc's built-in dataset loader. The model is added to
spacy.pipeline, and predictions are available via `doc.cats`. For more details,
see the documentation:
* Training: https://spacy.io/usage/training

Compatible with: spaCy v2.0.0+
"""
from __future__ import unicode_literals, print_function
import plac
import random
import warnings
from pathlib import Path
import csv
from collections import defaultdict
import json

import spacy
from spacy.util import minibatch, compounding

output_dir_name = "./model"

# training data
TRAIN_DATA = [
    ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),
    ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),
]

@plac.annotations(
    input_file=("File path to dataset", 'positional', None, str),
    n_iter=("Number of training iterations", "option", "n", int),
    n_split=("Split for train vs test. Default 80% train", "option", "s", float),
)
def main(input_file, n_iter=14, n_split=0.8):
    output_dir = Path(output_dir_name)
    if not output_dir.exists():
        output_dir.mkdir()

    nlp = spacy.load("en_core_web_md")  # load existing spaCy model

    # add the text classifier to the pipeline if it doesn't exist
    # nlp.create_pipe works for built-ins that are registered with spaCy
    #if "textcat" not in nlp.pipe_names:
    #    textcat = nlp.create_pipe(
    #       "textcat", config={"exclusive_classes": True}
    #    )
    #   nlp.add_pipe(textcat, last=True)
    # otherwise, get it, so we can add labels to it
    #else:
    #    textcat = nlp.get_pipe("textcat")

    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner, last=True)
    # otherwise, get it so we can add labels
    else:
        ner = nlp.get_pipe("ner")

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    # only train NER
    with nlp.disable_pipes(*other_pipes), warnings.catch_warnings():
        # show warnings for misaligned entity spans once
        warnings.filterwarnings("once", category=UserWarning, module='spacy')

        # reset and initialize the weights randomly – but only if we're
        # training a new model
        ##if model is None:
        ##    nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(
                    texts,  # batch of texts
                    annotations,  # batch of annotations
                    drop=0.5,  # dropout - make it harder to memorise data
                    losses=losses,
                )
            print("Losses", losses)

    # test the trained model
    for text, _ in TRAIN_DATA:
        doc = nlp(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])

    ########

    #print("Loading data...")
    #(train_texts, train_cats), (dev_texts, dev_cats), largestLabel, labels = load_data(input_file, splitRatio=n_split)

    # add labels to text classifier
    #for label in labels:
    #   textcat.add_label(label.upper())

    #print(
    #    "{} training, {} evaluation".format(
    #        len(train_texts), len(dev_texts)
    #    )
    #)
    #train_data = list(zip(train_texts, [{"cats": cats} for cats in train_cats]))

    # get names of other pipes to disable them during training
    #pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
    #other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
    #with nlp.disable_pipes(*other_pipes):  # only train textcat
    #    optimizer = nlp.begin_training()
    #    print("Training the model...")
    #    print("{:^5}\t{:^5}\t{:^5}\t{:^5}".format("LOSS", "P", "R", "F"))
    #    batch_sizes = compounding(4.0, 32.0, 1.001)
    #    for _ in range(n_iter):
    #        losses = {}
            # batch up the examples using spaCy's minibatch
    #        random.shuffle(train_data)
    #        batches = minibatch(train_data, size=batch_sizes)
    #        for batch in batches:
    #            texts, annotations = zip(*batch)
    #            nlp.update(texts, annotations, sgd=optimizer, drop=0.2, losses=losses)
    #        with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
    #            scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_cats, largestLabel.upper())
    #            print(
    #                "{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}".format(  # print a simple table
    #                    losses["textcat"],
    #                    scores["textcat_p"],
    #                    scores["textcat_r"],
    #                   scores["textcat_f"],
    #                )
    #            )
    #            with open('./stats.json', 'w') as fp:
    #                stats = {
    #                    "predict": "{0:.3f}".format(scores["textcat_p"]),
    #                    "recall": "{0:.3f}".format(scores["textcat_r"]),
    #                    "fvalue": "{0:.3f}".format(scores["textcat_f"]),
    #                }
    #                json.dump(stats, fp)

    # with nlp.use_params(optimizer.averages):
        nlp.to_disk(output_dir)
    print("Saved model to", output_dir)

    #test saved model
    print("Loading from", output_dir)
    nlp2 = spacy.load(output_dir)
    for text, _ in TRAIN_DATA:
        doc = nlp2(text)
        print("Entities", [(ent.text, ent.label_) for ent in doc.ents])
        print("Tokens", [(t.text, t.ent_type_, t.ent_iob) for t in doc])
    


def load_data(input_file, splitRatio=0.8):
    """Load data from the dataset."""
    # Partition off part of the train data for evaluation
    train_data = defaultdict(list)
    entries = 0
    with open(input_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            entries += 1
            train_data[row[1].strip()].append((row[0], row[1].strip()))


    train_texts = []
    train_cats = []
    test_texts = []
    test_cats = []
    largestLabel = ""
    largestLabelCount = -1
    print(f'Loaded {entries} entries.')
    for data_type in train_data:
        labeled_data = train_data[data_type]
        if len(labeled_data) > largestLabelCount:
            largestLabel = data_type
            largestLabelCount = len(labeled_data)
        random.shuffle(labeled_data)
        texts, labels = zip(*labeled_data)
        cats = []
        for label in labels:
            cat = {}
            for main_label in train_data:
                cat[main_label.upper()] = bool(label.strip() == main_label)
            cats.append(cat)
        if len(texts) < 6:
            test_texts.extend(texts[:1])
            test_cats.extend(cats[:1])
            train_texts.extend(texts[1:])
            train_cats.extend(cats[1:])
        else:
            split = int(len(texts) * splitRatio)
            train_texts.extend(texts[:split])
            train_cats.extend(cats[:split])
            test_texts.extend(texts[split:])
            test_cats.extend(cats[split:])

    zippedTrain = list(zip(train_texts, train_cats))
    random.shuffle(zippedTrain)
    zippedTest = list(zip(test_texts, test_cats))
    random.shuffle(zippedTest)
    
    return zip(*zippedTrain), zip(*zippedTest), largestLabel, train_data.keys()


def evaluate(tokenizer, textcat, texts, cats, first_label):
    docs = (tokenizer(text) for text in texts)
    tp = 0.0  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if label != first_label:
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.0
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.0
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"textcat_p": precision, "textcat_r": recall, "textcat_f": f_score}


if __name__ == "__main__":
    plac.call(main)