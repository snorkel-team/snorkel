# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
## [0.9.7]
### [Breaking Changes]
### [Added]
### [Changed]
### [Deprecated]
### [Removed]

## [0.9.6] - 2020-08-08
### [Added]

* PR #1572: Allow specification of `memoize_key` in preprocessors
* PR #1597: Improved error messages for MultitaskClassifier
* PR #1592: Improved LabelModel documentation

### [Contributors]

Thanks to @Wirg, @TrigonaMinima, and @dchichkov for recent contributions!

## [0.9.5] - 2020-04-06
### [Breaking Changes]

* PR #1565: Upgrade pytorch requirement to support torch>=1.2.0

## [0.9.4] - 2020-04-05
### [Breaking Changes]

* PR #1535: Refactor baseline model imports in snorkel.labeling.model
    * Now, from `from snorkel.labeling import MajorityLabelVoter, LabelModel`
    can be expressed `from snorkel.labeling.model import MajorityLabelVoter, LabelModel`

### [Added]

* PR #1533: Allow option to save optimizer state in Trainer
* PR #1523: Add GPU option for spaCy preprocessor

### [Changed]

* PR #1540: Fix squeeze bug in `to_int_label_array` function
* PR #1520: Fix error bucket documentation in snorkel.analysis.error_analysis

### [Contributors]

Thanks to @rjurney, @ptrcklv for recent contributions!

## [0.9.3] - 2019-11-11

### [Changed]

* PR #1502: Faster symmetry breaking in LabelModel using Munkres algorithm


## [0.9.2] - 2019-10-22

### [Breaking Changes]

* PR #1481: removed fault tolerant mode for labeling functions

### [Added]

* PR #1481: fault tolerant mode for appliers

### [Changed]

* PR #1450, 1467: ignore abstains in scoring, except coverage
* PR #1463: serialize all attributes of label model
* PR #1466: fix label model GPU training option
* PR #1477, #1492: pin dependency versions

### [Removed]

* PR #1454: `set_seed` utility removed

### [Contributors]

Thanks to @HiromuHota, @ferhatelmas, and @garaud for their recent contributions!


## [0.9.1] - 2019-09-05

### [Breaking Changes]

* PR #1453: `SlicingClassifier` renamed to `SliceAwareClassifier`

### [Added]

* PR #1451: add heuristic for breaking symmetry in multiple label model optima case
* PR #1442: integration test for `MultitaskClassifier`

### [Changed]

* PR #1444: fix label model weight clamping behavior
* PR #1445: fix JSON log writer
* PR #1447: fix correct/incorrect count bug in `LFAnalysis`
* PR #1428, 1449: catch invalid label model inputs
* PR #1441: make inputs to `Scorer.score` optional


## [0.9.0] - 2019-08-15
Version 0.9.0 is a complete redesign of the Snorkel library.
There's too much added, changed, and removed to list in this entry.
For more information on the release,
[check out the project homepage](https://snorkel.org).
From here forward, we'll keep a detailed changelog.
