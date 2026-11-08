# ForestLab

From-scratch binary CART forests with exact integer Gini split ordering, full bootstrap-count witnesses and an editable browser that distinguishes all-tree votes from out-of-bag votes. No estimator library, model API, private data or account is needed to run it.

## A strong control stays

The validation-selected 31-tree forest makes **3 errors on 270 test feature groups**. The independently validation-selected 5-nearest-neighbor control makes **0/270**, and the selected single tree makes **8/270**. All candidates and the stronger control remain visible. Zero errors on this one historical split are not zero future risk.

This is an educational benchmark on the original UCI numeric image descriptors. It does not authenticate banknotes, identify currencies or establish independent specimen performance. Class labels stay `0` and `1`: the source does not explicitly map them to physical genuine/forged status. Vote fractions, leaf fractions and nearest-neighbor fractions are not calibrated probabilities or confidence intervals.

## New in v0.2: joint permutation audit

[Open the joint permutation inspector](https://midya8.github.io/forestlab/permutation.html). Freeze the selected forest and its stronger 5NN control; move each of the **15 nonempty feature blocks** using the same **16 fixed row bijections**. All 240 cases and **129,600 model-row prediction counts** stay visible, including negative error changes. Follow a donor/recipient row, inspect all repeats on a shared scale and download the full witness.

This reuses the validation set that already selected the models, not a fresh test set. Within-block values move together; recipient labels stay put. A three-row exact control shows how singleton masking can leave observed support while joint masking does not. Joint changes are not additive or isolated causal interactions. Shuffle ranges are not confidence intervals, vote fractions are not calibrated confidence, and no feature-removal refit or reselection is performed. Original training, models, splits and test scores are unchanged. See [PERMUTATION.md](PERMUTATION.md).

```bash
python -m forestlab permutation
python -m forestlab permutation --fresh --output /tmp/permutation.json
python -m forestlab permutation --input /tmp/permutation.json --audit
python -m forestlab permutation --export /tmp/permutation-web
```

## Run offline

```bash
python -m pip install -e .
python -m forestlab report
python -m forestlab audit
python -m forestlab fit --output /tmp/forestlab-fresh.json
python -m forestlab audit --report /tmp/forestlab-fresh.json
python -m forestlab witness --group 750 --tree 6
printf '%s\n' '{"features":[[0,0,0,0]]}' > /tmp/features.json
python -m forestlab predict --input /tmp/features.json
python -m forestlab export --output /tmp/forestlab-web
python -m unittest discover -s tests -v
node --test tests/*.cjs
```

Python 3.10+ and NumPy are the only runtime requirements. Node 22+ and Chrome are needed for the independent scalar/browser tests. Set `CHROME_BIN` if Chrome is not named `google-chrome`. Open `docs/index.html` directly for the offline inspector. No network fonts or model downloads.

## What is implemented

- Binary weighted-Gini CART. Candidate costs are compared using unbounded integer cross products, not approximate Gini equality. Complete feature/cut ties, pure nodes, nonpositive gain, depth and minimum-leaf stops are explicit.
- Lower-observed-value cuts with `<=` left and `>` right, not midpoint thresholds. This convention matters between training values. Greedy node splits are not globally optimal trees.
- Eight fixed 31-tree bootstrap forests, with random feature subsets drawn per node. Paired configurations share bootstrap draws through a separate RNG stream. Full in-bag counts, all 7,102 tree nodes, leaf counts and stopping choices are serialized.
- OOB predictions use only trees with zero bootstrap draws of that exact training group. Counts and eligible tree indices are retained. Zero eligible trees means no prediction. Edited queries, validation groups and test groups have no OOB identity.
- Four single-tree controls, four trained standardized logistic controls, 1NN/5NN and training class-frequency controls. Logistic Newton fits include monotone line-search histories and independently checked final gradients.
- Exact original data bytes and complete original-row membership for every exact feature group. Exact copies do not cross partitions.
- An independent scalar-JavaScript reader with all votes and actual paths, editable inputs, nearest-neighbor IDs, membership-aware OOB display, and copy/download JSON witnesses.

## Frozen protocol

The original file has 1,372 rows, four numeric features and binary labels. It contains 1,348 distinct four-feature vectors with 24 extra exact-copy records and no conflicting labels. Exact copies are collapsed to one fitting/scoring unit. A NumPy `default_rng(20261108)` permutation partitions those groups into 808 train, 270 validation and 270 test groups. This is not chronology or independently identified specimens.

Forests use depth 4 or unrestricted, minimum leaf 2 or 8, and 2 or 4 candidate features per node, with 31 trees and a fixed bootstrap seed. Each family's model selection is validation errors, then validation Brier score, then ID. All selections are frozen before test features/labels are read. No refit on validation/test is performed. All 19 fixed candidates remain in the report and browser.

Nearest-neighbor standardization uses train means/scales only. Logistic regularization values are 0.001, 0.01, 0.1 and 1; the intercept is unpenalized. The source provides no specimen identifier. Near-duplicates or shared image/specimen origins may remain across exact-feature groups. Consequently these scores cannot establish deployment performance.

## Checks

**31 Python tests and nine Node/real-Chrome tests** cover:

- 60 tiny independent Fraction split oracles, independently enumerated again with JavaScript BigInt mask counts;
- fresh reproduction of every forest, single tree and fitted control, full bootstrap/source/node audits, training-only preprocessing and final logistic stationarity;
- test-label poisoning that leaves all fitted models, validation scores and selections unchanged;
- rejection of source/split, bootstrap-count, split-gain, metric, model-configuration and selection mutations;
- 10,784 native/scalar-JavaScript forest evaluations, 8,088 control scores, 6,464 OOB eligibility lists and 992 complete path witnesses;
- actual Chrome form states for every model and partition, all 992 paths, physical-keyboard feature edits that drop membership, invalid-input clearing/recovery, clipboard success/failure, JSON Blob download and 320/390/1280px layouts.

The installed-wheel check imports all eight modules including the root outside the checkout, runs all seven commands, re-fits and audits every candidate, and reproduces the original and permutation web exports. The added diagnostic has all-case native/scalar checks, strict CLI controls, exact tiny permutation oracles and 1,230 actual browser states; see PERMUTATION.md. Floating numerical comparisons use finite bounds; original source, source grouping, discrete votes, paths and checked archive exports have exact checks. Regenerated control-score references allow only finite numerical agreement across platforms. See [METHOD.md](METHOD.md) for formulas, conventions, evidence and limits.

## Source and license

Original data: Lohweg, V. (2012), *Banknote Authentication*, UCI Machine Learning Repository, DOI `10.24432/C55P57`. [Original catalog](https://archive.ics.uci.edu/dataset/267/banknote+authentication), [original file](https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode). Original bytes are retained in a checked base64 transport file, SHA-256 `d0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9`; exact-copy grouping and split are this project's transformations. No source images were used. Dataset license is separate from MIT code. Attribution is not endorsement.

Established CART, bootstrap aggregation and random-forest methods are attributed to their sources in METHOD.md. This is an original educational implementation, not a new learning algorithm or an official UCI authentication tool.

Packaging: the full report is stored as gzip/base64 with a SHA-256 of its decompressed JSON. Python and the browser verify that digest before reading it. The browser uses native DecompressionStream and Web Crypto; Node22+/current Chrome are tested. This packaging is lossless and does not change a fit, source value, split or score. The export forwards the exact checked archive rather than depending on platform-specific compression.
