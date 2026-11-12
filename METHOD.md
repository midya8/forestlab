# ForestLab fitting and evaluation method

## Unit and split

The original UCI bytes are retained in a checked base64 transport file under CC BY 4.0. Its 1,372 records contain 1,348 unique four-feature vectors. A group keeps all original zero-based row IDs and its original label. Conflicting exact-copy labels would reject the source; none exist here. Collapse each group to one fitting/scoring observation before the fixed group permutation and 808/270/270 partition. This prevents exact feature copies from appearing in two partitions. It cannot identify near-duplicates or physical specimen overlap: the source has no specimen IDs. Grouping is not a proof of independent observations.

The four features are wavelet-image variance, skewness, `curtosis` (source spelling), and image entropy. These are precomputed historical descriptors, not raw pixels. No image classifier, image upload or currency detector is implemented. The target mapping to genuine/forged is not explicitly defined by the source, so all code and UI use labels 0 and 1.

## Exact node criterion, not exact global trees

For a node with N binary labels and N1 ones, parent Gini is

```text
G(parent) = 2 N1 (N - N1) / N².
```

A candidate split has left counts L0,L1 and right counts R0,R1, with nL=L0+L1 and nR=R0+R1. Its weighted child Gini is

```text
G(children) = (2/N) [L0 L1 / nL + R0 R1 / nR].
```

The implementation compares the bracketed cost as the integer fraction

```text
numerator   = L0 L1 nR + R0 R1 nL
 denominator = nL nR.
```

Two candidates a,b are ordered by `a.num*b.den < b.num*a.den`, using arbitrary-precision integers. No floating Gini tolerance chooses a split. The parent comparison is also cross-multiplied exactly. Positive gain is required. Displayed gain is a numerical conversion of this exact count criterion, not a separate certificate.

Candidates use each distinct observed feature value except the maximum. Equal values stay on the same side. The threshold is the *lower observed value*, not the midpoint to the next value. `x <= cut` goes left and `x > cut` goes right. These cuts induce the same observed training partitions as midpoints, but differ for inputs between adjacent observed values. The boundary convention is deliberate and tested with the next representable float after a cut.

Sort features ascending and cuts ascending. Replace the current best only for strictly lower rational cost. Thus ties choose the lowest feature then lowest cut. A leaf predicts its majority class, with ties going to class 0. Stop at a pure node, insufficient samples for two minimum-size leaves, the specified depth, or no strictly improving split. Search is greedy at each node; it does not find a globally optimal tree, causal explanation or robustness certificate. Float feature comparisons use the represented finite input values; the exact arithmetic claim is about integer-count impurity ordering.

## Bootstrap and feature randomization

Each forest has 31 trees. For each tree, sample 808 integer indices uniformly with replacement from the unique training groups. Store the entire length-808 count vector. Bootstrap sample size is exactly 808, but the number of distinct groups is smaller. At each nontrivial node, sample the fixed feature budget without replacement, sort those features, and run the exact greedy node search.

Bootstrap and per-tree node-feature RNG streams are separate. The fixed depth/min-leaf/feature-budget configurations share their bootstrap draws. Their nodes and feature draws need not align after their paths diverge. Performance differences between configurations are descriptive comparisons, not isolated causal estimates of one mechanism.

Every tree casts one hard vote. The forest score is `number of class-1 votes / 31`; its hard class is 1 only if that fraction is greater than 0.5. Leaf sample fractions are exposed as counts, not used as soft tree votes. Do not confuse this hard-vote implementation with averaging leaf probabilities. Neither fraction is established as calibrated.

## OOB membership

For stored training group i, eligible trees are exactly those with `inbag[t][i] == 0`. Recompute the group prediction in each eligible tree, then aggregate only those hard votes. A tie goes to class 0. If the eligible set is empty, return null prediction and null fraction, never the full-forest vote. Every group happens to have at least one eligible tree in these fixed fits; the zero-tree branch is tested independently.

The Python API binds the ordered training feature matrix to the model with a SHA-256 digest. The full report audit binds training labels, original source rows, partitions and bootstrap counts too. A row ordering change rejects OOB computation. The browser uses the frozen report's original training group ID. Editing even one field clears original-group membership; matching a feature vector later does not silently restore it. Choose or restore the original sample to restore membership. Validation/test groups are held-out queries, not OOB observations.

OOB error is a training-sample diagnostic from a collection of partially trained trees, not the error of the exact same full forest used on the final holdout. OOB is not used to choose hyperparameters. Selecting by OOB would make a subsequently quoted OOB score selection-dependent. These data also lack independent specimen provenance, so neither OOB nor random group holdout establishes operational performance.

## Controls and selection

Eight forest configurations are the Cartesian product of depth {4, unrestricted}, minimum leaf {2,8}, feature budget {2,4}. Every forest uses 31 trees and fixed seed 20261108. Four same-training single trees use all features and the depth/min-leaf grid. Seven additional controls are standardized ridge-logistic fits at lambda {0.001,0.01,0.1,1}, standardized 1NN/5NN, and the training class frequency.

Standardization is fitted on training groups only. Logistic minimizes mean binary log loss plus `(lambda/2)||slopes||²`; the intercept is not penalized. Newton solves the full Hessian, uses Armijo backtracking and records objective/gradient histories. Tests independently central-difference the objective and recompute the final gradient. Nearest-neighbor ties use training-array index order. The class-frequency control is not constrained to be weak.

Each family selects by validation error count, then validation Brier score, then lexicographic ID. Test data are accessed only after all selections. No model is refit on validation/test. All candidates and all controls stay visible. The selected forest has 3/270 test errors, the selected single tree has 8/270, and the selected 5NN has 0/270. These are this split's descriptive results. The stronger control's success is not hidden and is not evidence of deployment safety.

## Independent checks

- Sixty small integer designs are solved by complete-mask enumeration with Python Fraction. A separate JavaScript BigInt implementation enumerates those masks again.
- A real recursively grown tree is checked at every node against the independent Fraction enumerator.
- All 248 bootstrap vectors are replayed; every represented node's source membership, counts, split witness, searched features, stopping limits and numerical gain are checked against reconstructed bootstraps. This auditor shares the production split routine for real full-node optimum rechecks; independence comes from the Fraction/BigInt oracles, not from calling the same splitter twice.
- All fits are regenerated, and flipped test labels leave all fits, validation metrics and selected IDs unchanged.
- Original source/group/split, bootstrap, split-gain, test-metric, model-configuration and selection mutations are rejected.
- All 10,784 native/scalar-JavaScript forest evaluations, 8,088 control scores, 6,464 OOB records and 992 full paths are compared. Chrome actually drives every native path reference and every model/partition state, plus keyboard edits, invalid recovery, clipboard/download and mobile behavior.

Exact source matching and discrete outputs are separate from finite floating-point comparisons. Logistic and score comparisons allow small finite platform differences; they reject material drift, NaN and changed discrete metadata. None of these tests is a proof of generalization or unseen-site/browser compatibility.

## Sources

- Original UCI catalog, dataset information and license: https://archive.ics.uci.edu/dataset/267/banknote+authentication . Lohweg (2012), DOI 10.24432/C55P57. Original file: https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt . Dataset CC BY 4.0: https://creativecommons.org/licenses/by/4.0/legalcode . No separate rights are inferred for papers, source images or third-party mirrors.
- Breiman (2001), *Random Forests*, primary paper: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf . Tree randomization, hard votes and OOB context. The infinite-forest convergence discussion is not a finite-data deployment guarantee.
- Breiman/Cutler, *Manual On Setting Up, Using, And Understanding Random Forests*, v3.1: https://www.stat.berkeley.edu/~breiman/Using_random_forests_V3.1.pdf . Bootstrap/per-node feature construction and diagnostics.
- Wasserman, CMU, *Partitions and Trees*: https://www.stat.cmu.edu/~larry/=sml/forests.pdf . Independent academic overview. Weighted Gini is explicitly derived above rather than relying on an abbreviated prose formula.
- scikit-learn decision-tree documentation: https://scikit-learn.org/stable/modules/tree.html . Independent implementation context, not a runtime dependency or copied estimator.
- Ishwaran (2015), *The Effect of Splitting on Random Forests*: https://pmc.ncbi.nlm.nih.gov/articles/PMC5599182/ . Split-selection preferences are another reason not to present split counts as causal feature importance.
- Angelopoulos/Bates, *A Gentle Introduction to Conformal Prediction*, revised 2022: https://arxiv.org/html/2107.07511 . Marginal coverage requires appropriate calibration/exchangeability assumptions. This project does not implement or claim conformal coverage; vote disagreement alone does not supply it.

Original educational implementation of established methods. No invented algorithm credit, runtime model call or source endorsement.

Packaging: the full report is stored as gzip/base64 with a SHA-256 of its decompressed JSON. Python and the browser verify that digest before reading it. The browser uses native DecompressionStream and Web Crypto; Node22+/current Chrome are tested. This packaging is lossless and does not change a fit, source value, split or score. The export forwards the exact checked archive rather than depending on platform-specific compression.

## Frozen geometry extension

Version 0.3 adds leaf-collision and jointly-OOB pair diagnostics without altering the original experiment. The all-tree PSD identity and exact negative OOB control are described in [PROXIMITY.md](PROXIMITY.md). These are model-geometry descriptions, not new predictions or statistical validation.
