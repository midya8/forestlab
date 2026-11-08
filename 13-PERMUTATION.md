# Joint permutation audit

ForestLab v0.2 adds a post-fit diagnostic of the frozen selected forest and its stronger selected 5NN control. The original data, 19 fitted candidates, train/validation/test split, selections and test scores are unchanged. This diagnostic does not train, tune, remove features or choose another model.

## The validation set is reused

The same 270 validation feature groups already selected both models. Their unpermuted validation errors are 4/270 for the forest and 1/270 for 5NN. These are not the original test errors of 3/270 and 0/270. This display is a reused-validation description of already selected models, not fresh held-out performance, cross-validation or population evidence.

For every nonempty subset S of the four features and a row bijection pi:

`Z[i,j] = X[pi[i],j] if j is in S, otherwise X[i,j]`.

All features in S use the same donor row. The recipient label y[i] stays in place. The within-block joint multiset is preserved; its dependencies with the remaining features and with the recipient label can break. This is not a conditional draw given the other features, a causal intervention or an independent shuffle of each selected column.

`Delta_errors(S,pi) = sum_i 1[predict(Z[i]) != y[i]] - baseline_errors`.

Higher means more errors under this particular shuffle; lower means fewer. We also retain each run's confusion counts, error fraction and mean squared hard-vote-fraction error, called Brier in the original project. Neither model's vote fraction is calibrated probability. Changing the score changes the diagnostic. Means are over repeats, with all individual values and min/max shown. No interval, p-value, statistical significance or population uncertainty is supplied.

## Fixed protocol

- Every one of the 15 nonempty subsets appears in ascending integer bit-mask order. There is no importance sorting or tuned grouping.
- One `numpy.random.default_rng(2026110802)` generator supplies 16 successive `permutation(270)` row bijections. The 16 full maps are retained. The same maps are reused for all blocks and both models. Fixed points are permitted. Repeats do not enumerate all 270! permutations, and are not forced derangements.
- Every shuffled matrix starts from the original X, not a cumulatively modified matrix. The input is not changed.
- The fixed selected models are `forest-depth-None-leaf-2-features-2` and `nearest-5`. Forest hard-vote ties and neighbor-score ties use class zero. Standardization is still training-derived. There is no model fitting or selection here.
- Prediction counts for all 15 × 16 × 270 × 2 = 129,600 model-row evaluations are retained, with denominators 31 tree votes and 5 neighbor votes. All runs, including negative changes, remain visible.
- The report binds original validation group order, feature matrix, labels and selected model objects by SHA-256. The new module accesses validation features/labels and stored fitted models only. Poisoning train/test labels without changing those stored models leaves this diagnostic unchanged. That check is not permission to fit models without labels or evidence against original training dependence.

## Observations, not a feature-removal recommendation

The image-entropy singleton has forest error changes from -2 to +2, with mean +0.4375 over these 16 shuffles. The 5NN changes are 0 to +6, mean +3.0625. Negative repeats are not clipped and do not imply a helpful real intervention or that entropy should be removed.

For variance alone the mean changes are +85.5625 forest errors and +86.1875 5NN errors; for skewness alone +62.25 and +76.625; for both jointly +134.625 and +138.3125. The joint changes are not sums of singleton changes. These comparisons describe different corrupted matrices, not an isolated causal interaction or a conditional-importance estimator.

Permuting all four features jointly preserves every complete observed validation vector. The predictions are exactly the baseline predictions reordered by pi, but labels are not reordered. Forest error changes range from +120 to +144 (mean +133); 5NN changes from +126 to +144 (mean +134.75). On-support rows alone do not preserve their relationship with recipient labels.

The count of hybrid vectors equal to a stored validation vector is shown per run. Absence from a finite observed set is not proof of impossible or out-of-population data. Presence is not validation of causal meaning or independence.

## Exact support control

Original constructed rows are (0,0), (1,1), (2,2), all with label zero. A fixed toy rule predicts one exactly when the features differ. Baseline error is zero. In lexicographic order over all six row bijections, either singleton shuffle produces 0, 2, 2, 3, 3, 2 errors: exact mean 2/3 as an error fraction. Joint shuffling produces zero errors in all six cases. Individual hybrids can leave the diagonal support, while a joint block stays on it. This isolates a masking/support failure of this toy rule, not the cause of any UCI result. No source figure or dataset was copied for this control.

## Implementation and checks

`forestlab/permutation.py` implements group masking, row-map validation, batched routing, metrics, report binding, lossless checked archive, regeneration, audit and export. Batched routing uses the same observed lower cuts and `<=` branches as the original scalar traversal; every original row's selected-forest vote count is independently compared against that scalar traversal. The trained 5NN control keeps its original stable distance/index tie handling.

There are **31 Python tests and nine Node/real-Chrome tests** across the full project. New checks cover:

- full fresh diagnostic reproduction, all 240 block/shuffle cases, exact original fitted-object immutability, and train/test-label poisoning;
- all 15 masks and every row bijection, preservation of every selected/unselected column, full-block baseline prediction reordering, and unclipped negative repeats;
- six exhaustive three-row permutations, independent hand/Fraction controls, exact error counts and Fraction Brier references for all 480 model/run metrics;
- mutation of model binding, donor maps and prediction counts; strict Boolean/index/duplicate/nonfinite validation, identity shuffles and CLI mode conflicts;
- independent scalar JavaScript predictions for all 129,600 model-row evaluations, all confusion/error diagnostics and finite Brier agreement;
- actual Chrome states for 960 block/repeat/recipient combinations plus all 270 recipients for one block, original recipient labels, independent row inference, stale-state clearing, physical keyboard, clipboard failure/success, actual Blob download, 320/390/1280 layouts and 44px controls.

The wheel check imports all eight modules including the root outside the checkout, runs all seven commands with the original 19 fresh fits and audits, regenerates all 240 new cases, checks fresh and stored diagnostics and reproduces seven exact checked archive/loader exports. Original control-score references retain the documented finite numerical tolerance; discrete counts, group membership and paths remain exact.

```bash
python -m pip install -e .
python -m forestlab permutation
python -m forestlab permutation --fresh --output /tmp/permutation.json
python -m forestlab permutation --input /tmp/permutation.json --audit
python -m forestlab permutation --audit
python -m forestlab permutation --export /tmp/permutation-web
python -m unittest discover -s tests -v
node --test tests/*.cjs
```

`--input` requires `--audit`. Export cannot silently ignore fresh/input/audit/output options. New `permutation.html` is an offline page using the same original checked model archive plus a separately SHA-256-checked gzip/base64 diagnostic archive. It has no external scripts, fonts, model APIs or analytics. Export forwards archived bytes rather than depending on platform zlib output. Direct local-file opening is tested with current Chrome; Node22+ tests native decompression and Web Crypto.

## Sources and distinctions

1. [scikit-learn permutation importance guide](https://scikit-learn.org/stable/modules/permutation%5Fimportance.html), official documentation, read November 8, 2026. Fitted-model score-drop definition, repeat protocol and score/correlation cautions. We implement the arithmetic independently, without scikit-learn. We use error increase rather than accuracy decrease, which differs by the fixed denominator 270.
2. [scikit-learn correlated-feature example](https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html), official documentation. Demonstrates low singleton importance despite predictive accuracy. Its clustering/retraining example is not implemented or attributed to our all-subset audit.
3. [scikit-learn permutation implementation](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/inspection/_permutation_importance.py), source implementation. Read copy semantics, fitted-estimator and repeat scoring. No code was copied; our mapping explicitly starts from original rows and shares complete maps across blocks/models.
4. [Thomas Altmann, PFI, LOCO and Correlated Features](https://slds-lmu.github.io/iml_methods_limitations/pfi-correlated.html), LMU educational book chapter. Dependence-breaking can create unrealistic observations; correlation can change apparent reliance in either direction. LOCO refits a model and answers another question; this diagnostic does not.
5. [Cross-validated permutation feature importance considering correlation between features](https://pmc.ncbi.nlm.nih.gov/articles/PMC10989554/), primary paper. Its cross-validated procedure refits over folds and addresses a different estimator. We do not claim CVPFI, unbiased validation or its results/guarantees.
6. [mlxtend permutation feature importance guide](https://rasbt.github.io/mlxtend/user_guide/evaluate/feature_importance_permutation/), library documentation. Model-agnostic fitted-model permutation procedure and possible overestimation with correlated features. This does not contradict redundancy causing low singleton reliance in another model: neither direction is universal.

No new algorithm, causal importance method, population result or state-of-the-art benchmark is claimed. Original UCI attribution and CC BY 4.0 source terms remain in README.md, METHOD.md and the packaged data license.
