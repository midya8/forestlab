# Frozen forest leaf-collision geometry

This v0.3 extension changes no original training data, exact-copy groups, split, bootstrap counts, fitted trees, model selection or scores. It reads the frozen partitions, features and indexed bootstrap multiplicities. It does not read labels during the new computation, but the partitions were learned with labels. This is supervised model geometry, not unsupervised discovery or semantic ground truth.

## The all-tree kernel

For tree t and leaf l, define Z[i,(t,l)] = 1 if source/query row i reaches that leaf, and 0 otherwise. Leaf identifiers are local to their tree. For T trees,

    C[i,j] = sum_t [leaf_t(i) = leaf_t(j)]
    P = C/T = Z Z^T / T.

Thus for every real vector c,

    c^T P c = sum_(t,l) (sum_(i in leaf(t,l)) c_i)^2 / T >= 0.

This is a general identity for a fixed finite ensemble of partitions, not a conclusion drawn from tested eigenvalues. PSD includes singular matrices. The native and browser panels evaluate the identity for integer coefficients, using Python integers and JavaScript BigInt respectively. Exact integers in witnesses are serialized as decimal strings, avoiding binary64 truncation. Native code also evaluates the independent full integer quadratic expression and checks equality with the leaf-bucket sum of squares. No eigensolver, tolerance-based positivity verdict or spectral repair is used.

Each row has T active columns, so ||Z_i/sqrt(T)||^2 = 1 and squared Euclidean feature-space distance is 2(1-P[i,j]). Its displayed square root is approximate. Distinct raw rows can have identical leaf codes and zero distance. The induced raw-input distance is a pseudometric, not a one-to-one identity or proof of useful semantic neighborhoods. A relabeling of leaves within a tree leaves the geometry unchanged. Accidentally pooling equal leaf IDs across different trees would be wrong.

## Jointly-OOB comparison is different

For an indexed training group i, M[i,t] is true exactly when its stored bootstrap multiplicity in tree t is zero. Indexed OOB requires the original training order. Define

    D[i,j] = sum_t M[i,t] M[j,t]
    N[i,j] = sum_t M[i,t] M[j,t] [leaf_t(i) = leaf_t(j)].

When both identities are known and D>0, P_oob=N/D. A known pair with D=0 has an integer zero denominator and an undefined value (JSON null), not zero similarity. A validation/test or feature-only row has no indexed training identity in this API: all its OOB counts/denominators are null. It is not reclassified as OOB merely because it was absent from the training split or happens to equal a training vector. A diagonal is also undefined when a training group has no OOB tree; no diagonal override is applied.

N and D have Gram forms, but their elementwise quotient is not generally a Gram matrix. Different pairwise denominators do not provide the all-tree identity. We do not assert that every OOB matrix is indefinite, nor that more trees universally remove or preserve indefiniteness. No eigenvalue clipping, shifting, projection or missing-entry fill is performed.

### Exact original combinatorial control

Use row-by-tree leaf codes and OOB eligibility:

    leaf codes:       OOB mask:
    [0, 1, 0]         [true,  false, true ]
    [0, 0, 1]         [true,  true,  false]
    [1, 0, 2]         [false, true,  true ]

Every off-diagonal pair is eligible in exactly one tree; each diagonal has two eligible trees. Therefore

    P_oob = [[1,1,0], [1,1,1], [0,1,1]].

For c=(1,-2,1), exact c^T P_oob c=-2. The corresponding all-tree quadratic form is 10/3. Also d=1-P_oob has d01=d12=0 and d02=1, violating the triangle inequality. This table is a constructed partition/eligibility control, not a fitted historical-data result or claim of realizability from three bootstrap training rows. Separate controls retain an undefined off-diagonal pair and two coincident leaf-code rows.

## Interface and consistency binding

- `leaf_codes(model, features)` traverses the frozen trees and returns row-by-tree terminal IDs.
- `geometry(codes, oob=None)` returns integer collision counts, fractions and squared distances. Optional masks retain undefined OOB pairs.
- `indexed(model, data, groups)` accepts 1-64 distinct source IDs, validates the exact training feature digest and all bootstrap multiplicities, then computes indexed masks. It includes full raw features, training-local indices and all leaf codes. It never changes the model or source data.
- `pair(model, left, right)` accepts two full-width finite numeric feature rows. It deliberately returns no OOB fields or source identity.
- `quadratic(codes, integer_coefficients)` provides exact direct and leaf-bucket witnesses. It does not assert that externally supplied leaf codes came from a particular model.

The browser accepts the published training-order binding, compares all training IDs and all training feature values, and checks the recorded training digest against the model. The original report is already loaded through its SHA-256-checked gzip archive. Numeric bindings are consistency checks, not authorship, authenticity or a security boundary for arbitrary untrusted models. The browser uses independent scalar tree routing and exact BigInt integer accumulation, not a learned JavaScript model or an imported estimator.

The matrix displays the first 24 training groups in stored order, not a label-sorted, performance-selected or representative whole-data sample. All eight original forests and all three matrix modes are available. Native reference matrices include those 24 groups and a separate 12-group train/validation/test mix. The source-pair UI allows the full 1,348 groups, including a group compared with itself. Feature edits need explicit Apply, and then have no OOB identity. Invalid edits retain the previous applied witness with a visible refusal. Changing a source selector or forest restores the selected source pair. No plot silently switches models underneath an applied witness.

## Reproduce

```bash
python -m pip install .
python -m unittest discover -s tests -v
node --test tests/*.cjs
python -m forestlab proximity --input examples/proximity.json
python -m forestlab proximity-controls
python -m forestlab proximity-export --output /tmp/forest-proximity-web
```

Example input is exactly `{"groups":[0,1,2]}`. For arbitrary feature-only comparison use `{"features":[[0,0,0,0],[1,2,3,4]]}`. Optional `--model-id` selects any existing frozen candidate; the default remains the original selected model. Duplicate JSON keys, mixed input shapes, Boolean features, duplicate/out-of-range group IDs and nonfinite values are refused. The 64-group CLI/API bound is a deliberate output-size limit, not a statistical requirement.

```python
from forestlab.experiment import saved
from forestlab.proximity import indexed, quadratic
r = saved()
m = next(c['model'] for c in r['candidates'] if c['id'] == r['selected_id'])
g = indexed(m, r['data'], r['data']['split']['train'][:6])
q = quadratic(g['leaf_codes'], [1, -2, 1, 0, 0, 0])
assert q['numerator'] == q['sum_of_squares']
assert int(q['numerator']) >= 0
```

## Independent checks and scope

39 Python and 12 Node/Chrome tests include the original suites. New checks exhaust all 512 three-row/three-tree binary partition tables against an independent one-hot matrix product; use exact Fraction quadratic comparisons; check 6,272 all-tree/source-OOB pair entries with independent traversal and set intersections across all eight forests; preserve state, labels and source bytes; test strict refusals and coefficients beyond binary64 integer range. The source matrices provide 1,152 mixed-partition pair entries and 4,608 training-heatmap pair entries, with 8,928 terminal coordinates. Python and scalar JavaScript compare every exported field, including exact Gram witnesses.

Actual Chrome checks all 1,152 mixed-partition pair UI states and all 13,824 heatmap cells across eight models and three views. Heatmaps are compared to the complete native matrices before all fill colors are checked. Invalid retention, unbound feature-only identity, source self-pairs, model context, keyboard input, clipboard failure/success, actual Blob content, 320/390/1280 layouts and 44-pixel controls are tested. Module/UI agreement alone is not an independent partition oracle. Ten-module/ten-command installed-wheel reproduction retains all 19 original fresh fits, original audits, full permutation reproduction and exact exports, then exercises the new geometry commands outside the checkout.

No proximity classifier, manifold embedding, imputation method, new held-out score, medical/financial authentication, uncertainty interval, causal influence or calibrated probability is added. A descriptive visualization and a PSD identity do not establish useful neighborhoods. OOB exclusion is not an independent specimen-validation dataset. Source labels and the stronger original 5NN control remain on the original pages.

## Sources and a documented discrepancy

Standard forest proximity and Gram identities are established methods, not a new algorithm claim. The code, combinatorial controls and inspector are original implementations; no source's code is copied and no extra estimator dependency is added.

1. TensorFlow Decision Forests, [Proximities with random forests](https://www.tensorflow.org/decision_forests/tutorials/proximities_colab), official tutorial, accessed November 8, 2026. Explicit same-leaf sum divided by T, supervised partitions and possible downstream uses. We use the definition, not the tutorial's model, Adult dataset, t-SNE results or an empirical quality claim.
2. Andy Liaw and Matthew Wiener, [Classification and Regression by randomForest](https://journal.r-project.org/articles/RN-2002-022/RN-2002-022.pdf), R News 2/3 (2002), primary package article. The proximity section calls it the fraction of trees sharing a terminal node; the OOB algorithm section supports indexed bootstrap membership.
3. Leo Breiman, [Using Random Forests V3.1](https://www.stat.berkeley.edu/~breiman/Using_random_forests_V3.1.pdf), original manual, accessed November 8, 2026. Historical context and terminal-node collisions. Its prose says divide by *twice* the number of trees and then override the diagonal. This differs from the explicit T-normalized definition in sources 1, 2 and 4. We document rather than silently merge that discrepancy: this project uses C/T and its literal Gram diagonal, not that legacy prose normalization/override.
4. Erwan Scornet, [Random forests and kernel methods](https://arxiv.org/html/1502.03836), 2015 paper, Section 2.2 and Proposition 1. Connection function as the average of same-cell indicators, PSD discussion, and distinction between KeRF and ordinary forest predictions. We do not replace hard-vote predictions with a KeRF predictor.
5. [Proximum introduction](https://cran.r-project.org/web/packages/Proximum/vignettes/Proximum-intro.html), independent package documentation, accessed November 8, 2026. Leaf-incidence Gram identity, pairwise OOB quotient and undefined denominators. Its empirical indefiniteness and debiasing statements are not adopted as universal facts; our exact table only disproves a universal PSD guarantee. No repair or package code is used.
6. [randomForest reference manual](https://cran.r-project.org/web/packages/randomForest/refman/randomForest.html), official package reference, accessed November 8, 2026. Distinguishes `proximity`, `oob.prox`, and `keep.inbag`. The package's documented in-bag flag is not this project's complete multiplicity archive, so their storage is not claimed identical.
