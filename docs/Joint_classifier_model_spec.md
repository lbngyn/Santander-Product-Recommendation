Implement a NEW modeling pipeline for the Santander Product Recommendation project.

IMPORTANT:
- Do NOT replace or modify the existing 24-model pipeline.
- Keep the current data pipeline, preprocessing pipeline, feature engineering,
  train/validation split, evaluation and submission logic reusable.
- Add a second modeling strategy/pipeline that trains ONE shared LightGBM model.

The experiment should allow us to compare:

Pipeline A — existing:
    existing data
        -> preprocessing
        -> existing features
        -> 24 independent LightGBM binary classifiers
        -> 24 probabilities
        -> ranking
        -> Top 7

Pipeline B — NEW:
    SAME existing data
        -> SAME preprocessing
        -> SAME existing features
        -> candidate expansion + product_id
        -> ONE LightGBM binary classifier
        -> acquisition probability for each candidate product
        -> ranking by probability
        -> Top 7


============================================================
1. DO NOT CHANGE THE UPSTREAM PIPELINE
============================================================

Reuse the current implementation for:

    raw data
        ->
    preprocessing
        ->
    feature engineering
        ->
    temporal train / validation / test preparation

Do not redesign those components.

The new pipeline should branch only at the modeling-input/modeling stage.

Conceptually:

                         existing processed features
                                   |
                  +----------------+----------------+
                  |                                 |
                  v                                 v
          EXISTING PIPELINE                  NEW PIPELINE
                  |                                 |
        24 product models                candidate expansion
                                                    |
                                              + product_id
                                                    |
                                                    v
                                           ONE LightGBM
                                                    |
                                                    v
                                        probability per product


============================================================
2. NEW MODEL FORMULATION
============================================================

The existing pipeline effectively learns:

    model_p(X_customer) -> P(y_p = 1)

for each of the 24 products independently.

The new pipeline should learn:

    model(X_customer, product_id)
        -> P(y = 1 | customer, target_product)

There is only ONE binary target:

    y ∈ {0, 1}

Interpretation:

    y = 1:
        customer acquires the candidate product at target month

    y = 0:
        customer does not acquire the candidate product at target month


The model configuration for this experiment is:

    LightGBM binary classifier
    n_estimators = 500

Train exactly ONE estimator.


============================================================
3. ADD `product_id` ONLY IN THE NEW PIPELINE
============================================================

Do NOT add `product_id` globally to the existing feature pipeline.

`product_id` is a model-specific feature belonging to the new unified
modeling pipeline.

Use the existing canonical PRODUCT_COLUMNS list.

Create a deterministic mapping:

    PRODUCT_ID_MAP = {
        product_name: index
        for index, product_name in enumerate(PRODUCT_COLUMNS)
    }

Example:

    ind_ahor_fin_ult1 -> 0
    ind_aval_fin_ult1 -> 1
    ...
    ind_recibo_ult1 -> 23


Also create the inverse mapping:

    PRODUCT_NAME_BY_ID = {
        index: product_name
        ...
    }


Store `product_id` using a compact integer dtype such as:

    uint8

because only 24 values exist.


IMPORTANT:

The integer represents a CATEGORY, not an ordinal value.

Therefore:

    product_id = 10

does NOT mean the product is numerically greater than:

    product_id = 2


When training LightGBM, explicitly register:

    product_id

as a categorical feature.


============================================================
4. WHY THE NEW PIPELINE NEEDS LONG-FORMAT MODEL INPUT
============================================================

Do NOT change the stored upstream dataset.

The existing processed/feature dataset remains in its current format.

Only create a LONG-FORMAT MODEL VIEW specifically for the new pipeline.


Suppose the existing processed sample is conceptually:

    customer = 123
    month = t

    features:
        age
        renta
        segmento
        ...
        historical/product features

    targets:
        product_A = 0
        product_B = 1
        product_C = 0
        ...


The existing pipeline can send this to 24 models.

The new pipeline instead expands the sample into candidate-product rows:

    customer | same existing features | product_id | target
    --------------------------------------------------------
    123      | X                      | A          | 0
    123      | X                      | B          | 1
    123      | X                      | C          | 0
    ...


Thus:

    one customer-month sample

becomes:

    multiple (customer-month, target-product) samples.


============================================================
5. CANDIDATE PRODUCTS
============================================================

Only create rows for products the customer does NOT own before the
prediction month.

For product p:

    if product_state[p, t-1] == 1:
        exclude p

    if product_state[p, t-1] == 0:
        create candidate row


Therefore:

    previous state | current state | unified target
    ------------------------------------------------
          0        |       0       |       0
          0        |       1       |       1
          1        |       0       |    excluded
          1        |       1       |    excluded


The new model therefore learns:

    among products the customer does not currently own,
    how likely is each product to be acquired?


============================================================
6. TRAIN DATA FOR THE NEW PIPELINE
============================================================

Use the SAME train split produced by the existing pipeline.

Do not perform another split after converting to long format.

Correct order:

    existing dataset
          |
          v
    existing temporal split
          |
          +--------------------+
          |                    |
       train                 validation
          |                    |
          v                    v
    expand candidates     expand candidates
          |                    |
          v                    v
    train_long            validation_long


This is important because all candidate rows belonging to the same
customer-month must remain in the same temporal partition.


For every training customer-month:

1. Read the existing model features.

2. Read product ownership at t-1.

3. Find every unowned product.

4. For every unowned product p:

       duplicate/reference the existing feature vector
       add product_id = ID(p)

5. Obtain that product's existing binary target.

6. Store it in ONE target column:

       target


Example:

Original training sample:

    customer = 1001

    previous ownership:
        A = 1
        B = 0
        C = 0
        D = 0

    existing targets:
        A = ...
        B = 0
        C = 1
        D = 0


New model rows:

    customer | existing X | product_id | target
    ------------------------------------------------
    1001     | X          | B          | 0
    1001     | X          | C          | 1
    1001     | X          | D          | 0


These three rows are fed to the SAME LightGBM model.


============================================================
7. TARGET DOES NOT CHANGE
============================================================

Do NOT introduce a new target definition.

Reuse the binary acquisition target already produced by the existing
pipeline.

The only structural change is:

OLD:

    target_product_A
    target_product_B
    ...
    target_product_24

used by 24 different models.


NEW:

    product_id     target
    ---------------------
       A             0
       B             1
       C             0


In other words, the product dimension moves from:

    "which model / which target column"

into:

    "a feature identifying the candidate product"


The target itself remains binary:

    0 / 1


============================================================
8. UNIFIED LIGHTGBM
============================================================

Train ONE:

    LGBMClassifier

using:

    X =
        all existing features
        +
        product_id

    y =
        target


Configuration for this experiment:

    objective = binary
    n_estimators = 500


Do not add:

    class_weight
    scale_pos_weight
    focal loss
    negative sampling

yet.

This experiment should isolate the effect of changing:

    24 independent models

to:

    1 shared model + product_id


============================================================
9. VALIDATION / INFERENCE
============================================================

At prediction time, start from the SAME processed customer features.

For each customer:

    existing features = X

Find candidate products:

    C = {p | customer does not own p at t-1}


Then construct:

    X + product_id=p1
    X + product_id=p2
    X + product_id=p3
    ...


Example:

Customer 123 does not own:

    B
    C
    E


Model input:

    customer | existing X | product_id
    -----------------------------------
    123      | X          | B
    123      | X          | C
    123      | X          | E


Run:

    unified_model.predict_proba(...)


Result:

    customer | product_id | P(y=1)
    --------------------------------
    123      | B          | 0.023
    123      | C          | 0.081
    123      | E          | 0.015


These probabilities are the scores used for recommendation.


============================================================
10. RANK BY P(y=1)
============================================================

Do NOT convert probability into binary predictions.

Do NOT use:

    predict()

or:

    probability >= 0.5


Use:

    predict_proba(...)[:, 1]


For each customer:

    candidates
        ->
    P(y=1) for every candidate
        ->
    sort descending
        ->
    take first 7


Example:

    C = 0.081
    B = 0.023
    E = 0.015

Ranking:

    C > B > E


The recommendation score is therefore:

    score(customer, product)
        =
    unified_model.predict_proba(
        customer features,
        product_id
    )[1]


============================================================
11. TEST DATA
============================================================

Reuse the existing test preprocessing and feature pipeline.

Do not create a separate preprocessing implementation for the unified
model.

After obtaining the existing model-ready test features:

    test features
          |
          v
    candidate expansion
          |
          v
    + product_id
          |
          v
    unified model
          |
          v
    P(y=1)
          |
          v
    rank per customer
          |
          v
    Top 7
          |
          v
    product_id -> original product name
          |
          v
    existing submission format


Test does not contain target, so candidate expansion should support:

    with_target=True

for train/validation and conceptually:

    with_target=False

for test/inference.

Choose the exact API according to the existing architecture.


============================================================
12. IMPLEMENT THE TRANSFORMATION AS PART OF THE NEW MODEL PIPELINE
============================================================

Do not modify the generic feature-engineering output just to accommodate
this model.

Prefer something conceptually similar to:

    existing features
          |
          v
    UnifiedProductDatasetBuilder
          |
          +---- train -> long X + y
          |
          +---- validation -> long X + y
          |
          +---- inference -> long X + metadata


The exact class/function name and location should follow the current
repository architecture.

The transformation should retain metadata needed to reconstruct predictions:

    ncodpers
    target_month
    product_id


`ncodpers` and target month are metadata/group identifiers unless they
are already legitimate model features. Do not accidentally introduce
customer ID into the model just because it is needed to reconstruct
predictions.


============================================================
13. MEMORY
============================================================

Do not change the upstream dataset into a permanently stored long-format
dataset.

Keep:

    existing processed/features dataset

as the canonical representation.

Generate/materialize the long representation only for the unified
modeling pipeline.

Because:

    1 customer-month

can generate up to approximately 24 rows.

Use the existing DuckDB/Parquet/batching infrastructure where appropriate
instead of blindly expanding the full dataset in pandas if memory usage
would become excessive.


============================================================
14. REQUIRED RESULT
============================================================

After implementation the project must contain TWO independent modeling
strategies that share the same upstream pipeline:


                 SAME DATA
                     |
                 PREPROCESS
                     |
             FEATURE ENGINEERING
                     |
          SAME TEMPORAL SPLIT
                     |
          +----------+-----------+
          |                      |
          v                      v
    EXISTING PIPELINE        NEW PIPELINE
          |                      |
     24 LightGBMs          candidate expansion
                                 |
                            + product_id
                                 |
                           1 LightGBM
                           500 trees
                                 |
                          P(y=1) per
                       customer-product
                                 |
                           rank P(y=1)
                                 |
                              Top 7


The existing 24-model pipeline must continue to work unchanged.

The new pipeline must reuse the same upstream data/preprocessing/features
and differ only in the model-specific representation and model strategy.


============================================================
15. SANITY CHECKS
============================================================

Before completing the implementation, verify:

1. Existing 24-model pipeline still runs.

2. Unified pipeline trains exactly ONE LightGBM model.

3. Unified model uses n_estimators=500.

4. `product_id` has exactly 24 stable values.

5. `product_id` is treated as categorical by LightGBM.

6. Train/validation/test all use the exact same product ID mapping.

7. Long-format train target remains binary 0/1.

8. Every unified candidate corresponds to a product not owned at t-1.

9. Existing customer features are unchanged.

10. No new feature engineering is introduced.

11. Prediction uses `predict_proba()[:, 1]`, not a 0.5 threshold.

12. Recommendations are ranked by P(y=1) within each customer.

13. Product IDs are converted back to the original Santander product
    names before producing the final recommendation/submission.