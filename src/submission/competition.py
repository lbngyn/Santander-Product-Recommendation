"""Santander test scoring built on generic ``predict``, not model internals."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

import pandas as pd
import numpy as np

from src.inference.predict import load_artifact, predict


CandidateBatchScorer = Callable[[pd.DataFrame], pd.DataFrame]


def run_ranked_candidate_submission(
    prepared_input: pd.DataFrame,
    sample_submission_path: str | Path | None,
    output_path: str | Path,
    *,
    score_batch: CandidateBatchScorer,
    top_k: int = 7,
    batch_customers: int = 50_000,
    customer_id_column: str = "ncodpers",
) -> Path:
    """Write the official submission from a small model-specific scorer.

    ``score_batch`` is the only model adapter. It receives one prepared,
    model-ready customer batch and returns ``ncodpers``, ``product`` and
    numeric ``score`` candidate rows. This module owns ranking, template
    ordering, column validation and CSV output for every model strategy.
    """
    if top_k < 1 or batch_customers < 1:
        raise ValueError("top_k and batch_customers must be positive.")
    if customer_id_column not in prepared_input or prepared_input[customer_id_column].duplicated().any():
        raise ValueError("Prepared competition input must contain exactly one row per ncodpers.")
    scored_batches: list[pd.DataFrame] = []
    for start in range(0, len(prepared_input), batch_customers):
        scored = score_batch(prepared_input.iloc[start:start + batch_customers])
        required = {customer_id_column, "product", "score"}
        if missing := required.difference(scored.columns):
            raise ValueError(f"Candidate scorer is missing required columns: {sorted(missing)}")
        scored = scored.loc[:, [customer_id_column, "product", "score"]].copy()
        if scored.duplicated([customer_id_column, "product"]).any():
            raise ValueError("Candidate scorer emitted duplicate customer-product rows.")
        scored["score"] = pd.to_numeric(scored["score"], errors="raise")
        if not np.isfinite(scored["score"]).all():
            raise ValueError("Candidate scorer emitted a non-finite score.")
        scored_batches.append(scored)
    candidates = pd.concat(scored_batches, ignore_index=True) if scored_batches else pd.DataFrame(columns=[customer_id_column, "product", "score"])
    ranked = candidates.sort_values([customer_id_column, "score", "product"], ascending=[True, False, True], kind="stable")
    ranked["rank"] = ranked.groupby(customer_id_column, sort=False).cumcount().add(1)
    recommendations = (
        ranked.loc[ranked["rank"].le(top_k)]
        .groupby(customer_id_column, sort=False)["product"].agg(" ".join)
        .rename("added_products").reset_index()
    )
    if sample_submission_path is None:
        template = prepared_input.loc[:, [customer_id_column]].copy()
    else:
        template = pd.read_csv(sample_submission_path)
        if list(template.columns) != [customer_id_column, "added_products"]:
            raise ValueError("sample_submission must contain exactly ncodpers and added_products in that order.")
        if template[customer_id_column].duplicated().any():
            raise ValueError("sample_submission must contain one row per ncodpers.")
    output = template.loc[:, [customer_id_column]].merge(
        recommendations, on=customer_id_column, how="left", sort=False, validate="one_to_one",
    )
    if output[customer_id_column].isin(prepared_input[customer_id_column]).eq(False).any() or len(output) != len(prepared_input):
        raise ValueError("sample_submission customer IDs do not exactly match prepared input.")
    # Customers who own every product receive a valid empty recommendation,
    # matching the existing 24-model submission behavior.
    output["added_products"] = output["added_products"].fillna("")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    return destination


def run_competition_test_inference(
    prepared_input: pd.DataFrame,
    sample_submission_path: str | Path,
    artifact_directories: Mapping[str, str | Path],
    output_path: str | Path,
    *,
    top_k: int = 7,
    customer_id_column: str = "ncodpers",
) -> Path:
    """Score 24 products, mask t-1 ownership, and fill the official template.

    ``prepared_input`` is produced by whichever preprocessing/feature pipeline
    the artifacts require.  It must include `prev_<product>` solely for the
    competition ownership mask; each artifact schema controls model inputs.
    """
    if customer_id_column not in prepared_input:
        raise ValueError(f"Prepared input lacks {customer_id_column!r}.")
    if prepared_input[customer_id_column].duplicated().any():
        raise ValueError("Competition test input must have exactly one row per customer.")
    scores: dict[str, pd.Series] = {}
    for model_number, (product, directory) in enumerate(artifact_directories.items(), start=1):
        print(f"[Submission {model_number}/24] scoring {product}", flush=True)
        artifact, model = load_artifact(directory)
        if artifact.product != product:
            raise ValueError(f"Artifact product mismatch: key={product}, artifact={artifact.product}")
        # IDs and prev_* ownership flags are orchestration context, not model
        # inputs.  Give predict exactly the artifact's declared schema.
        scores[product] = predict(prepared_input.loc[:, artifact.schema.feature_names], artifact, model)
    if len(scores) != 24:
        raise ValueError(f"Expected 24 product artifacts, received {len(scores)}.")
    score_frame = pd.DataFrame(scores, index=prepared_input.index)
    for product in score_frame:
        ownership = "prev_" + product
        if ownership not in prepared_input:
            raise ValueError(f"Prepared input lacks ownership mask column {ownership!r}.")
        score_frame.loc[pd.to_numeric(prepared_input[ownership], errors="raise").eq(1), product] = float("-inf")
    # Do not use masked (-inf) products as padding for customers who own more
    # than 17 products; a shorter list is preferable to an invalid suggestion.
    #
    # This must stay vectorised: calling ``nlargest`` through ``apply`` once
    # per test customer turns a 24-column rank into nearly one million Python
    # operations on the competition data.
    values = score_frame.to_numpy(copy=False)
    product_array = np.asarray(score_frame.columns, dtype=object)
    ranking = np.argsort(-values, axis=1, kind="stable")[:, :top_k]
    selected_scores = np.take_along_axis(values, ranking, axis=1)
    selected_products = product_array[ranking]
    ranked = [
        " ".join(products[np.isfinite(product_scores)])
        for products, product_scores in zip(selected_products, selected_scores, strict=True)
    ]
    recommendations = pd.DataFrame(
        {customer_id_column: prepared_input[customer_id_column].to_numpy(), "added_products": ranked}
    )
    template = pd.read_csv(sample_submission_path)
    if list(template.columns) != [customer_id_column, "added_products"]:
        raise ValueError("sample_submission must contain exactly ncodpers and added_products in that order.")
    output = template[[customer_id_column]].merge(recommendations, on=customer_id_column, how="left", sort=False, validate="one_to_one")
    if output["added_products"].isna().any():
        raise ValueError("sample_submission contains customer IDs missing from prepared input.")
    destination = Path(output_path); destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    return destination


def run_competition_test_from_raw(
    test_data: object,
    prepare_input: Callable[[object], pd.DataFrame],
    sample_submission_path: str | Path,
    artifact_directories: Mapping[str, str | Path],
    output_path: str | Path,
    **kwargs: object,
) -> Path:
    """Run an injected, versioned preparation pipeline then score the test set."""
    return run_competition_test_inference(
        prepare_input(test_data), sample_submission_path, artifact_directories,
        output_path, **kwargs,
    )
