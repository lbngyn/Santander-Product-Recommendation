"""Santander test scoring built on generic ``predict``, not model internals."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

import pandas as pd

from src.inference.predict import load_artifact, predict


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
    for product, directory in artifact_directories.items():
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
    ranked = score_frame.apply(lambda row: " ".join(row.nlargest(top_k).index), axis=1)
    recommendations = pd.DataFrame({customer_id_column: prepared_input[customer_id_column].to_numpy(), "added_products": ranked.to_numpy()})
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
