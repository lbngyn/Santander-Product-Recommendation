"""Small, reusable Google Cloud Storage operations for project data."""
from __future__ import annotations

import json
import os
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.cloud.storage import Client


def credentials_from_secret() -> object | None:
    """Build credentials from a JSON string stored in ``GCP_SERVICE_ACCOUNT_JSON``.

    This avoids writing a service-account key file when running on Kaggle or Colab.
    Return ``None`` when the secret is not configured so default Google credentials can
    be used instead.
    """
    secret_value = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    if not secret_value:
        return None
    try:
        service_account_info = json.loads(secret_value)
    except json.JSONDecodeError as error:
        raise ValueError("GCP_SERVICE_ACCOUNT_JSON must contain a valid service-account JSON object.") from error
    try:
        from google.oauth2 import service_account
    except ModuleNotFoundError as error:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from error
    return service_account.Credentials.from_service_account_info(service_account_info)


def get_gcs_client(project_id: str | None = None) -> "Client":
    """Create a client from a JSON secret, credential file, or Application Default Credentials.

    Credential order: ``GCP_SERVICE_ACCOUNT_JSON`` →
    ``GOOGLE_APPLICATION_CREDENTIALS`` → default credentials.
    """
    try:
        from google.cloud import storage
    except ModuleNotFoundError as error:
        raise RuntimeError("Install dependencies with: pip install -r requirements.txt") from error
    credentials = credentials_from_secret()
    return storage.Client(project=project_id, credentials=credentials)


def normalize_prefix(prefix: str) -> str:
    """Return a GCS prefix without leading/trailing slash."""
    return prefix.strip("/")


def object_name(prefix: str, relative_path: str | Path) -> str:
    """Build a POSIX GCS object name from a prefix and a relative local path."""
    base = normalize_prefix(prefix)
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Path must be relative and safe: {relative_path}")
    return "/".join(part for part in (base, relative.as_posix()) if part)


def local_path_for_object(object_name_value: str, prefix: str, destination_dir: str | Path) -> Path:
    """Map an object under prefix to a safe path below destination_dir."""
    base = normalize_prefix(prefix)
    object_path = PurePosixPath(object_name_value)
    relative = object_path.relative_to(base) if base else object_path
    if not relative.parts or ".." in relative.parts:
        raise ValueError(f"Object is not safely below prefix: {object_name_value}")
    return Path(destination_dir).joinpath(*relative.parts)


def directory_has_files(directory: str | Path) -> bool:
    """Report whether a directory has files other than its .gitkeep placeholder."""
    location = Path(directory)
    return location.is_dir() and any(p.is_file() and p.name != ".gitkeep" for p in location.rglob("*"))


def download_object_if_missing(
    bucket_name: str,
    object_name_value: str,
    destination_path: str | Path,
    project_id: str | None = None,
    force: bool = False,
) -> Path | None:
    """Download one object only when its matching local file is absent.

    With ``force=True``, overwrite the local file with the GCS object. Returning
    ``None`` means an existing local file was deliberately reused.
    """
    destination = Path(destination_path)
    if destination.is_file() and not force:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    client = get_gcs_client(project_id)
    client.bucket(bucket_name).blob(object_name_value).download_to_filename(str(destination))
    return destination


def download_prefix_if_missing(bucket_name: str, prefix: str, destination_dir: str | Path, project_id: str | None = None, force: bool = False) -> list[Path]:
    """Download every object under prefix unless local data already exists."""
    destination = Path(destination_dir)
    if directory_has_files(destination) and not force:
        return []
    client = get_gcs_client(project_id)
    normalized = normalize_prefix(prefix)
    blob_prefix = f"{normalized}/" if normalized else ""
    downloaded: list[Path] = []
    for blob in client.list_blobs(bucket_name, prefix=blob_prefix):
        if blob.name.endswith("/"):
            continue
        local_path = local_path_for_object(blob.name, normalized, destination)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        downloaded.append(local_path)
    return downloaded


def upload_file(local_path: str | Path, bucket_name: str, object_name_value: str, project_id: str | None = None) -> str:
    """Upload one local file and return its ``gs://`` URI."""
    source = Path(local_path)
    if not source.is_file():
        raise FileNotFoundError(source)
    client = get_gcs_client(project_id)
    blob = client.bucket(bucket_name).blob(object_name_value)
    blob.upload_from_filename(str(source))
    return f"gs://{bucket_name}/{object_name_value}"


def upload_directory(source_dir: str | Path, bucket_name: str, destination_prefix: str, project_id: str | None = None) -> list[str]:
    """Upload every data file under a local directory, preserving its hierarchy."""
    source = Path(source_dir)
    if not source.is_dir():
        raise FileNotFoundError(source)
    uploaded: list[str] = []
    for file_path in source.rglob("*"):
        if file_path.is_file() and file_path.name != ".gitkeep":
            uploaded.append(upload_file(file_path, bucket_name, object_name(destination_prefix, file_path.relative_to(source)), project_id))
    return uploaded
