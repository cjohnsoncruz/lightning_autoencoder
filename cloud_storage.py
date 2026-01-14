"""
Modular cloud storage utilities for GCS and Google Drive.
Usage:
    from cloud_storage import CloudStorage

    # Initialize once per session
    storage = CloudStorage(service_account_path='service_account_key.json')

    # Save/load files
    storage.save_to_gcs('local_file.parquet', 'remote/path/file.parquet')
    storage.save_to_drive('local_file.png', 'file.png', folder_id='xxx')
    df = storage.load_parquet_from_gcs('remote/path/file.parquet')
"""

import os
import io
import re
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Union
from pathlib import Path

import pandas as pd


class CloudStorage:
    """Unified interface for GCS and Google Drive operations."""

    def __init__(self,
                 service_account_path: str = 'service_account_key.json',
                 gcs_bucket_name: str = 'colab_autoencoder_output',
                 drive_folder_id: Optional[str] = None,
                 verbose: bool = True):
        """
        Initialize cloud storage clients.

        Args:
            service_account_path: Path to service account JSON key file
            gcs_bucket_name: Default GCS bucket name
            drive_folder_id: Default Google Drive folder ID for uploads
            verbose: Print status messages
        """
        self.service_account_path = service_account_path
        self.gcs_bucket_name = gcs_bucket_name
        self.drive_folder_id = drive_folder_id
        self.verbose = verbose

        # Lazy initialization - clients created on first use
        self._gcs_client = None
        self._gcs_bucket = None
        self._drive_service = None

    # ==================== GCS Properties (Lazy Loading) ====================

    @property
    def gcs_client(self):
        """Lazily initialize GCS client."""
        if self._gcs_client is None:
            from google.cloud import storage
            self._gcs_client = storage.Client.from_service_account_json(
                self.service_account_path
            )
            if self.verbose:
                print(f"GCS client initialized")
        return self._gcs_client

    @property
    def gcs_bucket(self):
        """Lazily initialize GCS bucket."""
        if self._gcs_bucket is None:
            self._gcs_bucket = self.gcs_client.bucket(self.gcs_bucket_name)
            if self.verbose:
                print(f"Connected to bucket: gs://{self.gcs_bucket_name}")
        return self._gcs_bucket

    @property
    def drive_service(self):
        """Lazily initialize Google Drive service."""
        if self._drive_service is None:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build

            SCOPES = ['https://www.googleapis.com/auth/drive']
            creds = service_account.Credentials.from_service_account_file(
                self.service_account_path, scopes=SCOPES
            )
            self._drive_service = build('drive', 'v3', credentials=creds)
            if self.verbose:
                print("Google Drive service initialized")
        return self._drive_service

    # ==================== GCS Operations ====================

    def save_to_gcs(self, local_path: str, blob_name: str,
                    bucket_name: Optional[str] = None) -> bool:
        """
        Upload a file to GCS.

        Args:
            local_path: Path to local file
            blob_name: Destination path in bucket
            bucket_name: Override default bucket

        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(local_path):
            print(f"Warning: {local_path} does not exist, skipping upload")
            return False

        try:
            bucket = self.gcs_client.bucket(bucket_name or self.gcs_bucket_name)
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_path)

            if self.verbose:
                print(f"Uploaded: gs://{bucket.name}/{blob_name}")
            return True

        except Exception as e:
            print(f"GCS upload error: {e}")
            return False

    def load_from_gcs(self, blob_name: str, local_path: str,
                      bucket_name: Optional[str] = None) -> bool:
        """Download a file from GCS to local path."""
        try:
            bucket = self.gcs_client.bucket(bucket_name or self.gcs_bucket_name)
            blob = bucket.blob(blob_name)
            blob.download_to_filename(local_path)

            if self.verbose:
                print(f"Downloaded: gs://{bucket.name}/{blob_name} -> {local_path}")
            return True

        except Exception as e:
            print(f"GCS download error: {e}")
            return False

    def load_parquet_from_gcs(self, blob_name: str,
                               bucket_name: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load a parquet file directly from GCS into a DataFrame."""
        try:
            bucket = self.gcs_client.bucket(bucket_name or self.gcs_bucket_name)
            blob = bucket.blob(blob_name)
            df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))

            if self.verbose:
                print(f"Loaded: gs://{bucket.name}/{blob_name} ({len(df)} rows)")
            return df

        except Exception as e:
            print(f"GCS parquet load error: {e}")
            return None

    def list_blobs(self, prefix: str = '',
                   bucket_name: Optional[str] = None,
                   suffix: Optional[str] = None) -> List[Dict]:
        """
        List blobs in GCS bucket.

        Args:
            prefix: Filter by path prefix
            bucket_name: Override default bucket
            suffix: Filter by file extension (e.g., '.parquet')

        Returns:
            List of dicts with 'name', 'size_kb', 'updated' keys
        """
        bucket = self.gcs_client.bucket(bucket_name or self.gcs_bucket_name)

        blobs = []
        for blob in bucket.list_blobs(prefix=prefix):
            if suffix is None or blob.name.endswith(suffix):
                blobs.append({
                    'name': blob.name,
                    'size_kb': blob.size / 1024 if blob.size else 0,
                    'updated': blob.updated
                })

        # Sort by update time (newest first)
        blobs.sort(key=lambda x: x['updated'] or datetime.min, reverse=True)
        return blobs

    def list_folders(self, prefix: str, pattern: str = r'([^/]+)/',
                     bucket_name: Optional[str] = None) -> List[str]:
        """
        List unique folder names matching a pattern.

        Args:
            prefix: Path prefix to search
            pattern: Regex pattern to extract folder name
            bucket_name: Override default bucket

        Returns:
            List of folder paths, sorted by embedded date if present
        """
        bucket = self.gcs_client.bucket(bucket_name or self.gcs_bucket_name)

        folders = set()
        for blob in bucket.list_blobs(prefix=prefix):
            match = re.match(f'({prefix}[^/]+)/', blob.name)
            if match:
                folders.add(match.group(1))

        # Try to sort by date in folder name (DD_Mon_YYYY format)
        def parse_date(folder):
            match = re.search(r'(\d{2})_(\w{3})_(\d{4})', folder)
            if match:
                try:
                    return datetime.strptime(
                        f"{match.group(1)}_{match.group(2)}_{match.group(3)}",
                        "%d_%b_%Y"
                    )
                except ValueError:
                    pass
            return datetime.min

        return sorted(folders, key=parse_date, reverse=True)

    def load_latest_from_folder(self, folder_prefix: str,
                                 file_suffix: str = '.parquet',
                                 bucket_name: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Find the latest folder matching prefix and load all matching files.

        Args:
            folder_prefix: Prefix to search for folders (e.g., '/content/NN_param_sweep_')
            file_suffix: File extension to load
            bucket_name: Override default bucket

        Returns:
            Tuple of (combined DataFrame, folder path) or (None, None) if not found
        """
        folders = self.list_folders(folder_prefix, bucket_name=bucket_name)

        if not folders:
            print(f"No folders found matching: {folder_prefix}")
            return None, None

        latest_folder = folders[0]
        if self.verbose:
            print(f"Found {len(folders)} folder(s), loading from: {latest_folder}")

        files = self.list_blobs(prefix=latest_folder, suffix=file_suffix,
                                bucket_name=bucket_name)

        if not files:
            print(f"No {file_suffix} files in {latest_folder}")
            return None, latest_folder

        # Load and concatenate all files
        dfs = []
        bucket = self.gcs_client.bucket(bucket_name or self.gcs_bucket_name)
        for f in files:
            blob = bucket.blob(f['name'])
            df = pd.read_parquet(io.BytesIO(blob.download_as_bytes()))
            dfs.append(df)

        combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        if self.verbose:
            print(f"Loaded {len(combined)} rows from {len(dfs)} file(s)")

        return combined, latest_folder

    # ==================== Google Drive Operations ====================

    def save_to_drive(self, local_path: str, drive_filename: str,
                      folder_id: Optional[str] = None) -> Optional[str]:
        """
        Upload a file to Google Drive.

        Args:
            local_path: Path to local file
            drive_filename: Filename in Drive
            folder_id: Drive folder ID (uses default if not specified)

        Returns:
            File ID if successful, None otherwise
        """
        from googleapiclient.http import MediaFileUpload

        if not os.path.exists(local_path):
            print(f"Warning: {local_path} does not exist, skipping upload")
            return None

        try:
            folder = folder_id or self.drive_folder_id
            metadata = {'name': drive_filename}
            if folder:
                metadata['parents'] = [folder]

            media = MediaFileUpload(local_path)
            file = self.drive_service.files().create(
                body=metadata,
                media_body=media,
                fields='id',
                supportsAllDrives=True
            ).execute()

            file_id = file.get('id')
            if self.verbose:
                print(f"Uploaded to Drive: {drive_filename} (id: {file_id})")
            return file_id

        except Exception as e:
            print(f"Drive upload error: {e}")
            return None

    # ==================== Convenience Methods ====================

    def save_dataframe(self, df: pd.DataFrame, filename: str,
                       gcs_path: Optional[str] = None,
                       drive_folder_id: Optional[str] = None,
                       local_dir: str = '/content') -> Dict[str, bool]:
        """
        Save DataFrame to local, GCS, and/or Drive in one call.

        Args:
            df: DataFrame to save
            filename: Base filename (e.g., 'results.parquet')
            gcs_path: GCS destination path (None to skip)
            drive_folder_id: Drive folder ID (None to skip)
            local_dir: Local directory for temp file

        Returns:
            Dict with 'local', 'gcs', 'drive' success status
        """
        local_path = os.path.join(local_dir, filename)
        results = {'local': False, 'gcs': False, 'drive': False}

        # Save locally first
        try:
            if filename.endswith('.parquet'):
                df.to_parquet(local_path)
            elif filename.endswith('.csv'):
                df.to_csv(local_path, index=False)
            else:
                df.to_parquet(local_path)
            results['local'] = True
        except Exception as e:
            print(f"Local save error: {e}")
            return results

        # Upload to GCS
        if gcs_path:
            results['gcs'] = self.save_to_gcs(local_path, gcs_path)

        # Upload to Drive
        if drive_folder_id:
            results['drive'] = self.save_to_drive(local_path, filename, drive_folder_id) is not None

        return results


# ==================== Standalone Functions (for backwards compatibility) ====================

_default_storage: Optional[CloudStorage] = None

def get_storage(service_account_path: str = 'service_account_key.json',
                bucket_name: str = 'colab_autoencoder_output') -> CloudStorage:
    """Get or create the default CloudStorage instance."""
    global _default_storage
    if _default_storage is None:
        _default_storage = CloudStorage(
            service_account_path=service_account_path,
            gcs_bucket_name=bucket_name
        )
    return _default_storage


def save_to_gcs(local_path: str, blob_name: str) -> bool:
    """Backwards-compatible function."""
    return get_storage().save_to_gcs(local_path, blob_name)


def load_parquet_from_gcs(blob_name: str) -> Optional[pd.DataFrame]:
    """Backwards-compatible function."""
    return get_storage().load_parquet_from_gcs(blob_name)
