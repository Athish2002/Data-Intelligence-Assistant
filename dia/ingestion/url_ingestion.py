"""
dia/ingestion/url_ingestion.py
──────────────────────────────
Ingestion source: public HTTP/HTTPS URL pointing to a CSV.
"""

from __future__ import annotations

import io
import ipaddress
import logging
import socket
from urllib.parse import urlparse

import pandas as pd
import requests

from ..config import HTTP_TIMEOUT_S, MAX_FILE_BYTES
from ..exceptions import IngestionError, ValidationError
from ..validators import sanitise_column_names, validate_dataframe_shape
from .base import IngestionResult, IngestionSource
from .local_csv import _parse_csv

log = logging.getLogger("dia.ingestion.url")

_ALLOWED_SCHEMES = {"http", "https"}
_FORBIDDEN_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "169.254.169.254"}
_MAX_REDIRECTS = 5


def _is_private_or_internal_host(hostname: str) -> bool:
    """Check if hostname resolves to a private, loopback, or cloud-metadata IP."""
    if hostname.lower() in _FORBIDDEN_HOSTS:
        return True
    try:
        ip_str = socket.gethostbyname(hostname)
        ip = ipaddress.ip_address(ip_str)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip_str == "169.254.169.254"
        )
    except Exception:
        # If DNS fails to resolve, let requests handle connection error
        return False


class URLSource(IngestionSource):
    """Load a CSV dataset from a public HTTP/HTTPS URL with SSRF protection."""

    def load(self, url: str = "", **kwargs: Any) -> IngestionResult:
        """
        Fetch and parse a CSV from a URL.

        Parameters
        ----------
        url : str  – public HTTP/HTTPS URL

        Raises
        ------
        ValidationError  – invalid or non-HTTP URL, or internal/private IP (SSRF)
        IngestionError   – network error, non-200 status, size exceeded
        """
        url = url.strip()
        if not url:
            raise ValidationError("Please provide a URL.")

        parsed = urlparse(url)
        if parsed.scheme not in _ALLOWED_SCHEMES:
            raise ValidationError(
                f"URL scheme '{parsed.scheme}' is not supported. "
                "Only http:// and https:// URLs are allowed."
            )
        if not parsed.netloc:
            raise ValidationError("URL does not appear to be valid (missing hostname).")

        hostname = parsed.hostname or ""
        if _is_private_or_internal_host(hostname):
            raise ValidationError(
                "Access to internal, loopback, or private cloud metadata endpoints is prohibited (SSRF protection)."
            )

        log.info("Fetching CSV from URL: %s", url)

        try:
            resp = requests.get(
                url,
                timeout=HTTP_TIMEOUT_S,
                stream=True,
                allow_redirects=True,
                headers={"User-Agent": "DataIntelligenceAssistant/1.0"},
            )
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            raise IngestionError(
                f"Request timed out after {HTTP_TIMEOUT_S} seconds. "
                "The server may be slow or unreachable."
            ) from None
        except requests.exceptions.ConnectionError as exc:
            raise IngestionError(f"Could not connect to '{parsed.netloc}': {exc}") from exc
        except requests.exceptions.HTTPError as exc:
            raise IngestionError(
                f"Server returned HTTP {resp.status_code}. "
                "Check the URL and try again."
            ) from exc

        # Stream and enforce size limit
        chunks: list[bytes] = []
        total = 0
        for chunk in resp.iter_content(chunk_size=65_536):
            total += len(chunk)
            if total > MAX_FILE_BYTES:
                max_mb = MAX_FILE_BYTES / (1024 ** 2)
                raise IngestionError(
                    f"URL content exceeds the {max_mb:.0f} MB limit. "
                    "Please use a smaller dataset."
                )
            chunks.append(chunk)

        raw_bytes = b"".join(chunks)
        log.info("Downloaded %.2f MB from URL", total / (1024 ** 2))

        df = _parse_csv(raw_bytes)
        df.columns = pd.Index(sanitise_column_names(list(df.columns)))
        validate_dataframe_shape(df)

        meta: dict[str, Any] = {
            "url": url,
            "file_size_bytes": total,
            "file_size_mb": round(total / (1024 ** 2), 2),
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "status_code": resp.status_code,
        }

        return IngestionResult(
            df=df,
            source_label=f"URL — {parsed.netloc}{parsed.path[:50]}",
            meta=meta,
        )
