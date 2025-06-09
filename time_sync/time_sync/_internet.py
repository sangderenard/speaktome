"""Fetch the current UTC time from the internet."""

from __future__ import annotations

import datetime as _dt
import json
import urllib.request

try:
    import ntplib  # type: ignore
except Exception:  # pragma: no cover - optional
    ntplib = None

# --- END HEADER ---


def fetch_internet_utc() -> _dt.datetime:
    """Return the current UTC time from an online source."""
    if ntplib is not None:
        try:
            client = ntplib.NTPClient()
            resp = client.request("pool.ntp.org", version=3, timeout=5)
            return _dt.datetime.fromtimestamp(resp.tx_time, tz=_dt.timezone.utc)
        except Exception:
            pass

    url = "https://worldtimeapi.org/api/timezone/Etc/UTC"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.load(resp)
    return _dt.datetime.fromisoformat(data["utc_datetime"]).astimezone(_dt.timezone.utc)
