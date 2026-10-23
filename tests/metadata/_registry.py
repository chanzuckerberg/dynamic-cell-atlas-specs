"""Replay recorded registry responses at the HTTP boundary.

Registries: Cellosaurus, OLS, Ensembl and PubChem. Each file in
``tests/fixtures/registry`` holds one real response and the request
(URL plus params) that produced it. An unrecorded request raises, so a test can
never pass on an empty answer.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REGISTRY_FIXTURES = Path(__file__).parent.parent / "fixtures" / "registry"


class UnrecordedRequest(AssertionError):
    """Raised when code under test makes a request no fixture records."""


@dataclass(frozen=True)
class RecordedResponse:
    status_code: int
    body: Any

    def json(self) -> Any:
        return self.body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise AssertionError(f"recorded response has status {self.status_code}")


def _key(url: str, params: dict) -> str:
    return json.dumps([url, params], sort_keys=True)


class RegistryReplay:
    """Callable with the signature of ``_get_with_retry`` that serves fixtures."""

    def __init__(self, directory: Path = REGISTRY_FIXTURES):
        self._responses: dict[str, RecordedResponse] = {}
        for path in sorted(directory.glob("*.json")):
            record = json.loads(path.read_text())
            request = record["request"]
            self._responses[_key(request["url"], request["params"])] = RecordedResponse(
                record["status_code"], record["body"]
            )
        self.requests: list[tuple[str, dict]] = []

    def __call__(self, url: str, params: dict) -> RecordedResponse:
        self.requests.append((url, params))
        try:
            return self._responses[_key(url, params)]
        except KeyError:
            raise UnrecordedRequest(
                f"no recorded response for {url} {params}; "
                f"{len(self._responses)} recorded in {REGISTRY_FIXTURES}"
            ) from None
