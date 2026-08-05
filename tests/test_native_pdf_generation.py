from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from evalanche.config import GenerationConfig
from evalanche.generation import (
    build_response_input,
    load_input_file_descriptors,
)


def _config(tmp_path: Path) -> GenerationConfig:
    return GenerationConfig.model_validate(
        {
            "run": {
                "name": "pdf_test",
                "input_path": tmp_path / "cases.csv",
                "output_path": tmp_path / "outputs.csv",
            },
            "candidate_models_path": tmp_path / "models.yaml",
            "prompt": {
                "system": "Extract only source facts.",
                "template": "{input}",
            },
            "generation": {"request_api": "responses"},
        }
    )


def _row(path: Path, sha256: str) -> dict[str, str]:
    return {
        "case_id": "pdf_case",
        "input": "Return the requested JSON.",
        "expected_output": "{}",
        "evaluation_type": "json",
        "input_files": json.dumps(
            [
                {
                    "path": path.name,
                    "filename": "source.pdf",
                    "media_type": "application/pdf",
                    "sha256": sha256,
                    "detail": "high",
                }
            ]
        ),
    }


def test_response_input_embeds_hash_verified_pdf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    content = b"%PDF-1.7\nsource bytes\n%%EOF\n"
    path = tmp_path / "source.pdf"
    path.write_bytes(content)
    row = _row(path, hashlib.sha256(content).hexdigest())

    response_input = build_response_input(_config(tmp_path), row)

    assert response_input[0]["role"] == "user"
    file_block = response_input[0]["content"][1]
    assert file_block["type"] == "input_file"
    assert file_block["filename"] == "source.pdf"
    assert file_block["detail"] == "high"
    encoded = file_block["file_data"].split(",", 1)[1]
    assert base64.b64decode(encoded) == content


def test_response_input_rejects_pdf_hash_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "source.pdf"
    path.write_bytes(b"changed")
    row = _row(path, "0" * 64)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_input_file_descriptors(row)


def test_response_input_rejects_paths_outside_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path.parent / "outside.pdf"
    path.write_bytes(b"outside")
    row = _row(path, hashlib.sha256(b"outside").hexdigest())
    row["input_files"] = row["input_files"].replace(
        '"outside.pdf"',
        '"../outside.pdf"',
        1,
    )

    with pytest.raises(ValueError, match="outside the repository root"):
        load_input_file_descriptors(row)


@pytest.mark.parametrize("filename", ["nested/source.pdf", "nested\\source.pdf"])
def test_response_input_rejects_non_basename_filenames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    content = b"pdf"
    path = tmp_path / "source.pdf"
    path.write_bytes(content)
    row = _row(path, hashlib.sha256(content).hexdigest())
    descriptors = json.loads(row["input_files"])
    descriptors[0]["filename"] = filename
    row["input_files"] = json.dumps(descriptors)

    with pytest.raises(ValueError, match="must be a basename"):
        load_input_file_descriptors(row)
