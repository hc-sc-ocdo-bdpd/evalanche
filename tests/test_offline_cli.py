from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _offline_environment(tmp_path: Path) -> dict[str, str]:
    guard_dir = tmp_path / "network_guard"
    guard_dir.mkdir(exist_ok=True)
    (guard_dir / "sitecustomize.py").write_text(
        """
import socket


def _blocked(*args, **kwargs):
    raise RuntimeError("network access attempted by an offline command")


class _OfflineSocket(socket.socket):
    def connect(self, *args, **kwargs):
        return _blocked(*args, **kwargs)

    def connect_ex(self, *args, **kwargs):
        return _blocked(*args, **kwargs)


socket.socket = _OfflineSocket
socket.create_connection = _blocked
""".lstrip(),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(guard_dir), str(ROOT))
    )
    return environment


def _run_offline(
    arguments: list[str],
    *,
    tmp_path: Path,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, *arguments],
        cwd=ROOT,
        env=_offline_environment(tmp_path),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_cli_import_does_not_initialize_litellm(tmp_path: Path) -> None:
    _run_offline(
        [
            "-c",
            (
                "import sys; import evalanche.cli; "
                "assert 'litellm' not in sys.modules"
            ),
        ],
        tmp_path=tmp_path,
    )


def test_repository_checks_run_with_network_blocked(tmp_path: Path) -> None:
    registry = _run_offline(
        [
            "-m",
            "evalanche.cli",
            "registry-validate",
            "--root",
            str(ROOT),
        ],
        tmp_path=tmp_path,
    )
    assert "Status: VALID" in registry.stdout

    evidence = _run_offline(
        [
            "-m",
            "evalanche.cli",
            "evidence-status",
            "--root",
            str(ROOT),
            "--check",
        ],
        tmp_path=tmp_path,
    )
    assert "No network or model calls were made." in evidence.stdout


def test_task_initialization_runs_with_network_blocked(
    tmp_path: Path,
) -> None:
    result = _run_offline(
        [
            "-m",
            "evalanche.cli",
            "init-task",
            "--root",
            str(tmp_path),
            "--task-id",
            "offline_classification",
            "--description",
            "Verify offline task initialization.",
            "--family",
            "classification",
            "--language",
            "en",
            "--unacceptable-error",
            "Returning the wrong label.",
            "--model",
            "offline_model=test/offline-model",
            "--sample-input",
            "Example input",
            "--sample-expected-output",
            "routine",
            "--non-interactive",
        ],
        tmp_path=tmp_path,
    )

    assert "No provider calls were made." in result.stdout
    assert (
        tmp_path
        / "local_tasks"
        / "offline_classification"
        / "task.yaml"
    ).is_file()
