"""Tests that the sandbox runs ordinary code and blocks the things it must block.

Every escape attempt is harmless even if the sandbox fails: the memory test
allocates at most 512 MB, the network test only talks to a server this test starts on
127.0.0.1, file tests use a canary in pytest's temporary folder, and process tests
start at most a few dozen short-lived `sleep`s and clean up after themselves.
"""

import os
import socket
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest

from services.sandbox import (
    Limits,
    available_packages,
    bubblewrap_available,
    run_python,
    seccomp_available,
)

# 1024 MB of address space: pandas and pyarrow alone reserve about 600 MB of it
LIMITS = Limits(timeout_s=10, memory_mb=1024, max_processes=16, max_file_mb=1)

needs_bubblewrap = pytest.mark.skipif(not bubblewrap_available(), reason="bubblewrap not available")
needs_seccomp = pytest.mark.skipif(not seccomp_available(), reason="libseccomp not available")
linux_only = pytest.mark.skipif(sys.platform != "linux", reason="Linux-specific check")

# Tests marked with this run twice: inside bubblewrap and in the reduced mode used
# where bubblewrap isn't available. Both must handle ordinary code correctly.
both_modes = pytest.mark.parametrize(
    "bubblewrap",
    [pytest.param(True, marks=needs_bubblewrap, id="bubblewrap"), pytest.param(False, id="reduced")],
)


def run(code: str, bubblewrap: bool = True, limits: Limits = LIMITS, files: dict[str, Path] | None = None):
    return run_python(code, files or {}, limits=limits, use_bubblewrap=bubblewrap)


# --- ordinary code -----------------------------------------------------------------


@both_modes
def test_runs_pandas_on_a_provided_table(tmp_path: Path, bubblewrap: bool):
    table = tmp_path / "t.parquet"
    pd.DataFrame({"Provincia": ["Salta", "Jujuy"], "Ventas": [1.5, 2.5]}).to_parquet(table)
    code = (
        "import json, pandas as pd\n"
        "df = pd.read_parquet('t.parquet')\n"
        "print('working...')\n"
        "print(json.dumps({'total': float(df['Ventas'].sum())}))\n"
    )
    result = run(code, bubblewrap, files={"t.parquet": table})
    assert result.ok, result.error
    assert result.result == {"total": 4.0}
    assert result.stdout.startswith("working...")  # earlier lines are kept for the UI


@both_modes
def test_exception_is_reported_with_its_traceback(bubblewrap: bool):
    result = run("x = 1\nraise ValueError('column not found')\n", bubblewrap)
    assert not result.ok and result.result is None
    assert result.exit_code == 1
    assert result.error is not None
    assert 'File "main.py", line 2' in result.error  # the LLM can see where it failed
    assert "ValueError: column not found" in result.error


@pytest.mark.parametrize(
    ("code", "fragment"),
    [
        ("pass", "printed nothing"),
        ("print('the answer is 42')", "isn't valid JSON"),
        ("print('[1, 2]')", "must be a JSON object, not list"),
    ],
)
def test_output_must_end_with_a_json_object(code: str, fragment: str):
    result = run(code)
    assert result.error is not None and fragment in result.error


def test_isolation_layers_are_reported():
    result = run("print('{}')")
    assert "timeout" in result.isolation and "clean environment" in result.isolation
    if bubblewrap_available():
        assert any("bubblewrap" in layer for layer in result.isolation)


def test_rejects_file_names_with_paths(tmp_path: Path):
    with pytest.raises(ValueError, match="Invalid sandbox file name"):
        run_python("pass", {"../escape.parquet": tmp_path / "x"}, limits=LIMITS)


# --- packages --------------------------------------------------------------------


def test_available_packages_are_the_declared_ones():
    packages = available_packages()
    assert set(packages) == {"numpy", "pandas", "pyarrow"}  # not the launcher's pyseccomp
    assert all(version[0].isdigit() for version in packages.values())


@both_modes
def test_only_the_sandbox_packages_can_be_imported(bubblewrap: bool):
    # The app's packages (LangChain, OpenAI, Pinecone, ...) aren't installed in the
    # sandbox environment, so generated code can't import them
    code = """
import importlib.util, json
names = ["pandas", "numpy", "pyarrow", "openai", "langchain", "pinecone", "pymupdf", "streamlit", "httpx"]
print(json.dumps({name: importlib.util.find_spec(name) is not None for name in names}))
"""
    result = run(code, bubblewrap)
    assert result.ok, result.error
    assert result.result == {
        "pandas": True, "numpy": True, "pyarrow": True,
        "openai": False, "langchain": False, "pinecone": False,
        "pymupdf": False, "streamlit": False, "httpx": False,
    }


# --- secrets and files -----------------------------------------------------------


@both_modes
def test_environment_variables_are_not_inherited(monkeypatch: pytest.MonkeyPatch, bubblewrap: bool):
    monkeypatch.setenv("SANDBOX_TEST_SECRET", "canary-value")
    result = run("import json, os\nprint(json.dumps({'env': dict(os.environ)}))", bubblewrap)
    assert result.ok, result.error
    assert result.result is not None
    env = result.result["env"]
    assert "SANDBOX_TEST_SECRET" not in env
    assert not any(key.endswith("_API_KEY") for key in env)


@needs_bubblewrap
def test_cannot_read_files_outside_the_sandbox(tmp_path: Path):
    canary = tmp_path / "outside" / "secret.txt"
    canary.parent.mkdir()
    canary.write_text("canary")
    code = (
        "import json\n"
        f"try:\n    open({str(canary)!r}).read(); found = True\n"
        "except OSError:\n    found = False\n"
        "print(json.dumps({'found': found}))\n"
    )
    result = run(code)
    assert result.result == {"found": False}


@needs_bubblewrap
def test_cannot_write_files_outside_the_sandbox(tmp_path: Path):
    target = tmp_path / "written-from-sandbox.txt"
    code = (
        "import json\n"
        f"try:\n    open({str(target)!r}, 'w').write('x'); wrote = True\n"
        "except OSError:\n    wrote = False\n"
        "print(json.dumps({'wrote': wrote}))\n"
    )
    result = run(code)
    assert result.result == {"wrote": False}
    assert not target.exists()


@needs_bubblewrap
def test_project_files_are_not_visible():
    project = Path(__file__).resolve().parent.parent
    code = f"import json, os\nprint(json.dumps({{'exists': os.path.exists({str(project / 'pyproject.toml')!r})}}))"
    assert run(code).result == {"exists": False}


# --- network and kernel interfaces -------------------------------------------------


@pytest.fixture
def local_server() -> Iterator[tuple[int, list[bool]]]:
    """A TCP server on 127.0.0.1 that records whether anyone connected to it."""
    server = socket.create_server(("127.0.0.1", 0))
    server.settimeout(0.2)
    connected: list[bool] = []
    stop = threading.Event()

    def accept() -> None:
        while not stop.is_set():
            try:
                conn, _ = server.accept()
                connected.append(True)
                conn.close()
            except TimeoutError:
                pass

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    yield server.getsockname()[1], connected
    stop.set()
    thread.join()
    server.close()


@pytest.mark.skipif(
    not (bubblewrap_available() or seccomp_available()), reason="needs bubblewrap or seccomp"
)
def test_no_network(local_server: tuple[int, list[bool]]):
    port, connected = local_server
    code = (
        "import json, socket\n"
        f"try:\n    socket.create_connection(('127.0.0.1', {port}), timeout=2); ok = True\n"
        "except OSError as e:\n    ok = False\n"
        "print(json.dumps({'connected': ok}))\n"
    )
    result = run(code)
    assert result.result == {"connected": False}
    assert connected == []  # the server saw nothing


@needs_seccomp
@pytest.mark.parametrize("bubblewrap", [pytest.param(True, marks=needs_bubblewrap), False])
def test_only_unix_sockets_can_be_created(bubblewrap: bool):
    # AF_ALG is the entry point of "Copy Fail", AF_RXRPC of "Dirty Frag". Creating a
    # socket only asks the kernel for one; nothing is sent anywhere.
    code = """
import json, socket
families = {"AF_ALG": 38, "AF_RXRPC": 33, "AF_NETLINK": 16, "AF_INET": 2, "AF_INET6": 10, "AF_PACKET": 17}
blocked = {}
for name, family in families.items():
    try:
        socket.socket(family, socket.SOCK_SEQPACKET if name == "AF_ALG" else socket.SOCK_DGRAM).close()
        blocked[name] = False
    except OSError:
        blocked[name] = True
a, b = socket.socketpair()  # AF_UNIX, used by some libraries internally
print(json.dumps({"blocked": blocked}))
"""
    result = run(code, bubblewrap)
    assert result.ok, result.error
    assert result.result is not None
    assert all(result.result["blocked"].values()), result.result


@pytest.mark.skipif(
    not (bubblewrap_available() or seccomp_available()), reason="needs bubblewrap or seccomp"
)
@linux_only
def test_cannot_create_user_namespaces():
    code = (
        "import json, os\n"
        "try:\n    os.unshare(os.CLONE_NEWUSER); ok = True\n"
        "except OSError:\n    ok = False\n"
        "print(json.dumps({'created': ok}))\n"
    )
    assert run(code).result == {"created": False}


# --- resources ---------------------------------------------------------------------


@linux_only
@both_modes
def test_memory_is_limited(bubblewrap: bool):
    # 512 MB against a 384 MB limit: harmless even if the limit didn't apply. The limit
    # counts address space, and a bare Python process already reserves ~260 MB of it.
    code = (
        "import json\n"
        "try:\n    block = bytearray(512 * 2**20); ok = True\n"
        "except MemoryError:\n    ok = False\n"
        "print(json.dumps({'allocated': ok}))\n"
    )
    result = run(code, bubblewrap, limits=Limits(timeout_s=10, memory_mb=384))
    assert result.result == {"allocated": False}


@both_modes
def test_infinite_loop_is_stopped(bubblewrap: bool):
    started = time.monotonic()
    result = run("while True:\n    pass\n", bubblewrap, limits=Limits(timeout_s=2))
    assert result.timed_out and result.exit_code is None
    assert result.error is not None and "time limit" in result.error
    assert time.monotonic() - started < 8


def _processes_running(marker: str) -> list[int]:
    """PIDs of processes whose command line is exactly `sleep <marker>`."""
    pids: list[int] = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                cmdline = (entry / "cmdline").read_bytes().split(b"\0")
            except OSError:
                continue
            if cmdline[-2:] == [marker.encode(), b""] and cmdline[0].endswith(b"sleep"):
                pids.append(int(entry.name))
    return pids


@linux_only
@both_modes
def test_timeout_also_kills_processes_the_code_started(bubblewrap: bool):
    marker = str(900_000 + os.getpid() % 1000)  # sleep's argument, to find it again
    code = f"import subprocess, time\nsubprocess.Popen(['sleep', '{marker}'])\ntime.sleep(60)\n"
    try:
        result = run(code, bubblewrap, limits=Limits(timeout_s=2))
        assert result.timed_out
        time.sleep(0.5)
        assert _processes_running(marker) == []
    finally:
        for pid in _processes_running(marker):  # clean up if the sandbox failed
            os.kill(pid, 9)


@needs_bubblewrap
def test_number_of_processes_is_limited():
    # Tries to start 40 short sleeps under a limit of 16, then stops them all
    code = """
import json, subprocess
procs = []
try:
    for _ in range(40):
        procs.append(subprocess.Popen(["sleep", "5"]))
except OSError:
    pass
for p in procs:
    p.kill()
    p.wait()
print(json.dumps({"started": len(procs)}))
"""
    result = run(code)
    assert result.ok, result.error
    assert result.result is not None and result.result["started"] < 16


@linux_only
def test_output_size_is_limited():
    # 3 MB of output against a 1 MB file-size limit: writing past it fails
    result = run("import sys\nsys.stdout.write('x' * (3 * 2**20))\n")
    assert not result.ok
    assert len(result.stdout) <= LIMITS.max_output_chars + 100  # and we read back only a little
