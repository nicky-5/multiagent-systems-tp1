"""Run untrusted Python code (written by the LLM) in a sandbox.

Python can't sandbox code inside its own process: anything the parent sets up there,
the code can undo. So the code runs in a separate process, locked down by the
kernel before the code starts. Layers, from the outside in:

1. bubblewrap (Linux namespaces). The process gets an empty filesystem to which we
   add, read-only, only the Python runtime and the sandbox's own small environment
   (sandbox/pyproject.toml: pandas, numpy, pyarrow; none of the app's packages), plus a
   writable temporary folder holding the script and the tables it may read. No network interfaces, no
   home folder, no project files, no .env. Everything is allowed in explicitly
   (a whitelist), so nothing we forgot to forbid is reachable.
2. seccomp (a system-call filter, loaded by sandbox/launcher.py before the code runs):
   sockets only of family AF_UNIX (blocking e.g. the AF_ALG and AF_RXRPC sockets
   used by the "Copy Fail" and "Dirty Frag" kernel exploits), no nested user
   namespaces, and no rarely-needed, risky system calls such as ptrace, bpf or
   io_uring. Once loaded, a filter can't be removed.
3. Resource limits (setrlimit): memory, CPU time, file size, processes, no core
   dumps. Hard limits can only be lowered, never raised again.
4. A clean environment (no API keys), Python's isolated mode (-I), and a timeout
   after which the whole sandbox is killed.

Where bubblewrap or libseccomp aren't available (macOS, Windows, some containers),
the remaining layers still apply; SandboxResult.isolation says which ones were active
so the UI can show it.

The code must print its result as a JSON object on the last line of stdout.

Setup: the sandbox environment is created with `uv sync --project sandbox`.
"""

import ctypes.util
import importlib.metadata
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from services.config import get_settings

# Denied outright. None of them is needed to compute over a DataFrame, and each is a
# common way to attack the kernel or other processes.
DENIED_SYSCALLS = [
    "ptrace", "process_vm_readv", "process_vm_writev",  # inspecting other processes
    "bpf", "perf_event_open", "userfaultfd",  # frequent kernel-exploit building blocks
    "io_uring_setup", "io_uring_enter", "io_uring_register",
    "keyctl", "add_key", "request_key",
    "unshare", "setns", "mount", "umount2", "pivot_root", "chroot",
    "open_by_handle_at", "name_to_handle_at",
    "kexec_load", "kexec_file_load", "init_module", "finit_module", "delete_module",
    "reboot", "swapon", "swapoff", "acct", "quotactl",
]

_FILE_NAME = re.compile(r"[A-Za-z0-9_.-]+")

SANDBOX_PROJECT = Path(__file__).resolve().parent.parent / "sandbox"
# Runs inside the sandbox before the untrusted code: loads the seccomp filter, lowers the
# limits, then executes main.py. Mounted read-only, so the code can't change it.
LAUNCHER = SANDBOX_PROJECT / "launcher.py"


class SandboxNotReady(RuntimeError):
    """The sandbox environment hasn't been created."""


@dataclass(frozen=True)
class SandboxEnvironment:
    """Where the sandbox's Python lives on this machine (real paths, no symlinks)."""

    interpreter: str  # the Python executable behind sandbox/.venv
    base_prefix: str  # its installation, which holds the standard library
    site_packages: str  # sandbox/.venv's packages


@dataclass(frozen=True)
class Limits:
    timeout_s: float = 30
    # Address space, not RAM in use: with pandas and pyarrow loaded, ~600 MB of it is
    # already reserved, leaving ~400 MB for the computation
    memory_mb: int = 1024
    max_processes: int = 32  # applied only inside bubblewrap (see _rlimits)
    max_file_mb: int = 50  # largest file the code may write, stdout included
    max_output_chars: int = 20_000  # how much stdout/stderr is read back

    @classmethod
    def from_settings(cls) -> Limits:
        settings = get_settings()
        return cls(timeout_s=settings.sandbox_timeout_s, memory_mb=settings.sandbox_memory_mb)


@dataclass
class SandboxResult:
    exit_code: int | None  # None if it was killed by the timeout
    stdout: str
    stderr: str
    timed_out: bool
    result: dict[str, Any] | None  # the JSON object on stdout's last line
    error: str | None  # why the run failed, worded for the LLM to fix it; None if ok
    isolation: list[str] = field(default_factory=list[str])  # layers that were active

    @property
    def ok(self) -> bool:
        return self.error is None


def run_python(
    code: str,
    files: Mapping[str, Path],
    limits: Limits | None = None,
    use_bubblewrap: bool | None = None,
) -> SandboxResult:
    """Run code with access to copies of files (name in the sandbox -> source path).

    use_bubblewrap: None uses it when available; False forces the reduced mode (tests).
    """
    limits = limits or Limits.from_settings()
    environment = sandbox_environment()
    bubblewrap = bubblewrap_available() if use_bubblewrap is None else use_bubblewrap
    seccomp = seccomp_available()

    with tempfile.TemporaryDirectory(prefix="sandbox-") as tmp:
        root = Path(tmp)
        work = root / "work"
        work.mkdir()
        for name, source in files.items():
            if not _FILE_NAME.fullmatch(name):
                raise ValueError(f"Invalid sandbox file name: {name!r}")
            shutil.copyfile(source, work / name)  # copies: the store itself stays out of reach
        (work / "main.py").write_text(code, encoding="utf-8")

        config = json.dumps({
            "site_packages": environment.site_packages,
            "rlimits": _rlimits(limits, inside_bubblewrap=bubblewrap),
            "seccomp": seccomp,
            "denied_syscalls": DENIED_SYSCALLS,
        })
        env = _environment(home="/work" if bubblewrap else str(work))
        if bubblewrap:
            sandboxed = [environment.interpreter, "-I", "/sandbox/launcher.py", config]
            command = [*_bubblewrap_args(work, environment), *sandboxed]
            cwd = None
        else:
            command = [environment.interpreter, "-I", str(LAUNCHER), config]
            cwd = work

        # stdout/stderr go to files rather than pipes, so the file-size limit caps
        # how much the code can write, and a flood of output can't fill our memory.
        stdout_path, stderr_path = root / "stdout.txt", root / "stderr.txt"
        with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
            process = subprocess.Popen(
                command, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                start_new_session=True,  # its own process group, to kill it as a whole
            )
            timed_out = False
            try:
                process.wait(timeout=limits.timeout_s)
            except subprocess.TimeoutExpired:
                timed_out = True
                # Killing bubblewrap tears down its PID namespace with every process
                # in it; without bubblewrap, kill the process group.
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

        stdout = _read_capped(stdout_path, limits.max_output_chars)
        stderr = _read_capped(stderr_path, limits.max_output_chars)

    exit_code = None if timed_out else process.returncode
    result, error = _interpret(stdout, stderr, exit_code, timed_out, limits)
    return SandboxResult(
        exit_code=exit_code, stdout=stdout, stderr=stderr, timed_out=timed_out,
        result=result, error=error, isolation=isolation_layers(bubblewrap, seccomp),
    )


def _interpret(
    stdout: str, stderr: str, exit_code: int | None, timed_out: bool, limits: Limits
) -> tuple[dict[str, Any] | None, str | None]:
    """Find the JSON result, or explain what went wrong."""
    if timed_out:
        return None, f"The program was stopped after {limits.timeout_s:g} s (time limit)."
    if exit_code != 0:
        details = stderr.strip()[-3000:] or "(no error output)"
        return None, f"The program exited with code {exit_code}.\n{details}"
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None, "The program printed nothing; print the result as one JSON object."
    try:
        parsed = json.loads(lines[-1])
    except json.JSONDecodeError as e:
        return None, f"The last line of output isn't valid JSON ({e.msg}): {lines[-1][:300]}"
    if not isinstance(parsed, dict):
        return None, f"The last line of output must be a JSON object, not {type(parsed).__name__}."
    return cast(dict[str, Any], parsed), None


def isolation_layers(bubblewrap: bool, seccomp: bool) -> list[str]:
    layers = ["separate process", "timeout", "clean environment"]
    if bubblewrap:
        layers.insert(0, "bubblewrap namespaces (no network, minimal filesystem)")
    if seccomp:
        layers.append("seccomp system-call filter")
    if sys.platform != "win32":
        layers.append("resource limits")
    return layers


# --- the pieces of the command -------------------------------------------------


def _bubblewrap_args(work: Path, environment: SandboxEnvironment) -> list[str]:
    args = [
        "bwrap",
        "--unshare-all",  # new network (empty), PID, IPC, UTS and cgroup namespaces
        "--unshare-user", "--disable-userns",  # and no nested user namespaces inside
        "--cap-drop", "ALL",
        "--die-with-parent",  # if our process dies, the sandbox dies too
        "--new-session",  # can't inject keystrokes into our terminal
        "--clearenv",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--ro-bind", "/usr", "/usr",
    ]
    for name in ("bin", "lib", "lib64", "sbin"):  # on most distributions, links into /usr
        top = Path("/") / name
        if top.is_symlink():
            args += ["--symlink", os.readlink(top), str(top)]
        elif top.is_dir():
            args += ["--ro-bind", str(top), str(top)]
    if Path("/etc/ld.so.cache").exists():  # index of shared libraries (finds libseccomp)
        args += ["--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache"]
    # The interpreter and the sandbox's packages, at their real paths (venvs are full of
    # symlinks). Only sandbox/.venv's packages: the app's environment isn't visible.
    for path in dict.fromkeys([environment.base_prefix, environment.site_packages]):
        args += ["--ro-bind", path, path]
    args += ["--ro-bind", str(LAUNCHER), "/sandbox/launcher.py"]
    args += ["--bind", str(work), "/work", "--chdir", "/work"]
    for key, value in _environment(home="/work").items():
        args += ["--setenv", key, value]
    return args


def _environment(home: str) -> dict[str, str]:
    """The whole environment the code gets: no API keys or other inherited variables."""
    return {
        "PATH": "/usr/bin:/bin",
        "HOME": home,
        "LANG": "C.UTF-8",
        "TMPDIR": "/tmp" if home == "/work" else home,
        # Keep memory use within the limit. By default OpenBLAS reserves memory per CPU
        # core at import, which fails under the memory limit.
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        # glibc gives each thread its own 64 MB memory pool, and pyarrow starts 8 I/O
        # threads: together they exceed a 512 MB limit, and pyarrow then hangs instead
        # of failing. Fewer pools and threads keep the footprint small.
        "MALLOC_ARENA_MAX": "2",
        "ARROW_IO_THREADS": "2",
        # pyarrow's default allocator (mimalloc) reserves 1 GB of address space up
        # front; the system allocator reserves only what it uses.
        "ARROW_DEFAULT_MEMORY_POOL": "system",
    }


def _rlimits(limits: Limits, inside_bubblewrap: bool) -> dict[str, int]:
    if sys.platform == "win32":
        return {}
    mb = 2**20
    rlimits = {
        "RLIMIT_AS": limits.memory_mb * mb,
        "RLIMIT_CPU": math.ceil(limits.timeout_s) + 1,  # backup for the wall-clock timeout
        "RLIMIT_FSIZE": limits.max_file_mb * mb,
        "RLIMIT_CORE": 0,  # no core dumps
    }
    if inside_bubblewrap:
        # The process count is per user namespace, so inside bubblewrap it counts only
        # the sandbox's processes. Outside, it would count all of the user's processes
        # and refuse the very first one.
        rlimits["RLIMIT_NPROC"] = limits.max_processes
    return rlimits


def _read_capped(path: Path, max_chars: int) -> str:
    with path.open("rb") as f:
        data = f.read(max_chars * 4)  # UTF-8 uses at most 4 bytes per character
    text = data.decode("utf-8", errors="replace")
    return text if len(text) <= max_chars else text[:max_chars] + "\n[... output truncated]"


# --- the sandbox environment -------------------------------------------------------


@lru_cache
def sandbox_environment() -> SandboxEnvironment:
    """Locate sandbox/.venv's interpreter and packages by asking its Python."""
    venv = SANDBOX_PROJECT / ".venv"
    python = venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    if not python.exists():
        raise SandboxNotReady(
            f"The sandbox environment is missing ({venv}). Create it with: uv sync --project sandbox"
        )
    query = "import json, sys, sysconfig; print(json.dumps([sys.executable, sys.base_prefix, sysconfig.get_paths()['purelib']]))"
    output = subprocess.run([str(python), "-I", "-c", query], capture_output=True, text=True, check=True).stdout
    interpreter, base_prefix, site_packages = (os.path.realpath(p) for p in json.loads(output))
    return SandboxEnvironment(interpreter, base_prefix, site_packages)


def available_packages() -> dict[str, str]:
    """Packages generated code may import (besides the standard library), with their
    installed versions. The code-generation prompt must list exactly these.

    They're the dependencies declared in sandbox/pyproject.toml; the launcher's own
    packages (dependency group "launcher") are left out.
    """
    project = tomllib.loads((SANDBOX_PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = [_requirement_name(r) for r in project["project"]["dependencies"]]
    installed = {
        _normalize(d.metadata["Name"]): d.version
        for d in importlib.metadata.distributions(path=[sandbox_environment().site_packages])
    }
    return {name: installed[_normalize(name)] for name in declared if _normalize(name) in installed}


def _requirement_name(requirement: str) -> str:
    """The package name of a requirement such as "pandas>=3"."""
    match = re.match(r"[A-Za-z0-9_.-]+", requirement)
    assert match is not None
    return match.group(0)


def _normalize(name: str) -> str:
    """Package names compare case-insensitively, with - _ . equivalent."""
    return re.sub(r"[-_.]+", "-", name).lower()


# --- what this machine supports --------------------------------------------------


@lru_cache
def bubblewrap_available() -> bool:
    """True if bwrap exists and may create the namespaces we need (some systems and
    containers forbid unprivileged user namespaces)."""
    if sys.platform != "linux" or shutil.which("bwrap") is None:
        return False
    probe = ["bwrap", "--unshare-all", "--unshare-user", "--disable-userns",
             "--ro-bind", "/", "/", "true"]
    try:
        return subprocess.run(probe, capture_output=True, timeout=10, check=False).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


@lru_cache
def seccomp_available() -> bool:
    """True if both the C library and its Python binding (in the sandbox environment,
    where the launcher imports it) are installed."""
    if sys.platform != "linux" or ctypes.util.find_library("seccomp") is None:
        return False
    try:
        packages = sandbox_environment().site_packages
    except SandboxNotReady:
        return False
    return any((Path(packages) / name).exists() for name in ("pyseccomp.py", "pyseccomp"))
