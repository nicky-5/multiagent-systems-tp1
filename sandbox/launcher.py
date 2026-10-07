"""Runs inside the sandbox, before the untrusted code and in the same process.

It loads the seccomp filter and lowers the resource limits, which the kernel then
enforces for the rest of the process's life (neither can be undone), and only then
executes main.py. services/sandbox.py starts it as:

    python -I launcher.py '<config as JSON>'

It runs with the sandbox's interpreter and packages (sandbox/.venv), so it must not
import anything from the app: the app's code and packages aren't visible in here.
"""

import errno
import io
import json
import os
import site
import socket
import sys
import traceback
from typing import TypedDict, cast


class Config(TypedDict):
    site_packages: str  # sandbox/.venv's packages (pandas, ...)
    rlimits: dict[str, int]  # e.g. {"RLIMIT_AS": 1073741824}; empty where unsupported
    seccomp: bool
    denied_syscalls: list[str]


def load_seccomp_filter(denied_syscalls: list[str]) -> None:
    import pyseccomp as sc  # only in the sandbox environment

    f = sc.SyscallFilter(sc.ALLOW)
    # Sockets only of family AF_UNIX: no network, and no kernel interfaces reached
    # through other families (e.g. AF_ALG, used by the "Copy Fail" exploit)
    no_family = sc.ERRNO(errno.EAFNOSUPPORT)
    f.add_rule(no_family, "socket", sc.Arg(0, sc.NE, socket.AF_UNIX))
    f.add_rule(no_family, "socketpair", sc.Arg(0, sc.NE, socket.AF_UNIX))
    deny = sc.ERRNO(errno.EPERM)
    f.add_rule(deny, "clone", sc.Arg(0, sc.MASKED_EQ, os.CLONE_NEWUSER, os.CLONE_NEWUSER))
    # clone3's flags live in memory a filter can't inspect; with ENOSYS the C library
    # falls back to clone, which the rule above does check.
    f.add_rule(sc.ERRNO(errno.ENOSYS), "clone3")
    for name in denied_syscalls:
        try:
            f.add_rule(deny, name)
        except OSError:
            pass  # a system call that doesn't exist on this CPU architecture
    f.load()


def apply_limits(rlimits: dict[str, int]) -> None:
    if not rlimits:
        return
    import resource  # not available on Windows, where rlimits is empty

    for name, value in rlimits.items():
        resource.setrlimit(getattr(resource, name), (value, value))  # soft = hard


def main() -> None:
    config = cast(Config, json.loads(sys.argv[1]))
    site.addsitedir(config["site_packages"])

    if config["seccomp"]:
        load_seccomp_filter(config["denied_syscalls"])
    # Limits last, so our own setup (pyseccomp needs ctypes, which needs memory) isn't
    # constrained by them; the code that follows is.
    apply_limits(config["rlimits"])

    # stdout/stderr are files here, so they're TextIOWrappers (declared as TextIO)
    cast(io.TextIOWrapper, sys.stdout).reconfigure(encoding="utf-8")
    cast(io.TextIOWrapper, sys.stderr).reconfigure(encoding="utf-8")
    sys.argv = ["main.py"]
    with open("main.py", encoding="utf-8") as fh:
        code = compile(fh.read(), "main.py", "exec")
    try:
        # A fresh namespace: the code sees none of the launcher's variables
        exec(code, {"__name__": "__main__"})  # noqa: S102  (running the code is the point)
    except SystemExit:
        raise  # sys.exit() in the code keeps its exit code
    except BaseException as error:  # noqa: BLE001  (any error the code raises is reported)
        # Print the traceback from main.py onwards, without this launcher's frames: the
        # error is shown to the LLM to fix its code, and our frames would only confuse it
        traceback.print_exception(type(error), error, error.__traceback__.tb_next if error.__traceback__ else None)
        sys.exit(1)


if __name__ == "__main__":
    main()
