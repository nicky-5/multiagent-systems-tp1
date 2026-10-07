"""Show what the sandbox allows and blocks.

Run with:  uv run python scripts/sandbox_demo.py
Every attempt is harmless: the network probe targets a closed local port, and the
file probes only read.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `services` can be imported when run as a script

from services.sandbox import Limits, run_python

TABLE = next((ROOT / "data" / "tables").glob("*.parquet"), None)

PROBES = {
    "pandas on a provided table": """
import pandas as pd
df = pd.read_parquet("table.parquet")
result = f"{len(df)} rows, columns {list(df.columns)[:3]}"
""",
    "read the project's .env": f"""
result = open({str(ROOT / ".env")!r}).read()[:20]
""",
    "read ~/.ssh": """
import os
result = os.listdir(os.path.expanduser("~/.ssh"))
""",
    "everything in / (its whole world)": """
import os
result = " ".join(sorted(os.listdir("/")))
""",
    "see API keys in the environment": """
import os
result = [k for k in os.environ if "KEY" in k] or "no keys"
""",
    "open a network connection": """
import socket
socket.create_connection(("127.0.0.1", 9), timeout=2)
result = "connected"
""",
    "AF_ALG socket (Copy Fail entry point)": """
import socket
socket.socket(38, socket.SOCK_SEQPACKET)
result = "created"
""",
    "create a user namespace": """
import os
os.unshare(os.CLONE_NEWUSER)
result = "created"
""",
    "allocate 2 GB": """
block = bytearray(2 * 2**30)
result = "allocated"
""",
    "run forever": """
while True:
    pass
""",
}

WRAPPER = """
import json
try:
{body}
    print(json.dumps({{"outcome": "ALLOWED", "detail": str(result)[:70]}}))
except BaseException as e:
    print(json.dumps({{"outcome": "blocked", "detail": f"{{type(e).__name__}}: {{e}}"[:70]}}))
"""


def main() -> None:
    files = {"table.parquet": TABLE} if TABLE else {}
    limits = Limits(timeout_s=3, memory_mb=1024)
    isolation: list[str] = []
    for name, body in PROBES.items():
        code = WRAPPER.format(body="\n".join("    " + line for line in body.strip().splitlines()))
        r = run_python(code, files, limits=limits)
        isolation = r.isolation
        if r.result:
            print(f"{name:40} {r.result['outcome']:8} {r.result['detail']}")
        else:
            print(f"{name:40} {'blocked':8} {(r.error or '').splitlines()[0][:70]}")
    print("\nIsolation layers:", ", ".join(isolation))


if __name__ == "__main__":
    main()
