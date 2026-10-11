"""Remove every table of one PDF from Pinecone and the local table store.

Run with:  uv run python scripts/remove_document.py data/sample_pdfs/<file>.pdf
The PDF itself is only read (to compute its hash), not deleted.
"""

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so `agents` and `services` can be imported when run as a script

from agents.ingestion.rules import document_prefix
from services.table_store import table_store
from services.vector_store import vector_store

pdf = Path(sys.argv[1])
prefix = document_prefix(hashlib.sha256(pdf.read_bytes()).hexdigest())
store = table_store()
ids = store.document_ids(prefix)
if not ids:
    print(f"No stored tables for {pdf.name} (prefix {prefix}).")
    sys.exit()

vector_store().delete(ids)  # Pinecone first, so no record points at deleted data
for table_id in ids:
    store.delete(table_id)
print(f"Removed {len(ids)} table(s) of {pdf.name}: {', '.join(ids)}")
