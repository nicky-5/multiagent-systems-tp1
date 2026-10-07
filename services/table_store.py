"""Local store for table data: one Parquet file per table, plus its record as JSON.

Pinecone holds each table's metadata and embedding; the data itself lives here,
keyed by the same table_id, so the analyst can load exactly the tables it selects.
"""

import os
import re
from pathlib import Path

import pandas as pd

from agents.ingestion.schemas import TableRecord
from services.config import get_settings

# table_ids come back from Pinecone at query time and become file names, so only
# accept the shape make_table_id produces (no "/" or "..").
_TABLE_ID = re.compile(r"[0-9a-f]{16}-p\d+-t\d+")


class TableStore:
    def __init__(self, root: Path):
        self.root = root

    def save(self, record: TableRecord, df: pd.DataFrame) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        # Parquet keeps the dtypes (Int64, Float64, string), unlike CSV
        _atomic_write(self._path(record.table_id, ".parquet"), df.to_parquet(index=False))
        _atomic_write(self._path(record.table_id, ".json"), record.model_dump_json(indent=2).encode())

    def exists(self, table_id: str) -> bool:
        return self.data_path(table_id).exists()

    def data_path(self, table_id: str) -> Path:
        """The table's Parquet file."""
        return self._path(table_id, ".parquet")

    def load_df(self, table_id: str) -> pd.DataFrame:
        return pd.read_parquet(self._path(table_id, ".parquet"))

    def load_record(self, table_id: str) -> TableRecord:
        return TableRecord.model_validate_json(self._path(table_id, ".json").read_text())

    def list_records(self) -> list[TableRecord]:
        if not self.root.exists():
            return []
        return [TableRecord.model_validate_json(p.read_text()) for p in sorted(self.root.glob("*.json"))]

    def _path(self, table_id: str, suffix: str) -> Path:
        if not _TABLE_ID.fullmatch(table_id):
            raise ValueError(f"Invalid table_id: {table_id!r}")
        return self.root / f"{table_id}{suffix}"


def _atomic_write(path: Path, data: bytes) -> None:
    """Write to a temporary file, then rename it into place. A rename is atomic, so a
    reader (the analyst, during a re-ingestion) never sees a half-written file."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def table_store() -> TableStore:
    return TableStore(get_settings().data_dir / "tables")
