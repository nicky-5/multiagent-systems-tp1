from pathlib import Path

import pandas as pd
import pytest

from agents.ingestion.schemas import ColumnInfo, Source, TableRecord
from services.table_store import TableStore

TABLE_ID = "0123456789abcdef-p1-t0"


def make_record() -> TableRecord:
    return TableRecord(
        table_id=TABLE_ID,
        name="Ventas",
        description="Una tabla. Dos oraciones.",
        columns=[ColumnInfo(name="Provincia", description="x", unit="text")],
        row_header_column="Provincia",
        source=Source(filename="informe.pdf", pages=[1]),
        row_headers=["Salta"],
        n_rows=2,
        preview="",
    )


def test_round_trip_keeps_nullable_dtypes(tmp_path: Path):
    df = pd.DataFrame({
        "Provincia": pd.Series(["Salta", None], dtype="string"),
        "Unidades": pd.Series([1, None], dtype="Int64"),
        "Monto": pd.Series([1.5, None], dtype="Float64"),
    })
    store = TableStore(tmp_path)
    store.save(make_record(), df)

    pd.testing.assert_frame_equal(store.load_df(TABLE_ID), df)
    assert store.load_record(TABLE_ID) == make_record()
    assert [r.table_id for r in store.list_records()] == [TABLE_ID]


@pytest.mark.parametrize("bad_id", ["../secrets", "x/y", "0123456789abcdef-p1-t0.json"])
def test_rejects_ids_that_are_not_table_ids(tmp_path: Path, bad_id: str):
    with pytest.raises(ValueError, match="Invalid table_id"):
        TableStore(tmp_path).load_df(bad_id)


def test_document_ids_and_delete(tmp_path: Path):
    store = TableStore(tmp_path)
    df = pd.DataFrame({"Provincia": pd.Series(["Salta"], dtype="string")})
    other = make_record().model_copy(update={"table_id": "fedcba9876543210-p2-t0"})
    store.save(make_record(), df)
    store.save(other, df)

    assert store.document_ids("0123456789abcdef") == [TABLE_ID]
    store.delete(TABLE_ID)
    assert not store.exists(TABLE_ID)
    assert [r.table_id for r in store.list_records()] == [other.table_id]
