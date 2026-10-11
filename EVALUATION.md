# Evaluation report

## Test set

12 questions over the two sample PDFs, in [evaluation/questions.json](evaluation/questions.json),
each with its expected status and expected answer. They cover what assignment.md §9.3
requires and the edge cases the design handles:

| Requirement (§9.3) | Questions |
|---|---|
| Needs more than one table | Q05, Q06 (tables from *two different PDFs*, joined on province names) |
| No indexed table is related (`cannot_answer`) | Q10 |
| A relevant table exists but lacks a needed **period** | Q07 (`partial`), Q08 |
| … a needed **conversion** that requires outside data (exchange rate) | Q09 |
| … a needed **comparison value** (a national average no table reports) | Q11 |
| … a needed **value** (an empty cell) | Q12 |
| Other | lookups across a table continued on the next page (Q01, Q03), sum of 12 months as an allowed conversion (Q02), growth rates (Q02, Q04), negatives in parentheses (Q04), Spanish and English questions |

Expected answers were computed with pandas from the stored tables, and every source value
they use was checked to appear verbatim in the PDF text (e.g. `701,973`, `$ 597.850`,
`584.272,7`), so they don't merely repeat the extraction.

### How answers are checked

`uv run python -m evaluation.evaluate` runs each question through the real analyst (live
LLM, Pinecone and sandbox) and checks, automatically:

- the **status** matches exactly;
- each **expected number** is among the values the program computed, within 0.5% (the same
  value in an equivalent unit is accepted and noted: thousands vs units, fraction vs
  percentage);
- expected **names** appear in the answer;
- the **expected tables** were selected.

Each run is a LangSmith trace with the question's id in its metadata (`eval_id`).

## Results

Models: `gpt-4.1-mini` for every step (also set as `LLM_CODE_MODEL`), at temperature 0.
`ANALYST_MAX_CODE_ATTEMPTS=2`, so each question allows one self-correction. Sample PDFs
ingested once beforehand.

| ID | Question (abridged) | Expected | Got | Correct | Programs | Answer (abridged) |
|----|---|---|---|:-:|:-:|---|
| Q01 | Turistas en julio de 2024 | `answered` | `answered` | ✅ | 1 | 701.973 turistas |
| Q02 | Total arrivals 2024 and growth vs 2023 | `answered` | `answered` | ✅ | 1 | 8,427,032; +8.65% |
| Q03 | Provincia con mayores ventas y su participación | `answered` | `answered` | ✅ | 1 | La Rioja, 7,5 % |
| Q04 | Net income 2024 and change vs 2022 | `answered` | `answered` | ✅ | 1 | 597,850,000 ARS; +104.15% |
| Q05 | 2024 sales per resident in Córdoba (2023 population) | `answered` | `answered` | ✅ | 1 | 162.02 ARS per resident |
| Q06 | Provincia con mayores ventas por habitante | `answered` | `answered` | ✅ | 1 | Formosa, 165,43 ARS por habitante |
| Q07 | Salta's 2023 population, tourists in 2025 | `partial` | `partial` | ✅ | 1 | 2,246,894; 2025 not available (data covers 2023-2024) |
| Q08 | Población de Mendoza en 2024 | `cannot_answer` | `cannot_answer` | ✅ | 0 | la tabla cubre 2022 y 2023 |
| Q09 | Ventas en Salta en dólares | `cannot_answer` | `cannot_answer` | ✅ | 0 | datos en miles de ARS, sin tipo de cambio |
| Q10 | Desempleo en Rosario | `cannot_answer` | `cannot_answer` | ✅ | 0 | no hay datos de desempleo |
| Q11 | Patagonia units vs national average | `cannot_answer` | `cannot_answer` | ✅ | 0 | no national average in the data |
| Q12 | Unidades del canal Online en 2023 | `cannot_answer` | `cannot_answer` | ✅ | 1 | falta el dato para ese año y canal |

**12/12 correct.** "Programs" is how many programs were
written: 0 means the question was judged unanswerable before any code; 2 or more would mean
self-correction. The full answers are in [evaluation/results.json](evaluation/results.json).

### What changed during evaluation

Two questions failed in earlier runs, with the right status but a failed check. Both were
investigated in the generated code:

- **Q03**: the agent found the right province, but *recomputed* the share (584.272,7 / total
  = 7,45 %) instead of reading the table's own *Participación* column (7,5 %, rounded in the
  PDF). Both are defensible, but a report's published figure should win. **Fix to the agent:**
  the code-generation prompt now says to read a figure the table already reports instead of
  recomputing it. Re-runs return 7,5 %.
- **Q03 and Q06**: my checks were stricter than the questions. Q03's check also required the
  sales amount, which the question doesn't ask for; Q06 asks *which* province, but its check
  required the per-resident figure. **Fix to the checks:** Q03 now requires the share (7,5,
  or 7,45 as an acceptable alternative), Q06 the province name. Both changes were made after
  seeing the results, which is why they're reported here.

An answer seen during development ("0.1639 thousands ARS per resident", for a similar
per-resident question) led to another prompt rule: express results in readable units. Q04 now answers in ARS
(597,850,000) where the table is in thousands of ARS (597,850); the checker accepts it and
notes the unit.

Reading the answers (not only the checks) showed one more problem. For Q06 (*which*
province had the highest sales per resident) the program often printed only the name,
`{"value": "Formosa", "unit": "ARS per inhabitant"}`, so the answer read "Formosa con
ARS per habitante" with no figure. The check passed, since it requires only the name, and
the answer writer correctly didn't invent the number. It happened in 3 of 4 runs, even at
temperature 0. Two fixes:

- **Prompt rule:** when a part asks which entity has the highest or lowest value, report
  the entity with its value as an object (`{"Formosa": 165.43}`); a text value has unit `""`.
- **Output check in code** (`rules.check_result`): a text value with a unit is rejected,
  because a unit belongs to a number, so the number was left out. The error goes back to
  `generate_code` as a normal self-correction.

With the rule, Q06 returned `{"Formosa": 165.43}` in 5 of 5 runs, on the first program.

## Observations

- **Answerability is the strongest part.** All six questions that can't be (fully) answered
  got the right status in every run, with the correct reason: a missing year (Q07, Q08), no
  exchange rate (Q09), no related table (Q10), no reported national average without
  substituting a computed one (Q11).
- **Q12 shows the run-time check working.** `select` judged the Online 2023 value
  available; the program found the cell empty, reported it as `missing`, and code lowered the
  status to `cannot_answer`. Its reason ("missing for that year and channel") is less
  informative than it could be: the table's description says the channel only started in
  March 2024, but the answer doesn't mention it. Q12 was also the slowest question (29 s).
- **No self-correction happened in this run:** every program worked first time. The retry
  path (an error or an invalid output going back to `generate_code`) is exercised by the
  graph tests with real programs in the real sandbox, but this report has no live example
  of it. The same holds for the text-with-a-unit check, which is covered by a unit test.
- **Multi-table questions** (Q05, Q06) joined tables from two PDFs on province names, which
  works because the names are spelled identically in both documents; with differing
  spellings the join would silently drop rows.

## Test on a real report

The sample PDFs have ruled tables. To check robustness, the system was also run on a real
INDEC report (EPH,
first half of 2026; 20 pages, 15 tables). The PDF was used temporarily and is not part
of the submission.

### What failed

Almost all the problems were in table extraction, not in the agents:

- **Rows were broken.** INDEC tables use shaded bands instead of cell borders. PyMuPDF's
  default finder splits columns at drawn lines, so every unshaded row came out as one cell
  holding all its values (`'Cuyo 2,32 2,31 2,32 …'`).
- **Headers were lost.** Headers sit above the first shaded band, outside the detected
  table, so most columns were named `column_1 … column_N`.
- **Tables were split or merged wrongly.** One table on page 16 became five; Cuadros 4.1
  and 4.2 (same layout, consecutive pages) were merged into one 86-row table by the
  continuation rule; the cover page and the table of contents were indexed as tables.
- **The context was wrong.** The PDF stores text in drawing order, so captions were
  attached to the wrong tables, and a repeated page footer became one table's name.

### Changes made

In `services/pdf_extract.py`:

- **Detection:** PyMuPDF's layout analysis (the `pymupdf-layout` package) with
  `find_tables(union=True, refine=True)`, which finds tables drawn without lines and
  re-splits merged rows and columns. Cells are read from PyMuPDF's refined grid, which keeps
  a label spanning two columns whole ("2026", where the plain extraction gives "20" / "26").
  The two sample PDFs extract exactly as before.
- **Headers:** a label over the whole table ("2026", "Pesos", "%") is passed to the
  metadata prompt as a table-wide note instead of becoming part of every column name.
  Group labels apply to the columns nearest them.
- **Rows:** label-only rows become a `Section` column.
  Confidence-interval rows (`[41,4 ; 43,7]`) become min/max columns, so the estimates
  stay numeric.
- **Continuations:** a caption ("Cuadro 4.2", "Table 2") between two parts means a new table.
- **Filtering:** tables of contents (dot leaders) and near-empty fragments are skipped
  with a warning.
- **Context:** text is sorted into reading order, each table's caption is always included,
  and running headers and footers (text repeated on three or more pages) are dropped.

Elsewhere:

- `services/numbers.py` parses a unit written before the value (`% 5,6`).
- **Retrieval** also adds tables whose row labels exactly match an entity in the question.
  "Gran Rosario" is one of Cuadro 4.1's 42 row labels, and semantic search over the whole
  table's embedding didn't find it.
- **Re-ingesting** a PDF now removes the tables it no longer yields from Pinecone and the
  table store (needed after an extractor change). A table found but rejected this time keeps
  its previous version. `scripts/remove_document.py` removes all of one PDF's tables.
- **Prompts:** the metadata prompt keeps the caption's own terms; `select` lists only the tables a
  part's calculation uses; the code prompt says a named entity's value comes from its own
  row, never from a total, and that only what each part asks for is reported. The retry
  message for an over-long table name says how many characters to cut.


The course questions were re-run after these changes, with the report still indexed:
12/12 correct (the table above in Results).

## Limitations of this evaluation

- The sample PDFs are synthetic and both they and the questions were written by the system's
  author, so the set tests the designed behaviours. The real report above is one document,
  and its questions were also written by the author, after the extraction had been fixed on
  that same document.
- Twelve questions over five tables (plus twelve over the real report) is small; results on a larger, independent set would be
  more convincing.
- The checks verify statuses, numbers, names and selected tables, not the wording of the
  explanation. Q06's answer without its figure passed them; it was found by reading the
  answers.

## LangSmith traces

Each node of each graph appears as a span. Representative traces:

| Trace | What it shows | Link |
|---|---|---|
| Ingestion of `provincial_statistics_yearbook.pdf` | `extract`, two parallel `process_table` branches (enrich → validate → build_record), then one `index` | _public link to be added_ |
| Q05: multi-table question | retrieval with bilingual searches, selection of two tables from two PDFs, generated code joining them, sandbox run, answer | _public link to be added_ |
| Q07: `partial` | one part answered, one judged unanswerable in `select` (2025 not in the data); only the answerable part goes to code | _public link to be added_ |
| Q12: status lowered at run time | `select` says answerable, the program reports the value as `missing`, status becomes `cannot_answer` | _public link to be added_ |

<!-- To fill in: open each trace in LangSmith, Share, and paste its public link (or a
screenshot). The traces can be found by their eval_id metadata (Q05, Q07, Q12). -->
