"""Generate the sample PDFs in data/sample_pdfs/.

All data is fictional. The two PDFs are designed to exercise the hard cases in
assignment.md §5.2 and to support the test questions in §9.3:

informe_anual_ferreteria_andina.pdf  (Spanish number format: 1.234,5)
  - Tabla 1: income statement, negatives in parentheses, $ symbols
  - Tabla 2: two-row header with merged cells, one empty cell
  - Tabla 3: sales by province, long enough to continue on the next page,
             header row REPEATED on the continuation

provincial_statistics_yearbook.pdf  (English number format: 1,234.5)
  - Table 1: population by province, only 2022-2023 (no 2024 -> "missing year" questions)
  - Table 2: monthly tourist arrivals, continues on the next page WITHOUT a repeated header

Run with:  uv run python scripts/make_sample_pdfs.py
"""

import random
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Flowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_pdfs"

# Same province names in both PDFs, so the analyst can join tables across documents.
PROVINCES = [
    ("Buenos Aires", "Centro"), ("CABA", "Centro"), ("Córdoba", "Centro"),
    ("Santa Fe", "Centro"), ("Entre Ríos", "Centro"), ("La Pampa", "Centro"),
    ("Mendoza", "Cuyo"), ("San Juan", "Cuyo"), ("San Luis", "Cuyo"),
    ("Tucumán", "NOA"), ("Salta", "NOA"), ("Jujuy", "NOA"),
    ("Catamarca", "NOA"), ("Santiago del Estero", "NOA"), ("La Rioja", "NOA"),
    ("Misiones", "NEA"), ("Corrientes", "NEA"), ("Chaco", "NEA"), ("Formosa", "NEA"),
    ("Neuquén", "Patagonia"), ("Río Negro", "Patagonia"), ("Chubut", "Patagonia"),
    ("Santa Cruz", "Patagonia"), ("Tierra del Fuego", "Patagonia"),
]

GRID = TableStyle([
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dde4ee")),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
])

styles = getSampleStyleSheet()
H1, H2, BODY, CAPTION = styles["Title"], styles["Heading2"], styles["BodyText"], styles["Italic"]


def es_num(value: float, decimals: int = 0) -> str:
    """Format like Spanish reports: 1.234.567,8 and negatives in parentheses."""
    english = f"{abs(value):,.{decimals}f}"
    spanish = english.replace(",", "_").replace(".", ",").replace("_", ".")
    return f"({spanish})" if value < 0 else spanish


def en_num(value: float, decimals: int = 0) -> str:
    return f"{value:,.{decimals}f}"


# Section text before Tabla 3. Besides giving the LLM real context, its length pushes
# the table to the bottom of page 1 so it continues on page 2.
PROVINCE_SECTION = [
    (
        "Este apartado describe la evolución comercial del ejercicio 2024 en cada una de "
        "las jurisdicciones donde opera la compañía. Las cifras se expresan en miles de "
        "pesos corrientes salvo indicación en contrario."
    ),
    (
        "La provincia de Buenos Aires y la Ciudad Autónoma de Buenos Aires concentran la "
        "mayor cantidad de sucursales, aunque el crecimiento más fuerte del año se observó "
        "en las provincias de la región Patagonia, impulsado por la actividad energética."
    ),
    (
        "Las ventas de cada provincia incluyen tanto las operaciones minoristas como las "
        "mayoristas realizadas en sus sucursales. Las ventas del canal Online no se asignan "
        "a ninguna provincia y por lo tanto no forman parte de la tabla siguiente."
    ),
    (
        "La columna de participación indica el porcentaje que representa cada provincia "
        "sobre el total de ventas provinciales del ejercicio."
    ),
]


def informe_anual(path: Path) -> None:
    rng = random.Random(42)  # fixed seed: same PDF every run
    story: list[Flowable] = [
        Paragraph("Ferretería Andina S.A. — Informe Anual 2024", H1),
        Paragraph(
            "Ferretería Andina S.A. es una cadena de ferreterías con presencia en todas las "
            "provincias argentinas. Este informe resume los resultados de los ejercicios "
            "2022 a 2024.", BODY),
        Paragraph("1. Resultados consolidados", H2),
        Paragraph("Tabla 1: Estado de resultados consolidado (en miles de ARS)", CAPTION),
    ]

    # --- Tabla 1: income statement --------------------------------------------
    ventas = [1_850_400, 2_310_750, 2_984_120]
    cost_ratios = [0.62, 0.60, 0.58]
    costo = [-venta * ratio for venta, ratio in zip(ventas, cost_ratios)]
    bruto = [venta + cost for venta, cost in zip(ventas, costo)]
    gastos = [-410_300, -498_900, -655_480]
    neto = [gross + expense for gross, expense in zip(bruto, gastos)]
    lines = [("Ventas netas", ventas), ("Costo de ventas", costo), ("Resultado bruto", bruto),
             ("Gastos operativos", gastos), ("Resultado neto", neto)]
    rows = [["Concepto", "2022", "2023", "2024"]] + [
        [label] + [f"$ {es_num(value)}" for value in values] for label, values in lines
    ]
    story += [Table(rows, style=GRID), Spacer(1, 4),
              Paragraph("Nota: los valores entre paréntesis son negativos.", CAPTION)]

    # --- Tabla 2: two-row header with merged cells, one empty cell ------------
    story += [
        Paragraph("2. Ventas por región", H2),
        Paragraph("Tabla 2: Unidades vendidas y facturación por región, 2023-2024", CAPTION),
    ]
    regions = ["Centro", "Cuyo", "NOA", "NEA", "Patagonia", "Online"]
    rows = [["Región", "2023", "", "2024", ""],
            ["", "Unidades", "Monto (miles ARS)", "Unidades", "Monto (miles ARS)"]]
    rows += [region_sales_row(region, rng) for region in regions]
    regions_table = Table(rows)
    regions_table.setStyle(GRID)
    regions_table.setStyle(TableStyle([
        ("SPAN", (0, 0), (0, 1)), ("SPAN", (1, 0), (2, 0)), ("SPAN", (3, 0), (4, 0)),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#dde4ee")),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, 1), "CENTER"), ("VALIGN", (0, 0), (-1, 1), "MIDDLE"),
    ]))
    story += [regions_table, Paragraph("El canal Online comenzó a operar en marzo de 2024.", CAPTION)]

    # --- Tabla 3: long table, continues on next page, header repeated ---------
    story += [Paragraph("3. Ventas por provincia", H2)]
    story += [Paragraph(text, BODY) for text in PROVINCE_SECTION] + [
        Paragraph("Tabla 3: Ventas netas por provincia, ejercicio 2024 (en miles de ARS)", CAPTION),
    ]
    amounts = [rng.uniform(15_000, 600_000) for _ in PROVINCES]
    total = sum(amounts)
    rows = [["Provincia", "Región", "Ventas 2024", "Participación"]] + [
        [province, region, es_num(amount, 1), f"{es_num(100 * amount / total, 1)} %"]
        for (province, region), amount in zip(PROVINCES, amounts)
    ]
    # repeatRows=1: reportlab redraws the header row on every page the table spans
    story += [Table(rows, style=GRID, repeatRows=1),
              Paragraph("Fuente: sistema de gestión comercial de la compañía.", CAPTION)]

    SimpleDocTemplate(str(path), pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm).build(story)


def region_sales_row(region: str, rng: random.Random) -> list[str]:
    units_2024 = rng.randint(40_000, 400_000)
    amount_2024 = units_2024 * rng.uniform(4.5, 6.5)
    if region == "Online":  # channel opened in 2024: no 2023 data -> empty cells
        return [region, "", "", es_num(units_2024), es_num(amount_2024, 1)]
    units_2023 = int(units_2024 * rng.uniform(0.75, 0.95))
    amount_2023 = units_2023 * rng.uniform(3.5, 5.0)
    return [region, es_num(units_2023), es_num(amount_2023, 1),
            es_num(units_2024), es_num(amount_2024, 1)]


def population_row(province: str, rng: random.Random) -> list[str]:
    population_2022 = rng.randint(150_000, 17_500_000)
    population_2023 = int(population_2022 * rng.uniform(1.002, 1.015))
    change = 100 * (population_2023 / population_2022 - 1)
    return [province, en_num(population_2022), en_num(population_2023), f"{change:.2f}%"]


def arrivals_row(year: int, month: str, arrivals_2023: dict[str, int], rng: random.Random) -> list[str]:
    previous = arrivals_2023[month]
    if year == 2023:
        return [str(year), month, en_num(previous), ""]
    current = int(previous * rng.uniform(0.9, 1.25))
    return [str(year), month, en_num(current), f"{100 * (current / previous - 1):+.1f}%"]


def yearbook(path: Path) -> None:
    rng = random.Random(7)
    story: list[Flowable] = [
        Paragraph("Provincial Statistics Yearbook 2024", H1),
        Paragraph(
            "This yearbook compiles demographic and tourism indicators for the Argentine "
            "provinces. Population figures come from the provincial registries; the 2024 "
            "estimates were not available at the time of publication.", BODY),
        Paragraph("Chapter 1. Population", H2),
        Paragraph("Table 1: Resident population by province (persons), 2022-2023", CAPTION),
    ]

    rows = [["Province", "Population 2022", "Population 2023", "Change (%)"]] + [
        population_row(province, rng) for province, _ in PROVINCES
    ]
    story += [Table(rows, style=GRID)]

    story += [Paragraph("Chapter 2. Tourism", H2)] + [
        Paragraph("International tourist arrivals are recorded at border crossings and "
                  "airports. Figures are monthly totals.", BODY)
    ] + [Paragraph("Table 2: International tourist arrivals by month, 2023-2024", CAPTION)]

    months = ["January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December"]
    arrivals_2023 = {month: rng.randint(380_000, 900_000) for month in months}
    rows = [["Year", "Month", "Arrivals", "Year-on-year change (%)"]] + [
        arrivals_row(year, month, arrivals_2023, rng) for year in (2023, 2024) for month in months
    ]
    # repeatRows=0 (default): the continuation on the next page has NO header row
    story += [Table(rows, style=GRID),
              Paragraph("Year-on-year change is not reported for 2023.", CAPTION)]

    SimpleDocTemplate(str(path), pagesize=A4, topMargin=2 * cm, bottomMargin=2 * cm).build(story)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    informe_anual(OUT_DIR / "informe_anual_ferreteria_andina.pdf")
    yearbook(OUT_DIR / "provincial_statistics_yearbook.pdf")
    print(f"Wrote PDFs to {OUT_DIR}")
