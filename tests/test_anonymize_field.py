"""Tests de scripts/anonymize_field.py: mes, ruta y cliente a partir de la ruta del archivo."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "anonymize_field.py"


def load_module():
    spec = importlib.util.spec_from_file_location("anonymize_field", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse(relative: str, flexible: bool = False) -> dict:
    module, src = load_module(), Path("/origen")
    end = module.CUSTOMER_END_RE if flexible else module.WATER_CUSTOMER_END_RE
    record = module.parse_photo(src / relative, src, end)
    return {k: record[k] for k in ("month", "route", "customer")}


def test_water_style_name_keeps_previous_behavior():
    assert parse("08-2018/RUTA 80/Juan  Perez - RUTA 80.jpg") == {
        "month": "2018-08", "route": "80", "customer": "JUAN PEREZ"}


@pytest.mark.parametrize("name", [
    "Juan Perez -RUTA 16.jpg",
    "Juan Perez- RUTA 2.jpg",
    "Juan Perez RUTA 18.jpg",
    "Juan Perez - 18.jpg",
    "Juan Perez -  RUTA 21 (07-2018).jpg",
    "Juan Perez - RUTA 8 (CONEX 2).jpg",
    "Juan Perez . RUTA 9.jpg",
])
def test_flexible_rule_ignores_irregular_route_suffixes(name):
    assert parse(f"07-2018/RUTA 16 (07-2018)/{name}", flexible=True)["customer"] == "JUAN PEREZ"


def test_water_rule_is_unchanged_for_irregular_names():
    # La regla de agua no se toca: sus identificadores ya están publicados.
    assert parse("08-2018/RUTA 80/Juan Perez -RUTA 80.jpg")["customer"] == "JUAN PEREZ -RUTA 80"


def test_month_falls_back_to_the_date_in_the_route_folder():
    assert parse("RUTA 7 (12-2019)/Juan Perez - RUTA 7.jpg", flexible=True) == {
        "month": "2019-12", "route": "7", "customer": "JUAN PEREZ"}


def test_output_name_uses_the_meter_type_prefix():
    module = load_module()
    assert module.photo_name("energia", "2019-12", "7", "m0003") == "energia_2019-12_r7_m0003.jpg"
    assert module.photo_name("agua", "2018-08", "80", "m0020") == "agua_2018-08_r80_m0020.jpg"
