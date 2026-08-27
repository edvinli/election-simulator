"""Process raw Valmyndigheten electoral files into clean, normalized datasets."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET
import zipfile
import pandas as pd

from .config import (
    CANONICAL_PARTIES,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    FIXED_SEATS_2018,
    FIXED_SEATS_2022,
    FIXED_SEATS_2026,
    OFFICIAL_CONSTITUENCIES,
    TOTAL_FIXED_SEATS,
    normalize_party_code,
)


def _parse_xlsx_sheet_cells(xlsx_path: Path, sheet_name: str | None = None) -> list[dict[str, str | None]]:
    """Parse rows of an xlsx sheet into list of dicts mapping column letter to string value."""
    ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(xlsx_path) as z:
        sst: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            sst_root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in sst_root.findall("s:si", ns):
                t_el = si.find("s:t", ns)
                if t_el is not None and t_el.text:
                    sst.append(t_el.text)
                else:
                    full_txt = "".join(si.itertext())
                    sst.append(full_txt)

        wb_root = ET.fromstring(z.read("xl/workbook.xml"))
        sheet_file = "xl/worksheets/sheet1.xml"
        if sheet_name:
            for idx, s in enumerate(wb_root.findall("s:sheets/s:sheet", ns)):
                if s.attrib.get("name") == sheet_name:
                    sheet_file = f"xl/worksheets/sheet{idx+1}.xml"
                    break

        ws_root = ET.fromstring(z.read(sheet_file))
        rows: list[dict[str, str | None]] = []
        for r in ws_root.findall("s:sheetData/s:row", ns):
            row_dict: dict[str, str | None] = {}
            for c in r.findall("s:c", ns):
                ref = c.attrib.get("r", "")
                col = "".join([ch for ch in ref if ch.isalpha()])
                t_attr = c.attrib.get("t")
                v_el = c.find("s:v", ns)
                val = v_el.text if v_el is not None else None
                if val is not None and t_attr == "s":
                    val = sst[int(val)]
                row_dict[col] = val
            rows.append(row_dict)
        return rows


def process_constituencies_2026(
    raw_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
) -> Path:
    """Build constituencies_2026.csv with official Valmyndigheten constituency codes, names, and fixed seats."""
    p_dir = Path(processed_dir) if processed_dir else DEFAULT_PROCESSED_DIR
    p_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for code, name in sorted(OFFICIAL_CONSTITUENCIES.items()):
        seats = FIXED_SEATS_2026[code]
        rows.append({
            "constituency_code": code,
            "constituency_name": name,
            "fixed_seats_2026": seats,
        })

    df = pd.DataFrame(rows)
    if df["fixed_seats_2026"].sum() != TOTAL_FIXED_SEATS:
        raise ValueError(f"2026 fixed seats sum {df['fixed_seats_2026'].sum()} != {TOTAL_FIXED_SEATS}")

    out_file = p_dir / "constituencies_2026.csv"
    df.to_csv(out_file, index=False)
    print(f"Generated {out_file} (29 constituencies, total fixed seats: {TOTAL_FIXED_SEATS})")
    return out_file


def process_historical_constituency_votes(
    raw_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
) -> Path:
    """Extract and normalize 2018 and 2022 constituency party votes from Valmyndigheten comparative dataset."""
    r_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    p_dir = Path(processed_dir) if processed_dir else DEFAULT_PROCESSED_DIR
    p_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = r_dir / "slutligt_valresultat_riksdagen_2018_2022.xlsx"
    sheet_rows = _parse_xlsx_sheet_cells(xlsx_path, sheet_name="Valkrets")

    # Mapping from Swedish constituency name to official code
    name_to_code = {name: code for code, name in OFFICIAL_CONSTITUENCIES.items()}

    # Data collection
    parsed_records: list[dict[str, Any]] = []
    current_vk_name: str | None = None

    valid_votes_by_year_const: dict[tuple[int, str], int] = {}

    # First pass: identify valid votes per constituency and year
    for r in sheet_rows:
        vk = r.get("B")
        if vk and vk != "Valkrets":
            current_vk_name = vk.strip()

        party_c = r.get("C")
        if current_vk_name and party_c == "Giltiga Röster":
            if r.get("D") is not None:
                valid_votes_by_year_const[(2022, current_vk_name)] = int(float(r["D"]))
            if r.get("H") is not None:
                valid_votes_by_year_const[(2018, current_vk_name)] = int(float(r["H"]))

    # Second pass: extract individual party votes
    current_vk_name = None
    for r in sheet_rows:
        vk = r.get("B")
        if vk and vk != "Valkrets":
            current_vk_name = vk.strip()

        party_c = r.get("C")
        if not current_vk_name or not party_c:
            continue
        if party_c in ["Giltiga Röster", "Röstberättigade", "Valdeltagande"] or party_c.startswith("Ogiltiga"):
            continue

        party_code = normalize_party_code(party_c)
        code = name_to_code[current_vk_name]

        # 2022 vote
        if r.get("D") is not None:
            v22 = int(float(r["D"]))
            parsed_records.append({
                "election_year": 2022,
                "constituency_code": code,
                "constituency_name": current_vk_name,
                "party": party_code,
                "party_source_name": party_c,
                "votes": v22,
                "constituency_valid_votes": valid_votes_by_year_const[(2022, current_vk_name)],
            })

        # 2018 vote
        if r.get("H") is not None:
            v18 = int(float(r["H"]))
            parsed_records.append({
                "election_year": 2018,
                "constituency_code": code,
                "constituency_name": current_vk_name,
                "party": party_code,
                "party_source_name": party_c,
                "votes": v18,
                "constituency_valid_votes": valid_votes_by_year_const[(2018, current_vk_name)],
            })

    df = pd.DataFrame(parsed_records)
    # Sort deterministically
    df = df.sort_values(by=["election_year", "constituency_code", "party"]).reset_index(drop=True)

    out_file = p_dir / "historical_constituency_votes.csv"
    df.to_csv(out_file, index=False)
    print(f"Generated {out_file} ({len(df)} rows across 2018 and 2022)")
    return out_file


def process_historical_certified_mandates(
    raw_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
) -> Path:
    """Extract official certified seat allocations (fixed, adjustment, total) per constituency and party."""
    r_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    p_dir = Path(processed_dir) if processed_dir else DEFAULT_PROCESSED_DIR
    p_dir.mkdir(parents=True, exist_ok=True)

    name_to_code = {name: code for code, name in OFFICIAL_CONSTITUENCIES.items()}

    records: list[dict[str, Any]] = []

    # 1. 2018 from val2018_valda.html
    html_18_path = r_dir / "val2018_valda.html"
    with open(html_18_path, "r", encoding="iso-8859-1") as f:
        html_18 = f.read()

    tables_18 = re.findall(r"<table[^>]*>(.*?)</table>", html_18, re.DOTALL | re.IGNORECASE)
    t2_18 = tables_18[2]
    rows_18 = re.findall(r"<tr[^>]*>(.*?)</tr>", t2_18, re.DOTALL | re.IGNORECASE)
    hdr_18 = [
        re.sub(r"<[^>]+>", "", c).strip()
        for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", rows_18[0], re.DOTALL | re.IGNORECASE)
    ]
    parties_18 = hdr_18[1:-1]

    for r in rows_18[1:]:
        cells = [
            re.sub(r"<[^>]+>", "", c).strip().replace("&nbsp;", "0")
            for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", r, re.DOTALL | re.IGNORECASE)
        ]
        if not cells or len(cells) < len(parties_18) + 2:
            continue
        vk_raw = cells[0].replace("&auml;", "ä").replace("&ouml;", "ö").replace("&aring;", "å").replace("&Auml;", "Ä").replace("&Ouml;", "Ö").replace("&Aring;", "Å")
        if vk_raw == "Sverige" or vk_raw not in name_to_code:
            continue

        c_code = name_to_code[vk_raw]
        for p_idx, p_str in enumerate(parties_18):
            p_code = normalize_party_code(p_str)
            tot_seats = int(cells[p_idx + 1]) if cells[p_idx + 1] != "0" else 0
            if tot_seats > 0:
                records.append({
                    "election_year": 2018,
                    "constituency_code": c_code,
                    "constituency_name": vk_raw,
                    "party": p_code,
                    "total_seats": tot_seats,
                })

    # 2. 2022 from val2022_RD_S.json / comparative spreadsheet
    xlsx_path = r_dir / "slutligt_valresultat_riksdagen_2018_2022.xlsx"
    sheet_rows = _parse_xlsx_sheet_cells(xlsx_path, sheet_name="Valkrets")
    current_vk_name = None

    for r in sheet_rows:
        vk = r.get("B")
        if vk and vk != "Valkrets":
            current_vk_name = vk.strip()

        mandate_party = r.get("L")
        m_2022 = r.get("M")
        if current_vk_name and mandate_party and m_2022 is not None and mandate_party not in ["Parti", "Summa"]:
            seats_22 = int(float(m_2022))
            if seats_22 > 0:
                p_code = normalize_party_code(mandate_party)
                c_code = name_to_code[current_vk_name]
                records.append({
                    "election_year": 2022,
                    "constituency_code": c_code,
                    "constituency_name": current_vk_name,
                    "party": p_code,
                    "total_seats": seats_22,
                })

    df = pd.DataFrame(records)
    df = df.sort_values(by=["election_year", "constituency_code", "party"]).reset_index(drop=True)

    # Verification: each election must sum to 349
    for yr in [2018, 2022]:
        sub_yr = df[df["election_year"] == yr]
        yr_seats = sub_yr["total_seats"].sum()
        if yr_seats != 349:
            raise ValueError(f"Election {yr} certified seats sum to {yr_seats} != 349")

    out_file = p_dir / "historical_certified_mandates.csv"
    df.to_csv(out_file, index=False)
    print(f"Generated {out_file} (Total seats: 2018=349, 2022=349)")
    return out_file


def process_all_mandate_data(
    raw_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
) -> dict[str, Path]:
    r_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    p_dir = Path(processed_dir) if processed_dir else DEFAULT_PROCESSED_DIR

    c_file = process_constituencies_2026(raw_dir=r_dir, processed_dir=p_dir)
    v_file = process_historical_constituency_votes(raw_dir=r_dir, processed_dir=p_dir)
    m_file = process_historical_certified_mandates(raw_dir=r_dir, processed_dir=p_dir)

    return {
        "constituencies_2026": c_file,
        "historical_votes": v_file,
        "historical_mandates": m_file,
    }
