"""Process raw electoral datasets into normalized 9-party constituency matrices and electorates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET
import zipfile
import pandas as pd

from .config import (
    CODE_TO_CONSTITUENCY_NAME,
    CONSTITUENCY_NAME_TO_CODE,
    DEFAULT_PROCESSED_GEOGRAPHY_DIR,
    DEFAULT_RAW_GEOGRAPHY_DIR,
    MODEL_PARTIES_9,
    OFFICIAL_CONSTITUENCY_CODES,
)


def _parse_xlsx_sheet(xlsx_path: Path, sheet_name: str) -> list[dict[str, str | None]]:
    """Helper to parse xlsx sheet without openpyxl dependency."""
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
                    sst.append("".join(si.itertext()))

        wb_root = ET.fromstring(z.read("xl/workbook.xml"))
        sheet_file = "xl/worksheets/sheet1.xml"
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


def process_constituency_electorates(
    raw_geo_dir: Path | str | None = None,
    raw_mandates_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
) -> Path:
    """Normalize constituency electorate and turnout history (2014, 2018, 2022, 2026)."""
    r_geo = Path(raw_geo_dir) if raw_geo_dir else DEFAULT_RAW_GEOGRAPHY_DIR
    r_man = Path(raw_mandates_dir) if raw_mandates_dir else DEFAULT_RAW_GEOGRAPHY_DIR.parents[0] / "mandates"
    p_dir = Path(processed_dir) if processed_dir else DEFAULT_PROCESSED_GEOGRAPHY_DIR
    p_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    # 1. 2014 Electorates and Valid Votes from JSON
    json_2014_path = r_geo / "val2014_constituencies_votes.json"
    with open(json_2014_path, "r", encoding="utf-8") as f:
        data_14 = json.load(f)

    for code in OFFICIAL_CONSTITUENCY_CODES:
        info = data_14[code]
        el_cnt = info["eligible_voters"]
        val_cnt = info["valid_votes"]
        turnout = val_cnt / el_cnt if el_cnt > 0 else 0.0
        records.append({
            "election_year": 2014,
            "constituency_code": code,
            "constituency_name": CODE_TO_CONSTITUENCY_NAME[code],
            "eligible_voters": el_cnt,
            "valid_votes": val_cnt,
            "turnout_rate": round(turnout, 6),
        })

    # 2. 2018 and 2022 Electorates and Valid Votes from comparative spreadsheet
    xlsx_path = r_man / "slutligt_valresultat_riksdagen_2018_2022.xlsx"
    sheet_rows = _parse_xlsx_sheet(xlsx_path, sheet_name="Valkrets")

    cur_vk: str | None = None
    data_18_22: dict[tuple[int, str], dict[str, int]] = {}

    for r in sheet_rows:
        vk = r.get("B")
        if vk and vk != "Valkrets":
            cur_vk = vk.strip()

        party_c = r.get("C")
        if cur_vk and party_c == "Giltiga Röster":
            if (2022, cur_vk) not in data_18_22:
                data_18_22[(2022, cur_vk)] = {}
            if (2018, cur_vk) not in data_18_22:
                data_18_22[(2018, cur_vk)] = {}
            if r.get("D") is not None:
                data_18_22[(2022, cur_vk)]["valid"] = int(float(r["D"]))
            if r.get("H") is not None:
                data_18_22[(2018, cur_vk)]["valid"] = int(float(r["H"]))
        elif cur_vk and party_c == "Röstberättigade":
            if r.get("D") is not None:
                data_18_22[(2022, cur_vk)]["eligible"] = int(float(r["D"]))
            if r.get("H") is not None:
                data_18_22[(2018, cur_vk)]["eligible"] = int(float(r["H"]))

    for yr in [2018, 2022]:
        for code in OFFICIAL_CONSTITUENCY_CODES:
            c_name = CODE_TO_CONSTITUENCY_NAME[code]
            info_dict = data_18_22.get((yr, c_name), {})
            el_cnt = info_dict.get("eligible", 0)
            val_cnt = info_dict.get("valid", 0)
            turnout = val_cnt / el_cnt if el_cnt > 0 else 0.0
            records.append({
                "election_year": yr,
                "constituency_code": code,
                "constituency_name": c_name,
                "eligible_voters": el_cnt,
                "valid_votes": val_cnt,
                "turnout_rate": round(turnout, 6),
            })

    # 3. 2026 Electorates from 2026 dataset
    xlsx_26_path = r_geo / "antal_rostberattigade_2026_riksdagen.xlsx"
    rows_26 = _parse_xlsx_sheet(xlsx_26_path, sheet_name="rostber_per_distrikt")
    el_2026: dict[str, int] = {c: 0 for c in OFFICIAL_CONSTITUENCY_CODES}

    for r in rows_26[1:]:
        vk_code = r.get("A")
        tot = r.get("O")
        if vk_code and tot and vk_code in el_2026:
            el_2026[vk_code] += int(float(tot))

    for code in OFFICIAL_CONSTITUENCY_CODES:
        records.append({
            "election_year": 2026,
            "constituency_code": code,
            "constituency_name": CODE_TO_CONSTITUENCY_NAME[code],
            "eligible_voters": el_2026[code],
            "valid_votes": None,
            "turnout_rate": None,
        })

    df = pd.DataFrame(records)
    out_file = p_dir / "constituency_electorates_2014_2026.csv"
    df.to_csv(out_file, index=False)
    print(f"Generated {out_file} ({len(df)} rows across 2014-2026)")
    return out_file


def process_constituency_party_votes(
    raw_geo_dir: Path | str | None = None,
    raw_mandates_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
) -> Path:
    """Normalize historical 9-party constituency vote matrices for 2014, 2018, and 2022."""
    r_geo = Path(raw_geo_dir) if raw_geo_dir else DEFAULT_RAW_GEOGRAPHY_DIR
    r_man = Path(raw_mandates_dir) if raw_mandates_dir else DEFAULT_RAW_GEOGRAPHY_DIR.parents[0] / "mandates"
    p_dir = Path(processed_dir) if processed_dir else DEFAULT_PROCESSED_GEOGRAPHY_DIR
    p_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []

    # 1. 2014 Votes
    json_2014_path = r_geo / "val2014_constituencies_votes.json"
    with open(json_2014_path, "r", encoding="utf-8") as f:
        data_14 = json.load(f)

    party_map_14 = {
        "M": "M", "C": "C", "FP": "L", "KD": "KD", "S": "S", "V": "V", "MP": "MP", "SD": "SD",
    }

    for code in OFFICIAL_CONSTITUENCY_CODES:
        c_name = CODE_TO_CONSTITUENCY_NAME[code]
        raw_votes = data_14[code]["party_votes"]
        const_valid = data_14[code]["valid_votes"]

        party_votes: dict[str, int] = {p: 0 for p in MODEL_PARTIES_9}
        for k, v in raw_votes.items():
            if k in party_map_14:
                party_votes[party_map_14[k]] += v
            else:
                party_votes["REST"] += v

        for p in MODEL_PARTIES_9:
            v_p = party_votes[p]
            records.append({
                "election_year": 2014,
                "constituency_code": code,
                "constituency_name": c_name,
                "party": p,
                "votes": v_p,
                "constituency_valid_votes": const_valid,
                "party_share": v_p / const_valid if const_valid > 0 else 0.0,
            })

    # 2. 2018 and 2022 Votes from processed mandates dataset
    mandates_votes_path = p_dir.parents[0] / "mandates" / "historical_constituency_votes.csv"
    df_man_votes = pd.read_csv(mandates_votes_path)

    for yr in [2018, 2022]:
        sub_yr = df_man_votes[df_man_votes["election_year"] == yr]
        for code in OFFICIAL_CONSTITUENCY_CODES:
            c_name = CODE_TO_CONSTITUENCY_NAME[code]
            sub_c = sub_yr[sub_yr["constituency_code"] == int(code)]
            const_valid = sub_c["constituency_valid_votes"].iloc[0]

            p_votes: dict[str, int] = {p: 0 for p in MODEL_PARTIES_9}
            for _, r in sub_c.iterrows():
                p_code = r["party"]
                if p_code in p_votes:
                    p_votes[p_code] += int(r["votes"])
                elif p_code in ["OTHER", "FI", "ÖVR"]:
                    p_votes["REST"] += int(r["votes"])

            for p in MODEL_PARTIES_9:
                v_p = p_votes[p]
                records.append({
                    "election_year": yr,
                    "constituency_code": code,
                    "constituency_name": c_name,
                    "party": p,
                    "votes": v_p,
                    "constituency_valid_votes": const_valid,
                    "party_share": v_p / const_valid if const_valid > 0 else 0.0,
                })

    df = pd.DataFrame(records)
    df = df.sort_values(by=["election_year", "constituency_code", "party"]).reset_index(drop=True)

    out_file = p_dir / "constituency_party_votes_2014_2022.csv"
    df.to_csv(out_file, index=False)
    print(f"Generated {out_file} ({len(df)} rows across 2014, 2018, 2022)")
    return out_file


def process_all_geography_data(
    raw_geo_dir: Path | str | None = None,
    raw_mandates_dir: Path | str | None = None,
    processed_dir: Path | str | None = None,
) -> dict[str, Path]:
    e_file = process_constituency_electorates(raw_geo_dir, raw_mandates_dir, processed_dir)
    v_file = process_constituency_party_votes(raw_geo_dir, raw_mandates_dir, processed_dir)
    return {
        "electorates": e_file,
        "party_votes": v_file,
    }
