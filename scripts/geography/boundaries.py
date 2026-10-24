"""Display boundaries for the 29 Riksdag constituencies.

The boundaries are presentation geometry only; no model input depends on them.
They are built deterministically from pinned municipality polygons by assigning
every municipality to its constituency (Vallagen 2005:837, 4 kap. 2 §) and
dissolving shared municipal borders.

The source polygons were simplified with mapshaper, which keeps shared borders
vertex-identical, so a border between two municipalities appears once in each
direction.  Dissolving is therefore exact edge cancellation: an edge whose
reverse belongs to the same constituency is interior and is dropped.  This
relies on that shared-vertex topology: a border that did not match
vertex-for-vertex would survive as a stray interior line.  The source is pinned
by hash and the committed output is regenerated and compared in tests, so a
different source cannot change the outlines unreviewed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from scripts.geography.config import (
    CODE_TO_CONSTITUENCY_NAME,
    DEFAULT_PROCESSED_GEOGRAPHY_DIR,
    DEFAULT_RAW_GEOGRAPHY_DIR,
    OFFICIAL_CONSTITUENCY_CODES,
)

BOUNDARY_SCHEMA_VERSION = "1.0"
BOUNDARY_ROLE = "riksdag_constituency_boundaries"
DEFAULT_SOURCE = DEFAULT_RAW_GEOGRAPHY_DIR / "swedish_municipalities_okfse.geojson"
DEFAULT_OUTPUT = DEFAULT_PROCESSED_GEOGRAPHY_DIR / "constituency_boundaries.json"
SOURCE_URL = (
    "https://raw.githubusercontent.com/okfse/sweden-geojson/"
    "ff67bcb030d7d09032ae5f443d27130f05fdf1e0/swedish_municipalities.geojson"
)
SOURCE_ATTRIBUTION = (
    "Municipality polygons from okfse/sweden-geojson (mapshaper-simplified, "
    "originally converted from Valmyndigheten data)"
)

#: Constituencies that are a single municipality.
MUNICIPALITY_CONSTITUENCIES: Mapping[str, str] = {
    "0180": "01",  # Stockholms kommun
    "1280": "11",  # Malmö kommun
    "1480": "16",  # Göteborgs kommun
}

#: Constituencies covering a whole county, or the rest of one after the
#: single-municipality constituencies above.
COUNTY_CONSTITUENCIES: Mapping[str, str] = {
    "01": "02", "03": "03", "04": "04", "05": "05", "06": "06", "07": "07",
    "08": "08", "09": "09", "10": "10", "13": "15", "17": "21", "18": "22",
    "19": "23", "20": "24", "21": "25", "22": "26", "23": "27", "24": "28",
    "25": "29",
}

#: The partitioned counties, by municipality name as listed in the law.
SPLIT_CONSTITUENCY_MUNICIPALITIES: Mapping[str, tuple[str, ...]] = {
    "12": ("Bjuv", "Eslöv", "Helsingborg", "Höganäs", "Hörby", "Höör", "Landskrona", "Svalöv"),
    "13": (
        "Burlöv", "Kävlinge", "Lomma", "Lund", "Sjöbo", "Skurup", "Staffanstorp",
        "Svedala", "Trelleborg", "Vellinge", "Ystad",
    ),
    "14": (
        "Bromölla", "Båstad", "Hässleholm", "Klippan", "Kristianstad", "Osby", "Perstorp",
        "Simrishamn", "Tomelilla", "Åstorp", "Ängelholm", "Örkelljunga", "Östra Göinge",
    ),
    "17": (
        "Ale", "Alingsås", "Härryda", "Kungälv", "Lerum", "Lilla Edet", "Mölndal",
        "Partille", "Stenungsund", "Tjörn", "Öckerö",
    ),
    "18": (
        "Bengtsfors", "Dals-Ed", "Färgelanda", "Lysekil", "Mellerud", "Munkedal", "Orust",
        "Sotenäs", "Strömstad", "Tanum", "Trollhättan", "Uddevalla", "Vänersborg", "Åmål",
    ),
    "19": (
        "Bollebygd", "Borås", "Herrljunga", "Mark", "Svenljunga", "Tranemo",
        "Ulricehamn", "Vårgårda",
    ),
    "20": (
        "Essunga", "Falköping", "Grästorp", "Gullspång", "Götene", "Hjo", "Karlsborg",
        "Lidköping", "Mariestad", "Skara", "Skövde", "Tibro", "Tidaholm", "Töreboda", "Vara",
    ),
}
SPLIT_COUNTIES: Mapping[str, tuple[str, ...]] = {"12": ("12", "13", "14"), "14": ("17", "18", "19", "20")}

SWEDISH_MUNICIPALITY_COUNT = 290
#: Spherical transverse Mercator about SWEREF 99 TM's central meridian.
CENTRAL_MERIDIAN_DEGREES = 15.0
VIEW_BOX_WIDTH = 1000.0
COORDINATE_DECIMALS = 1

Point = tuple[float, float]
Ring = list[Point]


def assign_municipalities(features: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Map each municipality code to its constituency code, failing closed."""

    by_name: dict[tuple[str, str], str] = {}
    assignment: dict[str, str] = {}
    for feature in features:
        properties = feature["properties"]
        code = str(properties["id"])
        if len(code) != 4 or not code.isdigit() or code in assignment:
            raise ValueError(f"Invalid or duplicate municipality code: {code!r}")
        county = code[:2]
        by_name[(county, str(properties["kom_namn"]))] = code
        if code in MUNICIPALITY_CONSTITUENCIES:
            assignment[code] = MUNICIPALITY_CONSTITUENCIES[code]
        elif county in COUNTY_CONSTITUENCIES:
            assignment[code] = COUNTY_CONSTITUENCIES[county]
        elif county not in SPLIT_COUNTIES:
            raise ValueError(f"Municipality {code} has an unknown county code {county}")
        else:
            assignment[code] = ""

    for county, constituencies in SPLIT_COUNTIES.items():
        for constituency in constituencies:
            for name in SPLIT_CONSTITUENCY_MUNICIPALITIES[constituency]:
                code = by_name.get((county, name))
                if code is None:
                    raise ValueError(f"Municipality {name!r} of constituency {constituency} is missing")
                if assignment[code]:
                    raise ValueError(f"Municipality {name!r} is assigned twice")
                assignment[code] = constituency

    unassigned = sorted(code for code, constituency in assignment.items() if not constituency)
    if unassigned:
        raise ValueError(f"Municipalities without a constituency: {unassigned}")
    if len(assignment) != SWEDISH_MUNICIPALITY_COUNT:
        raise ValueError(f"Expected {SWEDISH_MUNICIPALITY_COUNT} municipalities, found {len(assignment)}")
    if set(assignment.values()) != set(OFFICIAL_CONSTITUENCY_CODES):
        raise ValueError("Municipality assignment does not cover exactly the 29 constituencies")
    return assignment


def _feature_rings(feature: Mapping[str, Any]) -> Iterable[Ring]:
    geometry = feature["geometry"]
    if geometry["type"] == "Polygon":
        polygons = [geometry["coordinates"]]
    elif geometry["type"] == "MultiPolygon":
        polygons = geometry["coordinates"]
    else:
        raise ValueError(f"Unsupported geometry type: {geometry['type']}")
    for polygon in polygons:
        for ring in polygon:
            points = [(float(lon), float(lat)) for lon, lat in ring]
            if len(points) < 4 or points[0] != points[-1]:
                raise ValueError("Source polygon ring is not closed")
            yield points


def dissolve_rings(rings: Iterable[Ring]) -> list[Ring]:
    """Union rings that share vertex-identical borders into closed outline rings."""

    edges: set[tuple[Point, Point]] = set()
    for ring in rings:
        for start, end in zip(ring, ring[1:]):
            if start == end:
                continue
            if (start, end) in edges:
                raise ValueError("Source polygons overlap along a same-direction edge")
            edges.add((start, end))
    outgoing: dict[Point, list[Point]] = defaultdict(list)
    for start, end in edges:
        if (end, start) not in edges:
            outgoing[start].append(end)
    for targets in outgoing.values():
        targets.sort()

    dissolved: list[Ring] = []
    for origin in sorted(outgoing):
        while outgoing[origin]:
            ring = [origin]
            current = origin
            while True:
                if not outgoing[current]:
                    raise ValueError(f"Dissolved boundary is not closed near {current}")
                current = outgoing[current].pop(0)
                ring.append(current)
                if current == origin:
                    break
            dissolved.append(ring)
    return dissolved


def project(lon: float, lat: float) -> Point:
    """Spherical transverse Mercator in unit-sphere coordinates, y pointing down."""

    lam = math.radians(lon - CENTRAL_MERIDIAN_DEGREES)
    phi = math.radians(lat)
    x = math.atanh(math.cos(phi) * math.sin(lam))
    y = math.atan2(math.tan(phi), math.cos(lam))
    return x, -y


def _format(value: float) -> str:
    text = f"{value:.{COORDINATE_DECIMALS}f}".rstrip("0").rstrip(".")
    return "0" if text == "-0" else text


def _svg_path(rings: Sequence[Sequence[Point]]) -> str:
    commands = []
    for ring in rings:
        points: list[tuple[str, str]] = []
        for x, y in ring[:-1]:
            point = (_format(x), _format(y))
            if not points or points[-1] != point:
                points.append(point)
        if len(points) > 1 and points[0] == points[-1]:
            points.pop()
        if len(set(points)) < 3:
            continue
        commands.append("M" + "L".join(f"{x},{y}" for x, y in points) + "Z")
    if not commands:
        raise ValueError("Constituency outline collapsed at display precision")
    return "".join(commands)


def build_constituency_boundaries(source: Path = DEFAULT_SOURCE) -> dict[str, Any]:
    source_bytes = source.read_bytes()
    collection = json.loads(source_bytes)
    features = collection["features"]
    assignment = assign_municipalities(features)

    rings_by_constituency: dict[str, list[Ring]] = defaultdict(list)
    for feature in features:
        constituency = assignment[str(feature["properties"]["id"])]
        rings_by_constituency[constituency].extend(_feature_rings(feature))

    projected: dict[str, list[list[Point]]] = {}
    for code in OFFICIAL_CONSTITUENCY_CODES:
        outline = dissolve_rings(rings_by_constituency[code])
        projected[code] = [[project(lon, lat) for lon, lat in ring] for ring in outline]

    xs = [x for rings in projected.values() for ring in rings for x, _ in ring]
    ys = [y for rings in projected.values() for ring in rings for _, y in ring]
    min_x, min_y = min(xs), min(ys)
    scale = VIEW_BOX_WIDTH / (max(xs) - min_x)
    height = math.ceil((max(ys) - min_y) * scale)

    constituencies = []
    for code in OFFICIAL_CONSTITUENCY_CODES:
        scaled = [[((x - min_x) * scale, (y - min_y) * scale) for x, y in ring] for ring in projected[code]]
        constituencies.append({
            "code": code,
            "name": CODE_TO_CONSTITUENCY_NAME[code],
            "municipality_codes": sorted(m for m, c in assignment.items() if c == code),
            "path": _svg_path(scaled),
        })

    return {
        "schema_version": BOUNDARY_SCHEMA_VERSION,
        "role": BOUNDARY_ROLE,
        "source": {
            "url": SOURCE_URL,
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
            "attribution": SOURCE_ATTRIBUTION,
            "constituency_definition": "Vallag (2005:837) 4 kap. 2 §",
        },
        "projection": f"Spherical transverse Mercator, central meridian {CENTRAL_MERIDIAN_DEGREES:g}°E",
        "view_box": [0, 0, int(VIEW_BOX_WIDTH), height],
        "fill_rule": "evenodd",
        "constituencies": constituencies,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    boundaries = build_constituency_boundaries(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(boundaries, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
