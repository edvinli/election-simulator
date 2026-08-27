# Official Swedish Parliamentary Election Results Data Pipeline

## 1. Overview and Purpose

This pipeline acquires, parses, normalizes, and validates official historical results for Swedish parliamentary (*Riksdag*) general elections across the modern era:

* **Elections Covered**: **2002, 2006, 2010, 2014, 2018, 2022** (national Riksdag elections).
* **Authoritative Source**: **Valmyndigheten** (Swedish Election Authority) sole official source.
* **Output Datasets**:
  1. `data/processed/elections/riksdag_election_results_source_parties.csv`: Source-party-level table preserving all 265 original party lines across elections.
  2. `data/processed/elections/riksdag_election_results.csv`: Normalized canonical 10-party table (`M`, `L`, `C`, `KD`, `S`, `V`, `MP`, `SD`, `FI`, `OTHER`) with a fixed 10-row grid per election.

---

## 2. Authoritative Data Sources & Provenance

All raw files are stored without manual modification in `data/raw/elections/` alongside a SHA-256 retrieval manifest (`data/raw/elections/retrieval_manifest.json`):

| Election | Election Date | Source Authority & Document Format | Official Source URL |
| :---: | :---: | :--- | :--- |
| **2022** | 2022-09-11 | Valmyndigheten JSON API endpoint | `https://resultat.val.se/data/resultat/val2022/RD_S.json` |
| **2018** | 2018-09-09 | Valmyndigheten official final result HTML | `https://historik.val.se/val/val2018/slutresultat/R/rike/index.html` |
| **2014** | 2014-09-14 | Valmyndigheten official final result HTML | `https://historik.val.se/val/val2014/slutresultat/R/rike/index.html` |
| **2010** | 2010-09-19 | Valmyndigheten official final result HTML | `https://historik.val.se/val/val2010/slutresultat/R/rike/index.html` |
| **2006** | 2006-09-17 | Valmyndigheten official final result HTML | `https://historik.val.se/val/val2006/slutlig/R/rike/roster.html` & `ovriga.html` |
| **2002** | 2002-09-15 | Valmyndigheten official final result HTML | `https://historik.val.se/val/val_02/slutresultat/00R/00.html` |

---

## 3. Canonical Party Codes & Historical Normalization

Canonical party schema:

$$\text{M, L, C, KD, S, V, MP, SD, FI, OTHER}$$

### Party Mapping Rules
* **`M`**: Moderaterna / Moderata Samlingspartiet.
* **`L`**: Liberalerna / Liberalerna (tidigare Folkpartiet) / Folkpartiet / Folkpartiet liberalerna / FP.
* **`C`**: Centerpartiet.
* **`KD`**: Kristdemokraterna / Kristdemokratiska Samhällspartiet / KDS.
* **`S`**: Socialdemokraterna / Arbetarepartiet-Socialdemokraterna / Sveriges Socialdemokratiska Arbetareparti.
* **`V`**: Vänsterpartiet / Vänsterpartiet Kommunisterna / VPK.
* **`MP`**: Miljöpartiet / Miljöpartiet de gröna.
* **`SD`**: Sverigedemokraterna.
* **`FI`**: Feministiskt initiativ / F!.
* **`OTHER`**: All other valid votes for parties not mapped to the 9 named categories.

### Missing-Party Representation
If a canonical party did not participate in a given election (e.g. `FI` in 2002, prior to its founding in 2005), it is explicitly recorded with `votes = 0`, `vote_share = 0.0`, and `source_vote_share = 0.0`. Absence of official election votes is represented deterministically as zero rather than null.

---

## 4. Construction and Provenance of `OTHER`

To guarantee complete provenance and reconstructability:

1. **Source Parties Table** (`riksdag_election_results_source_parties.csv`):
   Contains every individual minor party line reported by Valmyndigheten (e.g. Piratpartiet, Medborgerlig Samling, Alternativ för Sverige, Partiet Nyans, Enhet, etc.).
2. **Canonical Table** (`riksdag_election_results.csv`):
   $\text{OTHER}$ is constructed as:
   $$\text{votes}_{\text{OTHER}} = \text{valid\_votes\_total} - \sum_{p \in \{\text{M, L, C, KD, S, V, MP, SD, FI}\}} \text{votes}_p$$
   This ensures that:
   $$\sum_{p \in \text{CANONICAL}} \text{votes}_p \equiv \text{valid\_votes\_total}$$
   with exactly **0 difference** across every single election.

---

## 5. Vote-Share Conventions

Both calculated and published vote shares are stored:

* `vote_share`: Exact floating-point percentage calculated directly from integer vote counts:
  $$\text{vote\_share} = \frac{\text{votes}}{\text{valid\_votes\_total}} \times 100$$
* `source_vote_share`: Published rounded percentage from Valmyndigheten's official report.
* Validation ensures that calculated shares agree with published shares within standard rounding tolerance ($\le 0.02\%$).

---

## 6. Refresh Commands

The election results pipeline is fully integrated into the project's `Makefile`:

```bash
# Fetch raw documents from Valmyndigheten, normalize, and validate
make fetch-election-results

# Offline mode: rebuild processed datasets from cached data/raw/elections/ files
make process-election-results

# Run full automated unit test suite (including test_elections.py)
make test-pollofpolls
```
