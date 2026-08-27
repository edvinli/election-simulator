# SCB Partisympatiundersökningen (PSU) Support-Voting Dataset

This document provides descriptive documentation, data provenance, and quality assurance diagnostics for the historical SCB PSU support-voting panel covering **2010M11 through 2026M05** (Step 1 of the support-voting research plan).

> [!IMPORTANT]
> **Conceptual Distinction Disclaimer**
> `Partisympati` (*bästa parti*) and `Röstningssympati / Val idag` are fundamentally different SCB survey concepts:
> - **Partisympati** measures general ideological and emotional sympathy or party identification among respondents.
> - **Röstningssympati / Val idag** measures intended vote in a hypothetical parliamentary election held today, calibrated to expected turnout and reweighted by SCB's demographic and register-based weighting model.
>
> The aggregate difference between overall party sympathy and overall vote intention must **not** be interpreted directly as tactical voting.

---

## 1. Available PSU Waves

The dataset covers **29 distinct waves** spanning from November 2010 to May 2026:

```text
2010M11, 2011M05, 2011M11, 2012M05, 2012M11, 2013M05, 2013M11, 2014M05, 2014M11,
2015M05, 2015M11, 2016M05, 2016M11, 2017M05, 2017M11, 2018M05, 2018M11, 2019M05,
2019M11, 2020M05, 2020M11, 2021M05, 2021M11, 2022M05, 2022M11, 2023M05, 2024M05,
2025M05, 2026M05
```

### Survey Frequency Timeline
- **2010M11 – 2022M11 (25 waves)**: PSU was fielded semiannually in **May** (`M05`) and **November** (`M11`).
- **2023M05 – 2026M05 (4 waves)**: Starting in 2023, SCB changed the survey schedule to an annual survey fielded exclusively in **May** (`M05`).

---

## 2. Table Coverage and Provenance

All raw datasets were retrieved directly from **SCB Statistikdatabasen (SSD API)**. For every table, the API schema metadata, submitted query payload, and data response are archived under `data/raw/scb_support_voting/` with SHA-256 integrity hashes recorded in `manifest.json`.

| Key | SCB Table ID | SSD Path | Content Description | Dimensions & Selectors | Rows Acquired |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Table A** | `Rostningssympati170` | `ME/ME0201/ME0201A/Rostningssympati170` | Intended vote conditional on party sympathy: $P(\text{vote } p \mid \text{best } j)$ | 11 donor × 11 destination × 29 waves | 3,509 |
| **Table B** | `Nastbastaparti190` | `ME/ME0201/ME0201D/Nastbastaparti190` | Second-choice party conditional on party sympathy: $P(\text{second } p \mid \text{best } j)$ | 11 donor × 11 second-choice × 29 waves | 3,509 |
| **Table C** | `Vid10` | `ME0201A/Vid10` | Overall headline vote intention (*Val idag*) among decided voters | 14 party/block categories × 29 waves | 406 |
| **Table D** | `Partisympati051` | `ME0201B/Partisympati051` | Overall party sympathy (*bästa parti*) | `Kon=TOT` (totalt kön), `Alder=tot18+` (totalt 18+ år), 10 parties × 29 waves | 290 |

Both percentage point estimates (`estimate_pct`) and margins of error (`margin_error_pp`) are preserved separately across all tables.

---

## 3. Missing and Suppressed Cell Rates

In official SCB publications, cells with insufficient sample sizes are suppressed using the string `..` to prevent unreliable estimates and protect respondent privacy.

In this pipeline:
- Suppressed cells (`..`) are parsed as `NaN` (float) and flagged with `value_status = 'suppressed'`.
- Missing cells are flagged with `value_status = 'missing'`.
- Observed cells are flagged with `value_status = 'observed'`.
- **Under no circumstances are suppressed or missing values coerced to zero.**

### Summary of Cell Statuses across 29 Waves

| Table | Total Cells | Observed Cells | Suppressed Cells (`..`) | Missing Cells | Suppression Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Table A (Vote Intention by Sympathy)** | 3,509 | 2,535 | 974 | 0 | **27.76%** |
| **Table B (Second Choice by Sympathy)** | 3,509 | 2,752 | 757 | 0 | **21.57%** |
| **Table C (Overall Vote Intention / Vid10)** | 406 | 289 | 117 | 0 | **28.82%** (primarily composite blocks & NYD) |
| **Table D (Overall Party Sympathy)** | 290 | 261 | 29 | 0 | **10.00%** (NYD in 2010–2026) |

Suppression in Table A occurs predominantly in cross-ideological, low-probability transfers (e.g. V sympathizers voting for KD or M sympathizers voting for V). Transfers between natural coalition partners (e.g. M $\to$ KD, M $\to$ L, S $\to$ MP, S $\to$ V) are largely observed.

---

## 4. Canonical Party Mapping and Category Handling

The pipeline maps raw SCB codes and labels into canonical party identifiers while preserving original raw codes and text in separate columns.

### Canonical Classifications
- **Parliamentary Parties (8)**: `M`, `C`, `L`, `KD`, `MP`, `S`, `V`, `SD`
  - Historical mapping: `FP` / `Folkpartiet` $\to$ `L` (Liberalerna).
- **Historical / Other Named Parties**:
  - `NYD` $\to$ `NYD` (historical party 1991–1994, retained as `historical_party`).
  - `övr` / `övriga` $\to$ `OTHER` (`other_party`).
- **Non-Party Response Categories** (kept distinct from political parties):
  - `ingen sympati/vet ej` $\to$ `NO_SYMPATHY_OR_DONT_KNOW` (`no_sympathy`)
  - `hela väljarkåren` $\to$ `TOTAL_ELECTORATE` (`total_electorate`)
  - `blankt` $\to$ `BLANK_VOTE` (`blank_vote`)
  - `vet ej` $\to$ `DONT_KNOW` (`dont_know`)
  - `inget parti` $\to$ `NO_SECOND_CHOICE` (`no_second_choice`)

---

## 5. Example Donor $\to$ Recipient Matrices

Below are illustrative empirical cross-tabulations for selected waves from the processed panel.

### Latest Wave: 2026M05

#### Intended Vote Conditional on Best Party: $P(\text{vote } p \mid \text{best } j)$
| Donor \ Recipient | M | C | L | KD | MP | S | V | SD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M** | 79.1% | 1.1% | 1.0% | 1.5% | .. | 1.1% | .. | 3.5% |
| **C** | 2.0% | 77.1% | 0.0% | .. | .. | 8.9% | .. | 0.0% |
| **L** | 8.5% | 5.2% | 68.8% | .. | 0.0% | 5.5% | 0.0% | .. |
| **KD** | .. | .. | .. | 79.5% | 0.0% | 0.0% | 0.0% | .. |
| **MP** | 0.0% | .. | 0.0% | 0.0% | 81.4% | 6.7% | 3.3% | .. |
| **S** | 1.0% | 0.5% | .. | .. | 1.1% | 85.2% | 0.6% | 0.7% |
| **V** | 0.0% | .. | .. | 0.0% | 3.6% | 4.9% | 78.5% | .. |
| **SD** | 0.9% | 0.0% | 1.2% | 1.0% | 0.0% | .. | 0.0% | 90.5% |

#### Second-Best Party Conditional on Best Party: $P(\text{second } p \mid \text{best } j)$
| Donor \ Recipient | M | C | L | KD | MP | S | V | SD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M** | 0.0% | 6.9% | 12.5% | 22.7% | 0.9% | 12.0% | 1.2% | 26.1% |
| **C** | 18.3% | 0.0% | 6.9% | 3.7% | 11.8% | 38.3% | 3.3% | .. |
| **L** | 41.6% | 20.4% | 0.0% | 4.8% | .. | 15.5% | .. | .. |
| **KD** | 43.8% | 8.2% | 4.7% | 0.0% | 0.0% | 6.2% | .. | 20.2% |
| **MP** | 1.1% | 10.3% | .. | .. | 0.0% | 39.0% | 34.7% | .. |
| **S** | 9.1% | 16.4% | 2.1% | 1.2% | 15.7% | 0.0% | 22.9% | 3.6% |
| **V** | .. | .. | .. | .. | 30.3% | 47.9% | 0.0% | 4.4% |
| **SD** | 44.0% | .. | 1.7% | 21.2% | 1.0% | 6.0% | 1.3% | 0.0% |

---

### Historical Wave: 2018M05 (Pre-Election Wave)

#### Intended Vote Conditional on Best Party: $P(\text{vote } p \mid \text{best } j)$
| Donor \ Recipient | M | C | L | KD | MP | S | V | SD |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **M** | 79.9% | 1.6% | .. | 0.5% | .. | .. | 0.0% | 4.2% |
| **C** | 2.7% | 79.8% | 1.0% | .. | 0.0% | .. | .. | 2.3% |
| **L** | 3.8% | 3.0% | 76.5% | .. | 0.0% | .. | .. | .. |
| **KD** | 4.3% | .. | .. | 75.9% | .. | .. | 0.0% | .. |
| **MP** | 4.4% | 3.0% | .. | .. | 67.7% | 3.8% | .. | .. |
| **S** | 1.5% | 0.4% | 0.6% | .. | .. | 81.3% | 1.4% | 1.6% |
| **V** | 0.0% | .. | 0.0% | 0.0% | .. | 3.4% | 84.5% | 1.6% |
| **SD** | 0.9% | 0.0% | 0.0% | .. | 0.0% | .. | .. | 93.8% |

---

## 6. Stability of Second-Choice Relationships Over Time

Over the 2010M11–2026M05 window, voters' declared second-choice parties exhibit persistent bloc structures with quantifiable variation:

- **Moderate Party (M) voters**:
  - Second-choice KD: Mean = 15.3% (range: 5.6% – 39.1%, std = 8.1%, observed in 29/29 waves).
  - Second-choice L: Mean = 23.5% (range: 12.5% – 36.5%, std = 6.4%, observed in 29/29 waves).
  - Second-choice SD: Mean = 13.3% (range: 3.2% – 26.1%, std = 6.7%, observed in 29/29 waves).
- **Social Democrat (S) voters**:
  - Second-choice V: Mean = 21.5% (range: 12.9% – 26.2%, std = 3.6%, observed in 29/29 waves).
  - Second-choice MP: Mean = 17.7% (range: 9.1% – 32.8%, std = 6.0%, observed in 29/29 waves).
  - Second-choice C: Mean = 10.2% (range: 2.0% – 22.9%, std = 5.2%, observed in 29/29 waves).
- **Christian Democrat (KD) & Liberal (L) voters**:
  - KD $\to$ M second choice: Mean = 40.5% (range: 24.3% – 53.9%, std = 7.7%, observed in 29/29 waves).
  - L $\to$ M second choice: Mean = 38.5% (range: 27.9% – 50.0%, std = 6.1%, observed in 29/29 waves).
- **Left Party (V) & Green Party (MP) voters**:
  - V $\to$ S second choice: Mean = 46.3% (range: 35.8% – 55.3%, std = 5.3%, observed in 29/29 waves).
  - MP $\to$ S second choice: Mean = 38.4% (range: 28.9% – 49.6%, std = 5.5%, observed in 29/29 waves).

*Note: These statistics quantify descriptive empirical stability and missingness rates across waves without drawing behavioral or causal inferences about tactical voting.*

---

## 7. Cross-Party Flow Observability Near the 4% Threshold

For parties that historically or currently poll near the 4% parliamentary threshold (**L, KD, MP, C**):

1. **Large Donor $\to$ Threshold Party Flows**:
   - **M $\to$ KD**: Observed in **29/29 waves** (100% coverage). Mean intended vote flow is 1.6% (range 0.4% – 8.3%).
   - **M $\to$ L**: Observed in **24/29 waves** (82.8% coverage, 5 waves suppressed). Mean observed flow is 1.0% (range 0.3% – 1.8%).
   - **S $\to$ MP**: Observed in **28/29 waves** (96.6% coverage, 1 wave suppressed). Mean observed flow is 1.2% (range 0.5% – 4.2%).
   - **S $\to$ C**: Observed in **24/29 waves** (82.8% coverage, 5 waves suppressed). Mean observed flow is 0.7% (range 0.0% – 1.8%).
2. **Threshold Party $\to$ Anchor Party Flows**:
   - **KD $\to$ M**: Observed in 23/29 waves (mean 5.0%).
   - **L $\to$ M**: Observed in 29/29 waves (mean 6.4%).
   - **MP $\to$ S**: Observed in 29/29 waves (mean 6.6%).
   - **V $\to$ S**: Observed in 29/29 waves (mean 6.2%).

These high coverage rates indicate that the panel preserves sufficient detail for subsequent empirical analysis of cross-party intentions.

---

## 8. SCB Comparability Caveats and Methodology

1. **2020 Weighting Methodology Revision**:
   - In 2020, SCB introduced an improved calibration and weighting methodology that incorporates updated register-based auxiliary variables and refined non-response adjustments.
   - SCB recalculated historical `Vid10` (*Val idag*) estimates back to 2010 using this revised model.
   - Other PSU tables (such as `Rostningssympati170` and `Nastbastaparti190`) reflect SCB's standard survey weighting at publication.
2. **Frequency Shift (2023)**:
   - Prior to 2023, PSU surveyed voters twice per year (May and November).
   - From 2023 onward, PSU is conducted only in May. Researchers using this panel should account for the change in time step ($\Delta t = 6\text{ months}$ pre-2023 vs $\Delta t = 12\text{ months}$ post-2023).
3. **Response Universe & Reconciliation**:
   - In `Vid10`, percentage shares are calculated over decided voters expressing a party preference (summing to ~100% across parties).
   - In `Rostningssympati170`, the universe for each donor category includes `blankt` and `vet ej`.
   - Reconstructed estimates from the conditional matrix will differ slightly from headline `Vid10` due to turnout weighting, blank/undecided handling, and rounding. These differences are recorded as diagnostic metrics in `validation_report.json` and are never forced into equality.

---

## 9. Processed Dataset Summary

All processed files are stored under `data/processed/scb_support_voting/`:

- [`vote_by_sympathy.csv`](../data/processed/scb_support_voting/vote_by_sympathy.csv): 3,509 rows (wave × best party × intended vote party).
- [`second_choice_by_sympathy.csv`](../data/processed/scb_support_voting/second_choice_by_sympathy.csv): 3,509 rows (wave × best party × second-best party).
- [`overall_vote_intention.csv`](../data/processed/scb_support_voting/overall_vote_intention.csv): 406 rows (wave × party headline *Val idag*).
- [`overall_party_sympathy.csv`](../data/processed/scb_support_voting/overall_party_sympathy.csv): 290 rows (wave × party overall sympathy for total electorate).
- [`scb_donor_recipient_panel.csv`](../data/processed/scb_support_voting/scb_donor_recipient_panel.csv): 1,856 rows ($8 \times 8 \times 29$ parliamentary donor-recipient panel with separate uncertainty measures).
- [`validation_report.json`](../data/processed/scb_support_voting/validation_report.json): Comprehensive machine-readable report with row sums, reconciliation gaps, coverage metrics, and assertion passes.
