# Swedish Riksdag Mandate Allocation & Electoral Mechanics

## 1. Executive Summary

This module implements the deterministic electoral mechanics foundation for the Swedish Riksdag simulator. The implementation adheres strictly to Swedish constitutional and statutory election law (**Regeringsformen 1974:152** and **Vallagen 2005:837 14 kap.**) and has been verified against official certified election data from **Valmyndigheten**.

The allocator reproduces the certified official seat allocations for the **2018 and 2022 Riksdag elections** down to the exact seat for every party across all **29 Riksdag constituencies** (0 mismatches).

---

## 2. Statutory Framework

### 2.1 Constitutional Thresholds (Regeringsformen 3 kap. 7 §)
* **National 4.0% Threshold**: A party must obtain at least **4.0%** of the valid votes cast nationally to participate in the nationwide proportional entitlement and receive adjustment seats (utjämningsmandat).
* **Constituency 12.0% Exception**: A party obtaining less than 4.0% nationally but at least **12.0%** of the valid votes in a specific constituency is entitled to participate in the allocation of **fixed constituency seats (fasta valkretsmandat)** in that constituency. It is excluded from the nationwide proportional entitlement and receives no adjustment seats.

### 2.2 Seat Quantities and Geometry (Vallagen 4 kap. 3 §)
* **Total Seats**: **349**.
* **Fixed Constituency Seats (Fasta valkretsmandat)**: **310**, distributed among the 29 constituencies prior to each election based on the number of eligible voters as of March 1 using the largest remainder (Hare quota / Hamilton) method.
* **Adjustment Seats (Utjämningsmandat)**: **39**, distributed across the country to restore nationwide proportionality among qualifying parties.

### 2.3 2026 Decided Fixed-Seat Distribution
Valmyndigheten determined the 2026 fixed seat distribution on April 10, 2026 (published in `fasta-valkretsmandat-val-2026.xlsx`):

| Valkretskod | Valkretsnamn | Fasta mandat 2026 | Förändring vs 2022 |
|---|---|---|---|
| **01** | Stockholms kommun | 29 | $\pm 0$ |
| **02** | Stockholms län | 41 | **+1** |
| **03** | Uppsala län | 12 | $\pm 0$ |
| **04** | Södermanlands län | 9 | $\pm 0$ |
| **05** | Östergötlands län | 14 | $\pm 0$ |
| **06** | Jönköpings län | 11 | $\pm 0$ |
| **07** | Kronobergs län | 6 | $\pm 0$ |
| **08** | Kalmar län | 7 | **-1** |
| **09** | Gotlands län | 2 | $\pm 0$ |
| **10** | Blekinge län | 5 | $\pm 0$ |
| **11** | Malmö kommun | 10 | $\pm 0$ |
| **12** | Skåne läns västra | 9 | $\pm 0$ |
| **13** | Skåne läns södra | 12 | $\pm 0$ |
| **14** | Skåne läns norra och östra | 10 | $\pm 0$ |
| **15** | Hallands län | 10 | $\pm 0$ |
| **16** | Göteborgs kommun | 18 | **+1** |
| **17** | Västra Götalands läns västra | 11 | $\pm 0$ |
| **18** | Västra Götalands läns norra | 8 | $\pm 0$ |
| **19** | Västra Götalands läns södra | 7 | $\pm 0$ |
| **20** | Västra Götalands läns östra | 8 | $\pm 0$ |
| **21** | Värmlands län | 9 | $\pm 0$ |
| **22** | Örebro län | 9 | $\pm 0$ |
| **23** | Västmanlands län | 8 | $\pm 0$ |
| **24** | Dalarnas län | 9 | $\pm 0$ |
| **25** | Gävleborgs län | 9 | $\pm 0$ |
| **26** | Västernorrlands län | 7 | **-1** |
| **27** | Jämtlands län | 4 | $\pm 0$ |
| **28** | Västerbottens län | 8 | $\pm 0$ |
| **29** | Norrbottens län | 8 | $\pm 0$ |
| **Summa** | **29 valkretsar** | **310** | **310 fasta + 39 utjämningsmandat = 349** |

---

## 3. The Mandate Allocation Algorithm (Vallagen 14 kap.)

The algorithm executes sequentially with exact rational arithmetic (`fractions.Fraction`):

```
                        [ Constituency Votes ]
                                  │
                                  ▼
             ┌──────────────────────────────────────────┐
             │ Step 1: Threshold & Eligibility Check    │
             │  - National >= 4.0% (inclusive)          │
             │  - Constituency >= 12.0% (inclusive)     │
             └──────────────────────────────────────────┘
                                  │
                                  ▼
             ┌──────────────────────────────────────────┐
             │ Step 2: Fixed Constituency Seats (310)   │
             │  - Modified Sainte-Laguë (1.2, 3, 5...)  │
             │  - Allocated within each 29 constituency │
             └──────────────────────────────────────────┘
                                  │
                                  ▼
             ┌──────────────────────────────────────────┐
             │ Step 3: National Entitlement (349 - L)   │
             │  - Deduct fixed seats won by <4% parties │
             │  - Modified Sainte-Laguë (1.2, 3, 5...)  │
             │  - Allocated nationally across >=4%      │
             └──────────────────────────────────────────┘
                                  │
                                  ▼
             ┌──────────────────────────────────────────┐
             │ Step 4: Overhang Check & Återföring      │
             │  - Excess fixed seats (F_p > E_p)        │
             │  - Retracted from lowest winning comp    │
             │    (excluding const with <3 fixed seats) │
             │  - Reallocated to eligible parties in c  │
             └──────────────────────────────────────────┘
                                  │
                                  ▼
             ┌──────────────────────────────────────────┐
             │ Step 5: Adjustment Seats Placement (39)  │
             │  - U_p = E'_p - F'_p                     │
             │  - Placed into constituency c with max   │
             │    comparison number:                    │
             │    * Divisor = 1.0 if party has 0 fixed  │
             │    * Divisor = 2*s + 1 if party has s>=1 │
             └──────────────────────────────────────────┘
                                  │
                                  ▼
                   [ Exact 349 Seat Allocation ]
```

### 3.1 Step-by-Step Rules
1. **Modified Sainte-Laguë Divisors**:
   - First divisor: $d_1 = \mathbf{1.2} = \frac{6}{5}$ (effective since Jan 1, 2015 via SFS 2014:1384; applies to 2018, 2022, 2026).
   - Subsequent divisors: $3, 5, 7, 9, \dots = 2k + 1$.
2. **Fixed Constituency Seats Allocation (Vallagen 14 kap. 3 §)**:
   - In each constituency $c$, participating parties compete for $F_c$ seats based on comparison numbers $J_{c, p} = \frac{V_{c, p}}{d_k}$.
3. **National Proportional Entitlement (Vallagen 14 kap. 4 §)**:
   - Fixed seats won by parties below 4% nationally ($L$) are deducted: $T' = 349 - L$.
   - The remaining $T'$ seats are allocated among $\ge 4\%$ parties nationally using modified Sainte-Laguë ($d_1 = 1.2$).
4. **Excess Fixed Seats & Return / Reallocation (Återföring - Vallagen 14 kap. 4a–4c §§)**:
   - If party $p$ won $F_p > E_p$ fixed seats, excess seats are retracted from the constituency where $p$ won a seat with the lowest comparison number (prohibited from constituencies with $<3$ fixed seats, e.g. Gotland).
   - Returned seats are reallocated globally among candidate parties in affected constituencies in descending order of comparison quotient. Sub-4% parties with $\ge 12\%$ locally are eligible to receive returned fixed seats in that constituency.
   - National proportional entitlements are updated to guarantee strict 349-seat conservation.
5. **Adjustment Seats Placement (Vallagen 14 kap. 5 §)**:
   - Each nationally qualifying party $p$ is entitled to $U_p = E'_p - F'_p$ adjustment seats.
   - Each adjustment seat is assigned to the constituency where party $p$ has the highest comparison quotient:
     $$J_{c, p} = \begin{cases} \frac{V_{c, p}}{1.0} & \text{if party } p \text{ holds 0 fixed seats in constituency } c \\ \frac{V_{c, p}}{2(\text{seats held in } c) + 1} & \text{if party } p \text{ holds } \ge 1 \text{ seats in constituency } c \end{cases}$$
6. **Tie Breaking (Vallagen 14 kap. 2 §)**:
   - Resolved by lottery (`TieBreaker` protocol / `DeterministicLotteryTieBreaker`). Ties are never resolved alphabetically or by party ID.

---

## 4. Verification & Acceptance Results

### 4.1 Official Phase-by-Phase Mandate Decomposition

The allocator matches the certified 2018 and 2022 Riksdag results across the tested phases (initial fixed seats, post-return fixed seats, adjustment seats, and final seats):

#### 2018 Riksdag Election (Certified Official vs Allocator Output):
| Parti | Fasta valkretsmandat | Återförda / Nyfördelade | Utjämningsmandat | Slutliga mandat |
|---|---|---|---|---|
| **S** | 94 | 0 | 6 | **100** |
| **M** | 66 | 0 | 4 | **70** |
| **SD** | 61 | 0 | 1 | **62** |
| **C** | 31 | 0 | 0 | **31** |
| **V** | 25 | 0 | 3 | **28** |
| **KD** | 16 | 0 | 6 | **22** |
| **L** | 12 | 0 | 8 | **20** |
| **MP** | 5 | 0 | 11 | **16** |
| **Summa** | **310** | **0** | **39** | **349** |

*Note on 2018 fixed seat configuration*: In 2018, Västra Götalands läns norra had 8 fixed seats and Västra Götalands läns östra had 9 fixed seats.

#### 2022 Riksdag Election (Certified Official vs Allocator Output):
| Parti | Fasta valkretsmandat | Återförda / Nyfördelade | Utjämningsmandat | Slutliga mandat |
|---|---|---|---|---|
| **S** | 104 | 0 | 3 | **107** |
| **SD** | 69 | 0 | 4 | **73** |
| **M** | 67 | 0 | 1 | **68** |
| **V** | 16 | 0 | 8 | **24** |
| **C** | 23 | 0 | 1 | **24** |
| **KD** | 13 | 0 | 6 | **19** |
| **MP** | 10 | 0 | 8 | **18** |
| **L** | 8 | 0 | 8 | **16** |
| **Summa** | **310** | **0** | **39** | **349** |

### 4.2 Statutory Verification Matrix

The test suite in `tests/test_mandate_allocation.py` validates all statutory branches:

| Verification Target | Test Description | Status |
|---|---|---|
| **2022 Certified Election** | Full 29-constituency $\times$ 8-party phase reproduction | **PASS (0 mismatches)** |
| **2018 Certified Election** | Full 29-constituency $\times$ 8-party phase reproduction | **PASS (0 mismatches)** |
| **349 Seats Invariant** | Total seats strictly equals 349 across all branches | **PASS** |
| **4.0% National Threshold** | Exact inclusive $\ge 4.0\%$ boundary | **PASS** |
| **12.0% Constituency Threshold** | Exact inclusive $\ge 12.0\%$ local party qualification | **PASS** |
| **Sub-4% / Above-12% Fixture** | Sub-4% party wins local fixed seat, 0 adjustment seats, deducted from national pool | **PASS** |
| **Adjustment Divisor = 1.0** | Divisor is 1.0 (pure votes) when party has 0 fixed seats in constituency | **PASS** |
| **Gotland Return Prohibition** | Constituencies with $<3$ fixed seats cannot have seats retracted | **PASS** |
| **Overhang & Återföring** | Excess fixed seats retracted and reallocated correctly | **PASS** |
| **Multi-Constituency Returns** | Multi-constituency global quotient ranking and update | **PASS** |
| **Sub-4% Return Recipient** | Sub-4% local party eligible for returned fixed seat in constituency | **PASS** |
| **Exact Fraction Arithmetic** | Zero floating-point rounding or precision drift across all calculations | **PASS** |
| **Injected TieBreaker** | Pluggable lottery interface invoked upon exact equality | **PASS** |

The official Valmyndigheten Manual Example 5 is archived verbatim in `tests/fixtures/valmyndigheten_example_5_valkoping.json`. It is a three-constituency, 75-seat municipal example, not a 29-constituency Riksdag fixture; the production Riksdag allocator therefore records it as an external fixture limitation rather than transforming it into a synthetic oracle. Synthetic Riksdag return scenarios are labelled separately in the tests and freeze audit.

---

## 5. Usage & Makefile Commands

```bash
# Fetch raw official files from Valmyndigheten into data/raw/mandates/
make fetch-mandate-data

# Process raw files into normalized CSVs in data/processed/mandates/
make process-mandate-data

# Run the mandate allocation test suite
make test-mandate-allocation
```
