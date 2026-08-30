"""Historically versioned Riksdag mandate-allocation law selection.

Two versions of Vallagen (2005:837) 14 kap. are relevant to this repository:

``POST_2018``
    The law in force since 1 January 2015 (SFS 2014:1384, prop. 2013/14:48),
    first applied at the **2018** general election and applicable to 2018, 2022
    and 2026. Modified Sainte-Laguë first divisor **1.2**; excess fixed
    constituency seats are **returned** (``återföring``, 14 kap. 4a–4c §§) and
    reallocated. **This is the production default and must stay so.**

``PRE_2018``
    The law in force for the **2010** and **2014** general elections (and
    earlier). Modified Sainte-Laguë first divisor **1.4**; there is **no**
    return mechanism. Under 3 kap. 8 § RF and the then-current 14 kap. 5 §
    vallagen, a party that won more fixed constituency seats than its
    nationwide proportional entitlement **keeps** those seats and is set aside,
    together with those seats, from the remaining adjustment-seat distribution,
    which is then carried out among the other participating parties so that
    *they* are proportional among themselves.

Legal provenance (archived under
``diagnostics/election_noise_v2/historical_evidence/raw/``):

* Prop. 2013/14:48 *Proportionell fördelning av mandat och förhandsanmälan av
  partier i val*, sha256 ``5aa84ff21840515a126928dd1300f847752ca2ed4c497ecc4115bee5bcf923cb``.
  - §4.1.4 "Regeringens förslag: Vid fördelningen av mandaten mellan partierna i
    riksdagsval ska den jämkade uddatalsmetoden tillämpas med 1,2 som första
    delningstal … i stället för 1,4."
  - §4.1.1 records the pre-reform rule: "det parti som blev överrepresenterat
    fick behålla de mandat som det fått i första omgången och att
    utjämningsmandaten fördelades mellan övriga deltagande partier så att dessa
    blev riksproportionellt representerade sinsemellan. Den reglering som
    beredningen föreslog … finns i nuvarande lagstiftning intagen i 3 kap. 8 §
    RF och 14 kap. 5 § vallagen."
  - §4.5 confirms the reform did **not** apply to the 2014 election.
* Enacted as SFS 2014:1384.

Only these two provisions differ for Riksdag elections. The 349-seat total, the
310 fixed / 39 adjustment split, the 4 % national threshold, the 12 %
constituency exception, the adjustment-seat placement divisors and the
resolution of ties by lot are unchanged across both versions and are therefore
shared code, not duplicated.

**Law version is never inferred from the wall clock.** A caller either passes the
version explicitly or derives it from the target election year via
:func:`mandate_law_for_election_year`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction


class MandateLaw(str, Enum):
    """Version of Vallagen 14 kap. governing a Riksdag mandate allocation."""

    POST_2018 = "POST_2018"
    PRE_2018 = "PRE_2018"


#: First general election at which SFS 2014:1384 applied.
FIRST_POST_2018_ELECTION_YEAR: int = 2018

#: Modified Sainte-Laguë first divisor per law version.
FIRST_DIVISOR_BY_LAW: dict[MandateLaw, Fraction] = {
    MandateLaw.POST_2018: Fraction(6, 5),  # 1.2
    MandateLaw.PRE_2018: Fraction(7, 5),   # 1.4
}


@dataclass(frozen=True)
class MandateLawConfig:
    """Complete, self-consistent legal configuration for one election year."""

    election_year: int
    law: MandateLaw
    first_divisor: Fraction
    statute: str

    @property
    def has_mandate_return(self) -> bool:
        """Whether excess fixed seats are returned and reallocated (14 kap. 4a-4c §§)."""
        return self.law is MandateLaw.POST_2018


def mandate_law_for_election_year(election_year: int) -> MandateLawConfig:
    """Deterministically map a target election year to the law in force for it.

    Never consults the current date. ``election_year`` is the year of the
    election being allocated, not the year the code is run.
    """
    if not isinstance(election_year, int) or isinstance(election_year, bool):
        raise TypeError(f"election_year must be an int, got {type(election_year)}")
    if election_year < 1970:
        raise ValueError(
            f"Election year {election_year} predates the 1970 unicameral Riksdag; "
            "no mandate-allocation law version is defined for it here."
        )

    if election_year >= FIRST_POST_2018_ELECTION_YEAR:
        law = MandateLaw.POST_2018
        statute = "Vallagen (2005:837) 14 kap. as amended by SFS 2014:1384 (first divisor 1.2; mandate return 14 kap. 4a-4c §§)"
    else:
        law = MandateLaw.PRE_2018
        statute = "Vallagen (2005:837) 14 kap. as in force before SFS 2014:1384 (first divisor 1.4; no mandate return; over-represented party keeps its fixed seats and is set aside from the adjustment distribution, 3 kap. 8 § RF)"

    return MandateLawConfig(
        election_year=election_year,
        law=law,
        first_divisor=FIRST_DIVISOR_BY_LAW[law],
        statute=statute,
    )
