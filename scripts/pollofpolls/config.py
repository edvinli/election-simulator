"""Static, discovered Pollofpolls source configuration.

The URLs in this file were read from Pollofpolls page source on 2026-08-26;
they are not guessed URL patterns.  The party-file mapping is declared by the
AmCharts.loadFile call on each corresponding first-party party page.
"""

from dataclasses import dataclass


SITE_HOME = "http://pollofpolls.se/"
TIMESERIES_PAGE = (
    "http://pollofpolls.se/"
    "skattat-opinionslage-for-poll-of-polls-fran-och-med-15-september-2014/"
)
TIMESERIES_CSV = "http://pollofpolls.se/poll_img/data_table_tot.csv"
ARCHIVE_TIMESTAMP = "20260809000000"
SWEDISHPOLLS_CSV = (
    "https://raw.githubusercontent.com/MansMeg/SwedishPolls/master/Data/Polls.csv"
)
SWEDISHPOLLS_SOURCES_CSV = (
    "https://raw.githubusercontent.com/MansMeg/SwedishPolls/master/Sources/sources.csv"
)


@dataclass(frozen=True)
class Source:
    key: str
    url: str
    raw_filename: str
    kind: str
    party: str | None = None
    page_url: str | None = None
    archive_original_url: str | None = None
    allow_archive_fallback: bool = True

    @property
    def archive_url(self) -> str:
        original_url = self.archive_original_url or self.url
        return (
            "https://web.archive.org/web/"
            f"{ARCHIVE_TIMESTAMP}id_/{original_url}"
        )


SOURCES = (
    Source(
        key="timeseries",
        url=TIMESERIES_CSV,
        raw_filename="pollofpolls_timeseries_source.dat",
        kind="timeseries",
        page_url=TIMESERIES_PAGE,
        # Archive crawlers consistently preserved the HTML table even when the
        # separately linked CSV asset was not captured.
        archive_original_url=TIMESERIES_PAGE,
    ),
    Source(
        key="homepage",
        url=SITE_HOME,
        raw_filename="homepage.html",
        kind="homepage",
        page_url=SITE_HOME,
    ),
    Source(
        key="party_M",
        url="http://pollofpolls.se/poll_img/data_big_2.csv",
        raw_filename="party_M.csv",
        kind="party_chart",
        party="M",
        page_url="http://pollofpolls.se/opinionsutveckling-for-moderaterna/",
    ),
    Source(
        key="party_L",
        url="http://pollofpolls.se/poll_img/data_big_3.csv",
        raw_filename="party_L.csv",
        kind="party_chart",
        party="L",
        page_url="http://pollofpolls.se/liberalerna/",
    ),
    Source(
        key="party_C",
        url="http://pollofpolls.se/poll_img/data_big_4.csv",
        raw_filename="party_C.csv",
        kind="party_chart",
        party="C",
        page_url="http://pollofpolls.se/centerpartiet/",
    ),
    Source(
        key="party_KD",
        url="http://pollofpolls.se/poll_img/data_big_5.csv",
        raw_filename="party_KD.csv",
        kind="party_chart",
        party="KD",
        page_url="http://pollofpolls.se/kristdemokraterna/",
    ),
    Source(
        key="party_S",
        url="http://pollofpolls.se/poll_img/data_big_6.csv",
        raw_filename="party_S.csv",
        kind="party_chart",
        party="S",
        page_url="http://pollofpolls.se/socialdemokraterna/",
    ),
    Source(
        key="party_V",
        url="http://pollofpolls.se/poll_img/data_big_7.csv",
        raw_filename="party_V.csv",
        kind="party_chart",
        party="V",
        page_url="http://pollofpolls.se/vansterpartiet/",
    ),
    Source(
        key="party_MP",
        url="http://pollofpolls.se/poll_img/data_big_8.csv",
        raw_filename="party_MP.csv",
        kind="party_chart",
        party="MP",
        page_url="http://pollofpolls.se/miljopartiet/",
    ),
    Source(
        key="party_SD",
        url="http://pollofpolls.se/poll_img/data_big_9.csv",
        raw_filename="party_SD.csv",
        kind="party_chart",
        party="SD",
        page_url="http://pollofpolls.se/sverigedemokraterna/",
    ),
    Source(
        key="party_FI",
        url="http://pollofpolls.se/poll_img/data_big_10.csv",
        raw_filename="party_FI.csv",
        kind="party_chart",
        party="FI",
        page_url="http://pollofpolls.se/feministiskt-initiativ/",
    ),
    Source(
        key="swedishpolls",
        url=SWEDISHPOLLS_CSV,
        raw_filename="swedishpolls_polls.csv",
        kind="supplementary_individual",
        page_url="https://github.com/MansMeg/SwedishPolls/blob/master/Data/Polls.csv",
        allow_archive_fallback=False,
    ),
    Source(
        key="swedishpolls_sources",
        url=SWEDISHPOLLS_SOURCES_CSV,
        raw_filename="swedishpolls_sources.csv",
        kind="supplementary_provenance",
        page_url="https://github.com/MansMeg/SwedishPolls/blob/master/Sources/sources.csv",
        allow_archive_fallback=False,
    ),
)

SOURCE_BY_KEY = {source.key: source for source in SOURCES}
PARTY_SOURCES = tuple(source for source in SOURCES if source.kind == "party_chart")
