"""Cross-repository mirroring of certified static publication generations.

This package is intentionally separate from the statistical publication
pipeline.  It never simulates, never commits, and never pushes; it only
copies an already-certified generation into a website repository and then
writes the consumer pointer.
"""

from .publisher import (
    GENERATION_FILES,
    SITE_PUBLICATION_RELATIVE,
    SitePublishError,
    publish_generation_to_site,
)

__all__ = [
    "GENERATION_FILES",
    "SITE_PUBLICATION_RELATIVE",
    "SitePublishError",
    "publish_generation_to_site",
]
