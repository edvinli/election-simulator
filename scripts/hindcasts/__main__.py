"""CLI entry point for scripts.hindcasts package."""

import sys
from .hindcast import main

if __name__ == "__main__":
    sys.exit(main())
