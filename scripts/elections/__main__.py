"""CLI entrypoint for scripts.elections package."""

import sys
from .pipeline import main

if __name__ == "__main__":
    sys.exit(main())
