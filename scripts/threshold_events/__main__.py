"""Main CLI entrypoint for threshold_events module.
"""
import sys
from scripts.threshold_events.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
