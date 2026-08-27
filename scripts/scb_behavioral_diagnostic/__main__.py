"""Main CLI entrypoint for scb_behavioral_diagnostic module.
"""
import sys
from scripts.scb_behavioral_diagnostic.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
