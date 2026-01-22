"""Command-line interface for JEDI."""

import argparse
import sys

from jedi import __version__


def main() -> int:
    """Main entry point for the JEDI CLI."""
    parser = argparse.ArgumentParser(
        prog="jedi",
        description="JEDI - Journalistic Excellence through "
                    "Domain-specific Intelligence",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    parser.parse_args()

    # Default action: print version
    print(f"jedi {__version__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
