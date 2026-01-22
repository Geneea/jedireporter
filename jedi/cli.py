"""Command-line interface for JEDI."""

import argparse
import sys
from importlib.metadata import version


def main() -> int:
    """Main entry point for the JEDI CLI."""
    _version = version("jedi")

    parser = argparse.ArgumentParser(
        prog="jedi",
        description="JEDI - Journalistic Excellence through "
                    "Domain-specific Intelligence",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_version}",
    )

    parser.parse_args()

    # Default action: print version
    print(f"jedi {_version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
