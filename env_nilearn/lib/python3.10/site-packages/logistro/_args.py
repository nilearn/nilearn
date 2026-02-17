import argparse
import sys

# Verify arg sanity
if "--logistro-human" in sys.argv and "--logistro-structured" in sys.argv:
    raise ValueError(
        "Choose either '--logistro-human' or '--logistro-structured'.",
    )

parser: argparse.ArgumentParser = argparse.ArgumentParser(add_help=False)
"""
The argsparse parser is exported if you'd like to include it as a parent in your own
`argparse.ArgumentParser` and thereby getting better help messages.
"""

parser.add_argument(
    "--logistro-human",
    action="store_true",
    dest="human",
    default=True,
    help="Format the logs for humans",
)
parser.add_argument(
    "--logistro-structured",
    action="store_false",
    dest="human",
    help="Format the logs as JSON",
)

parser.add_argument(
    "--logistro-level",
    default=None,
    type=str,
    dest="log",
    help="Set the logging level (no default, fallback to system default)",
)

# Get the Format
parsed, remaining_args = parser.parse_known_args()
