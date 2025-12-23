import os
import sys
import argparse

from parflow.tools.io import read_pfidb, write_dict_as_pfidb
from parflow.tools.helper import sort_dict


def write_sorted_pfidb(input_file, output_file=None):
    if not os.path.exists(input_file):
        print(f"Input file does not exist: {input_file}")
        return

    if output_file is None:
        output_file = input_file

    write_dict_as_pfidb(sort_dict(read_pfidb(input_file)), output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parflow PFidb sorter")
    parser.add_argument(
        "--input-file",
        "-i",
        default=None,
        dest="input",
        help="Path to ParFlow database file to sort",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        default=None,
        dest="output",
        help="Output file path to write sorted result to",
    )
    args = parser.parse_args()

    if args.input is None:
        parser.print_help()
        sys.exit(0)

    write_sorted_pfidb(args.input, args.output)
