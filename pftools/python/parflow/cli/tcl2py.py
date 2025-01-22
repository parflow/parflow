import argparse
from pathlib import Path
import sys


def tcl_to_python(input_file, output_file=None, run_name=None):
    try:
        if run_name is None:
            run_name = Path(input_file).stem

        if output_file is None:
            output_file = str(Path(input_file).with_suffix(".py"))
    except Exception:
        print(f"Invalid input file: {input_file}")
        return

    if not Path(input_file).exists():
        print(f"Input file does not exist: {input_file}")
        return

    run_str = str(run_name) + "."
    with open(input_file, "r") as fin:
        with open(output_file, "w") as fout:
            prev_line = ""
            for original_line in fin:
                line = original_line
                if "lappend auto_path $env(PARFLOW_DIR)/bin" in line:
                    line = "from parflow import Run\n"

                if "package require parflow" in line:
                    line = ""

                if "namespace import Parflow::*" in line:
                    line = f'{run_name} = Run("{run_name}", __file__)\n'

                if line[0:6] == "pfset ":
                    line = line.replace("pfset ", run_str)
                    line_subs = line.split()
                    line_subs[0] = line_subs[0].replace("-", "_")
                    if line_subs[1][0].isalpha() or line_subs[1][0] == '"':
                        line = (
                            line_subs[0]
                            + " = "
                            + "'"
                            + " ".join(line_subs[1:])
                            + "'"
                            + "\n"
                        )
                        line = (
                            line.replace("-", "_")
                            .replace('"', "")
                            .replace("'False'", "False")
                            .replace("'True'", "True")
                        )
                    elif line_subs[1][0] == "$" and len(line_subs) == 2:
                        line = line_subs[0] + " = " + line_subs[1][1:] + "\n"
                    else:
                        line = line_subs[0] + " = " + " ".join(line_subs[1:]) + "\n"

                if line[0:4] == "set " and "run_name" not in line:
                    line = line.replace("set ", "")
                    line_subs = line.split()
                    if line_subs[1][0].isalpha():
                        line = (
                            line_subs[0]
                            + " = "
                            + "'"
                            + " ".join(line_subs[1:])
                            + "'"
                            + "\n"
                        )
                    else:
                        line = line_subs[0] + " = " + " ".join(line_subs[1:]) + "\n"

                # commenting out all lines of code that haven't been edited yet
                if line[0:1] != "#" and line[0:1] != "\n" and line == original_line:
                    # testing for lines that continue to the next line
                    if len(prev_line) >= 2 and prev_line[-2] == "\\":
                        pass
                    else:
                        line = "# " + line

                prev_line = line

                fout.write(line)

            fout.write(f"{run_name}.run()\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parflow TCL script converter")
    parser.add_argument(
        "--name", "-n", default=None, dest="name", help="Name of the run to use"
    )
    parser.add_argument(
        "--input-file",
        "-i",
        default=None,
        dest="input",
        help="Path to ParFlow TCL script to convert",
    )
    parser.add_argument(
        "--output-file",
        "-o",
        default=None,
        dest="output",
        help="Python file path to use for writting the " "converted input",
    )

    args = parser.parse_args()

    if args.input is None:
        parser.print_help()
        sys.exit(0)

    tcl_to_python(args.input, args.output, args.name)
