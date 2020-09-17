import os
import sys
import argparse

def tclToPython(inputFile, outputFile = None, runname = None):
  try:
    if runname is None:
      basename = os.path.basename(inputFile)
      lastIndex = basename.rindex('.')
      runname = basename[:lastIndex]

    if outputFile is None:
      lastIndex = inputFile.rindex('.')
      outputFile = f'{inputFile[:lastIndex]}.py'
  except:
    print(f'Invalid input file: {inputFile}')
    return

  if not os.path.exists(inputFile):
    print(f'Input file does not exist: {inputFile}')
    return

  runstr = str(runname) + '.'
  with open(inputFile, 'r') as fin:
    with open(outputFile, 'w') as fout:
      lines = fin.readlines()
      prevLine = ''
      for line in lines:
        newline = line
        if 'lappend auto_path $env(PARFLOW_DIR)/bin' in newline:
          newline = 'from parflow import Run\n'

        if 'package require parflow' in newline:
          newline = ''

        if 'namespace import Parflow::*' in newline:
          newline = f'{runname} = Run("{runname}", __file__)\n'

        if newline[0:6] == 'pfset ':
          newline = newline.replace('pfset ', runstr)
          newline_subs = newline.split()
          newline_subs[0] = newline_subs[0].replace('-', '_')
          if newline_subs[1][0].isalpha() or newline_subs[1][0] == "\"":
            newline = newline_subs[0] + ' = ' + "'" + ' '.join(newline_subs[1:]) + "'" + '\n'
            newline = newline.replace('-', '_').replace('\"', '').replace("'False'", "False").replace("'True'", "True")
          elif newline_subs[1][0] == '$' and len(newline_subs) == 2:
            newline = newline_subs[0] + ' = ' + newline_subs[1][1:] + '\n'
          else:
            newline = newline_subs[0] + ' = ' + ' '.join(newline_subs[1:]) + '\n'

        if newline[0:4] == 'set ' and 'runname' not in newline:
          newline = newline.replace('set ', '')
          newline_subs = newline.split()
          if newline_subs[1][0].isalpha():
            newline = newline_subs[0] + ' = ' + "'" + ' '.join(newline_subs[1:]) + "'" + '\n'
          else:
            newline = newline_subs[0] + ' = ' + ' '.join(newline_subs[1:]) + '\n'

        # commenting out all lines of code that haven't been edited yet
        if newline[0:1] != '#' and newline[0:1] != '\n' and newline == line:
          # testing for lines that continue to the next line
          if len(prevLine) >= 2 and prevLine[-2] == "\\":
            pass
          else:
            newline = '# ' + newline

        prevLine = newline

        fout.write(newline)

      fout.write(f'{runname}.run()\n')


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Parflow TCL script converter")
  parser.add_argument("--name", "-n",
                       default=None,
                       dest="name",
                       help="Name of the run to use")
  parser.add_argument("--input-file", "-i",
                       default=None,
                       dest="input",
                       help="Path to ParFlow TCL script to convert")
  parser.add_argument("--output-file", "-o",
                       default=None,
                       dest="output",
                       help="Python file path to use for writting the converted input")

  args = parser.parse_args()

  if args.input is None:
    parser.print_help()
    sys.exit(0)

  tclToPython(args.input, args.output, args.name)

