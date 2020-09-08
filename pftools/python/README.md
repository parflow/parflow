# pftools

This is a package to run ParFlow via a Python interface. This package allows 
the user to build a script in Python that builds the database (.pfidb file) 
which ParFlow reads as input.

## How to use this package

1. Install with the following command:

        pip install pftools

2. Open a new Python script in your favorite text editor or IDE.

    - You can find example Python scripts in the main ParFlow repo under
        */parflow/test/python/*


3. At the top of the script, make sure you include the following lines:

        from parflow import Run
        runname = Run("runname", __file__)

    This imports the package and initializes your run as the object "runname"


4. Set your desired keys and values on your ParFlow run object, such as:

        runname.FileVersion = 4

    Note: for user-defined key names, make sure you've defined the names before
    you use them as a key name. For example:

        runname.GeomInput.Names = 'domain_input'

    needs to be set before:

        runname.GeomInput.domain_input.InputType = 'SolidFile'


5. After you have assigned values to your keys, you can call multiple methods on your
ParFlow run object:

    - `validate()`: This will validate your set of key/value pairs and print validation
    messages. This does not require ParFlow.
    - `write(file_name=None, file_format='pfidb')`: This will write your key/value
    pairs to a file with your choice of format (default is the ParFlow database `pfidb`
    format). Other acceptable formats passed as the `file_format` argument include 
    `yml`, `yaml`, and `json`. This method does not require ParFlow. 
    - `clone(name)`: This will generate a clone object of your run with the given `name`.
    See `parflow/test/python/new_features/serial_runs/serial_runs.py` for an example of
    how to use this. 
    - `run(working_directory=None, skip_validation=False)`: This will execute the 
    `write()` method. If `skip_validation` is set to `False`, it will also execute the 
    `validate()` method. The `working_directory` can be defined as an argument if you
    would like to change the directory where the output files will be written, but it 
    defaults to the directory of the Python script. Finally, `run()` will execute 
    ParFlow. This will print the data for your environment (ParFlow directory, 
    ParFlow version, working directory, and the generated ParFlow database file).
    If ParFlow runs successfully, you will get a message `ParFlow ran successfully`. 
    Otherwise, you will get a message `ParFlow run failed.` followed by a print of the 
    contents of the `runname.out.txt` file. 


6. Once you have completed your input script, save and run it via the Python terminal
or command line:

        python3 runname.py
        
    You can append one or more of the following arguments to the run:
    
    - `--parflow-directory [None]`: overrides environment variable for 
    `$PARFLOW_DIR`.
    - `--parflow-version [None]`: overrides the sourced version of ParFlow used to validate
    keys. 
    - `--working-directory [None]`: overrides the working directory for the ParFlow run.
    This is identical to specifying `working_directory` in the `run()` method.
    - `--skip-validation [False]`: skips the `validate()` method if set to `True`. This is 
    identical to specifying `skip_validation` in the `run()` method.
    - `--show-line-error [False]`: shows the line where an error occurs when set to `True`.
    - `--exit-on-error [False]`: causes the run to exit whenever it encounters an error when
    set to `True`.
    - `--write-yaml [False]`: writes the key/value pairs to a yaml file when set to `True`. 
    This is identical to calling the method `runname.write(file_format='yaml)`.
    - `-p [0]`: overrides the value for `Process.Topology.P` (must be an integer).
    - `-q [0]`: overrides the value for `Process.Topology.Q` (must be an integer).
    - `-r [0]`: overrides the value for `Process.Topology.R` (must be an integer).
    
## How to update this package (developers only)

This assumes that you are using CMake with the pftools package as it is 
contained within the main ParFlow repo (see https://github.com/parflow/parflow)

1. Update the version number in setup.py

2. Run the following command to create and test a source archive and a wheel 
   distribution of the package: 

        make PythonCreatePackage

3. If the distributions pass, run the following command to publish the
   distributions: 

        make PythonPublishPackage
        
4. Check PyPI to make sure your package update was published correctly. Thanks
   for contributing!

## Getting help

If you have any issues or questions about the code, please refer to one of the
following options:

   - User mailing list: https://mailman.mines.edu/mailman/listinfo/parflow-users
   - ParFlow blog: http://parflow.blogspot.com
   - GitHub repo (for tracking issues and requesting features): https://github.com/parflow/parflow
