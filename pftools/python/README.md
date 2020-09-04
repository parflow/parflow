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


5. At the end of your Python script, call the run() method:

        runname.run()


6. Once you have completed your input script, save and run it via the Python terminal
or command line:

        python3 runname.py

    This will print the data for your environment (ParFlow directory, ParFlow version,
    working directory, and the generated ParFlow database file) run the key validation,
    and run ParFlow. If ParFlow does not run successfully, it will also print out the
    contents of the out.txt file for your run.
    
## How to update this package

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
