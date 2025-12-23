import os

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------

WORKING_DIRECTORY = os.getcwd()
PRINT_LINE_ERROR = False
EXIT_ON_ERROR = False
PARFLOW_VERSION = "3.6.0"


# ---------------------------------------------------------------------------


def get_working_directory():
    return WORKING_DIRECTORY


# ---------------------------------------------------------------------------


def set_working_directory(new_working_directory=None):
    """This will set the working directory to use for all the
    relative file path.
    """
    global WORKING_DIRECTORY

    if new_working_directory:
        WORKING_DIRECTORY = new_working_directory
    else:
        WORKING_DIRECTORY = os.getcwd()


# ---------------------------------------------------------------------------


def enable_line_error():
    """Calling that method will enable line feedback on validation
    error
    """
    global PRINT_LINE_ERROR
    PRINT_LINE_ERROR = True


def disable_line_error():
    """Calling that method will disable line feedback on validation
    error
    """
    global EXIT_ON_ERROR
    EXIT_ON_ERROR = False


def enable_exit_error():
    """Calling that method will force the program to exit on
    validation error.
    """
    global EXIT_ON_ERROR
    EXIT_ON_ERROR = True


def disable_exit_error():
    """Calling that method will force the program to not exit on
    validation error.
    """
    global EXIT_ON_ERROR
    EXIT_ON_ERROR = False


def set_parflow_version(version):
    """Globally store the ParFlow version to test against"""
    global PARFLOW_VERSION
    PARFLOW_VERSION = version
