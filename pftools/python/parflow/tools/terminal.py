# -*- coding: utf-8 -*-
"""Terminal helper

This module provides colors and symbols for generating nice terminal output.

"""


class Colors:
    """Terminal color helper
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'


class Symbols:
    """Terminal emoji helper
    """
    ok = u'\u2714'
    ko = u'\u2718'
    errorItem = u'\u2605'
    warning = u'\u26A0'
    default = u'\u2622'
    droplet = u'\U0001F4A7'
    splash = u'\U0001F4A6'
    x = u'\u274C'
