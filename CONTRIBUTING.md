# Contributing to ParFlow

This file is inspired by Atom's CONTRIBUTING.md file.

Thanks for taking the time to contribute!

The following is a set of guidelines for contributing to ParFlow.
ParFlow is hosted on [ParFLow Github](https://github.com/parflow)
site. These are mostly guidelines, not rules. Use your best judgment,
and feel free to propose changes to this document in a pull request.

If you are new to contributing to open source projects there are many
general guides available to understand the basic ideas and processes.
A large collection is available
[here](https://github.com/freeCodeCamp/how-to-contribute-to-open-source).

#### Table Of Contents

[I just have a question, where can I ask it?](#i-have-a-question)

[What should I know before I get started?](#what-should-i-know-before-i-get-started)

[How Can I Contribute?](#how-can-i-contribute)
  * [Reporting Bugs](#reporting-bugs)
  * [Suggesting Enhancements](#suggesting-enhancements)
  * [Your First Code Contribution](#your-first-code-contribution)
  * [Pull Requests](#pull-requests)

## I have a question

> **Note:** Please don't file an issue to ask a question. You'll get
faster results by using the resources below.  Issues are meant for bug
reports and feature requests.

The [ParFlow-Users](https://groups.google.com/g/parflow) is
seen by many ParFlow users and developers.  You can also subscribe to
the email list.  The traffic is minimal and is the platform for
discussing ParFlow questions.

The ParFlow GitHub wiki is where the team is starting to assemble a list of frequently asked questions:
- [ParFlow FAQ](https://github.com/parflow/parflow/wiki/ParFlow-FAQ)

A ParFlow blog is available with notes from users on how to compile and use ParFlow:
- [ParFlow Blog](http://parflow.blogspot.com/)

To report ParFlow bugs or make change requests, please use the GitHub issue tracker for ParFlow:
- [ParFlow Issue Tracker](https://github.com/parflow/parflow/issues)

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report for
ParFlow. Following these guidelines helps maintainers and the community
understand your report, reproduce the behavior, and find related reports.

Before creating bug reports, please check [this
list](#before-submitting-a-bug-report) as you might find out that you
don't need to create one. When you are creating a bug report, please
[include as many details as
possible](#how-do-i-submit-a-good-bug-report).

#### Before Submitting A Bug Report

* **Check the [FAQs](https://github.com/parflow/parflow/wiki/ParFlow-FAQ)** for a list of common questions and problems.
* **Perform a [cursory search](https://github.com/parflow/parflow/issues)** to see if the problem has already been reported. If it has **and the issue is still open**, add a comment to the existing issue instead of opening a new one.

> **Note:** If you find a **Closed** issue that seems like it is the
same thing that you're experiencing, open a new issue and include a
link to the original issue in the body of your new one.

#### How Do I Submit A (Good) Bug Report?

Bugs are tracked as [GitHub
issues](https://guides.github.com/features/issues/). Create an issue
and as much detail as possible.  Explain the problem and include
additional details to help maintainers reproduce the problem:

* **Use a clear and descriptive title** for the issue to identify the problem.
* **Describe the exact steps which reproduce the problem** in as many details as possible. 
* **Providing specific examples of ParFlow input that demonstrate the problem helps enormously in debugging**. Include links to files or GitHub projects, or copy/paste-able snippets, which you use in those examples.  Developers unable to recreate a problem makes it very hard to fix.
* **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior.
* **Explain which behavior you expected to see instead and why.**
* **If you're reporting that ParFlow crashed**, include a stack trace if available. 
* **If you're reporting that CMake doesn't configure or ParFlow doesn't compile**, include the output from the CMake configure and the compiler output.  "It doesn't compile" is not enough information for the team to help determine what has gone wrong.
* **Large files** can be attached to the issue via a [file attachment](https://help.github.com/articles/file-attachments-on-issues-and-pull-requests/), or put it in a [gist](https://gist.github.com/) and provide link to that gist.

Provide more context by answering these questions:

* **Did the problem start happening recently** (e.g. after updating to a new version of ParFlow) or was this always a problem?
* **Can you reliably reproduce the issue?** If not, provide details about how often the problem happens and under which conditions it normally happens.

Include details about your configuration and environment:

* **Which version of ParFlow are you using?** You can get the exact version by running `parflow -v`.
* **What OS and version are you using**?
* **What compiler and version are you using**?
* **What OS and version are you using**?

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion
for ParFlow, including completely new features and minor improvements to
existing functionality. Following these guidelines helps maintainers
and the community understand your suggestion and find related
suggestions.

Before creating enhancement suggestions, please check [this
list](#before-submitting-an-enhancement-suggestion) as you might find
out that you don't need to create one. When you are creating an
enhancement suggestion, please [include as many details as
possible](#how-do-i-submit-a-good-enhancement-suggestion).

#### Before Submitting An Enhancement Suggestion

* **Perform a [cursory search](https://github.com/parflow/parflow/issues)** to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.

#### How Do I Submit A (Good) Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues](https://guides.github.com/features/issues/). Create an issue and provide the following information:

* **Use a clear and descriptive title** for the issue to identify the suggestion.
* **Provide a step-by-step description of the suggested enhancement** in as many details as possible.
* **Provide specific examples to demonstrate the steps**. Include copy/paste-able snippets which you use in those examples, as [Markdown code blocks](https://help.github.com/articles/markdown-basics/#multiple-lines).
* **Describe the current behavior** and **explain which behavior you expected to see instead** and why.
* **Explain why this enhancement would be useful** to most ParFlow users.

## What should I know before I get started in the code?

### Code organization

The ParFlow is split into two main parts.  The pftools directory
contains the input parsing (written using Python or TCL) and the
pfsimulator directory which contains the simulator code.  The
stand-along executable code ('main') is in pfsimulator/parflow\_exe.
Most of ParFlow is written as a library so it can be embedded inside other
applications; the code for the library is in in the
pfsimulator/parflow_lib directory. ParFlow can optionally be linked
with CLM; the code for for CLM is in pfsimulator/clm.  ParFlow has an
embedded a copy of KINSOL which is in the pfsimulator/kinsol
directory.

ParFlow was written before the MPI standard was created.  In order to
easily port to vendor specific message passing libraries a
communication layer was created called Another Message Passing System
(AMPS).  We continue to use AMPS since it has proved useful for easily
porting ParFlow to use with other frameworks, like [OASIS 3](
https://journals.ametsoc.org/doi/10.1175/MWR-D-14-00029.1).  The AMPS
library is in pfsimulator/amps.

The pftools directory contains the Python and TCL interfaces and a
verity of tools for pre/post processing input/output files.  The
normal workflow for ParFlow is to use the pftools interface (Python or
TCL) to create an input for the ParFlow executable (these files use
the '.pfidb' file extension by convention).

Additional documentation can be found in the [ParFlow Users
Manual](https://github.com/parflow/parflow/blob/master/parflow-manual.pdf)

### ParFlow Input and Input Database

ParFlow was written to be principally driven by an input file written
in Python or TCL.  This was done to enable a much more feature rich
input file than a standard XML/JSON file.  Expressions can be used for
computing parameters and for loops can be used for easier setup up
large numbers of similar input elements.  The Python/TCL scripts were
also meant to capture the problem workflow in an documented script for
repeatability.  For example, data post-processing could be part of a
standard input script.  Jupyter Notebooks or similar systems are
examples of the intent; ParFlow was written before the current
workflow technologies existed so we used a scripting language.  TCL
was initially chosen since it was widely used at the time and a Python
interface has been added.

A ParFlow input file is simply a set of key/value assignments which go
into a 'ParFlow Input Database'.  The pfset commands in the TCL file
set the key/value pairs.  The Python input is a bit more structured,
see the users manual and test cases for examples.  This database is
saved to a simple text file with the key/values when the pfrun command
is executed in the TCL script.  The '<runame>.pfidb' file is read by
the main ParFlow executable to setup a problem.  The 'pfidb' file is
similar in intent to a JSON file (we would likely use JSON if starting
today).

This input dataset file (pfidb file) is used by the main parflow
executable to configure and setup a problem.  The keys are read by
each ParFlow module to setup a problem.  The key names are a
structured hierarchy separated by '.'.  When adding new modules, new
keys can be added to the input database to configure the new module.

### ParFlow Modules

ParFlow modules are a way to provide a bit of object oriented
structure in C.  ParFlow was writtenb before C++ was widely available
on HPC systems.  A ParFlow module is simply a structure with function
pointers and data.  The module has a public and instance structure.
The basic use for public is functions/data that are instance
independent (similar to class methods/members in C++).  Instance data
is used for functions/data that are specific to an instance.  In
theory one can have many instances of a specific type, this is used in
a few places in ParFlow but most modules only have a single instance.
It is still a good idea to split things out to future proof the
implementation in case multiple instance are desired at some point.

The module pattern provides methods for constructing and destructing
the instances.  Different kinds of modules have specific methods for
the class of module that they are (e.g. linear solver).

For anyone with an object oriented background this should all sound
pretty familiar.  Since C doesn't have the language constructs it
makes the infrastructure for supporting OO more visible, manual and
error prone.

## Automatic style checking

ParFlow uses automated code formatters to check code as one of the
checks in our CI system.  Incorrectly formatted code will fail the
check and won't be merged.  The check is done using Uncrustify for C
and Black for Python.  A script is provided to automatically reformat 
code and do the check.

To check if the coding style is correct run the pfformat script at the 
root of the PF source tree:

```
cd parflow
./bin/pfformat --check
```

Checking is done with the same script:

```
cd parflow
./bin/pfformat
```

You must have installed Uncrustify version 0.79.0 in order for this to
work.  The Python Black code formatter will be installed during the
ParFlow installation so you don't need to manually install Black.

Information and source on these tools can be found here:

[Uncrustify](https://github.com/uncrustify/uncrustify)

[Black](https://github.com/psf/black)

### Why auto code formatting?

Enforcing a common code indentation and other style format is fairly
common on Open Source projects.  Everyone has different preferred
styles and has setup their editors with different options.  Reviewing
and merging code conflicts that have divered primarily due to code
formatting creates unnecessary headaches and makes it harder to
understand what 'real' changes have occurred.

## ParFlow C Conventions

Most of ParFlow is written in ANSI C.  ParFlow has been compiled with
C++ and used with a C++ library (the AMR framework SAMRAI) so should
be usable with C++.  CLM is written in FORTRAN.

### Indentation and whitespace

ParFlow doesn't have a full coding style guide document; follow the
existing coding style to when doing a code submission.

ParFlow uses Uncrustify to perform automatic code indentation/spacing.
The bin/pfuncrustify script is supplied to run the uncrustify tool on
the ParFlow source code.  The Uncrustify setup is in the
bin/parflow.cfg file.  Code contributions should use the the
pfformat tool to maintain consistency.

#### Source Code Documentation

ParFlow has historically had inadaquate source code documentation.   We are
working to address this by requiring new code submissions to have
documentation to aid others in understanding the code.  The team has
selected to use Doxygen for code documentation.

All new source code file header files should document EVERY user
visible structure, function, and macro.   Reviewers will reject
submissions that do not do this.

Target comments at new users to the code, focus on intent and purpose
of the method/structure.

Format for documenting a function/macro:

```c
/** @brief brief single line description of function.
 * 
 * Longer description of function
 *
 * @param arg1 description of arg1
 * @param arg2 description of arg2
 * ...
 * @param argN description of argN
 *
 * @return description of return value
 */
```

All structures should have documentation for the purpose of the structure and a description of the fields
in the structure.

```c
/**
 * @brief brief single line description of the structure.
 *
 * Longer description of structure
 *
 */
typedef struct mystruct {
  /** Field1 description */
  unsigned char field1;

  /** Field2 description */
  unsigned char faces;

} MyStruct;
```

In addition, it is expected that code sections are documented with
functionality information and explanations of any nonintuitive code.
Documentation information should be set off from the code through
standard C language delimiters using /* and */.  You don't need to
state the obvious, focus on intent and purpose to help guide others
reading your code.

## ParFlow Python Conventions

### Python Coding Style Guidelines

ParFlow tries to follow several different style guides when writing 
Python.

[PEP-8](https://peps.python.org/pep-0008/) guidelines should be used as a
base for naming conventions, whitespace and other basic coding style
issues.

[Googlestyle](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstringsa)
Python docstrings formatting should be used.  Please document every
method, ParFlow has not done this historically and it has made the
code more difficult to maintain.  We are documenting now so don't follow
the bad practices from our past.

Using the [Black](https://pypi.org/project/black/) code formatter is
enforced in the CI process for checking pull requests.

## Your First Code Contribution

The ParFlow reviewers will be looking for several specific items
on submitted code contributions.

#### License and Copyright

ParFlow is released under the GNU Lesser General Public License
(LGPL).  The full text of the license is included in the ParFlow
LICENSE.txt file located at:
<https://github.com/parflow/parflow/blob/master/LICENSE.txt>.  All
contributions to ParFlow must be compatible with the LGPL.  It is
strongly preferred to use GNU General Public License version 2.1 for
contributions since other licenses will have to be checked to see if
including them is allowed.  If a contributor has a particular reason
to provide new work with a license other than GPLv2, they should
contact the code maintainers before contributing a pull request.

All new source files must contain a license statement, and all
modifications to source files must conform to the the license in the
original file.  You must make sure that the licensing attribution is
correct and that the code is suitable for ParFLow inclusion. No
contributions will be accepted if they include code (even snippets)
from sources that have incompatible licenses.

In addition to the license you should include a copyright
statement in the header if that is required by your employer, such as:
Copyright 2009 John Doe.  If the you work for an employer or a
university, the you should check with how to properly identify the
copyright holder (which may not be the contributor as an individual).

Please use a header similar to this example in ALL new code files.

We will not accept code without a proper license statement.

```c
/**********************************************************************
 *
 *  Your Copyright statement if needed.
 *
 *  Please read the LICENSE file for the GNU Lesser General Public License.
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License (as published
 *  by the Free Software Foundation) version 2.1 dated February 1999.
 *
 *  This program is distributed in the hope that it will be useful, but
 *  WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
 *  and conditions of the GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public
 *  License along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
 *  USA
 ***********************************************************************/

```

#### Testing

> **Note:** ParFlow will not accept Pull Requests that do not pass ALL tests

Contributions with feature additions should include tests for that
feature.  Submissions that do not pass the test suite will not be
accepted.  A major goal is to ensure the master branch of ParFlow
always builds and successfully runs the test suite.

You can run the tests by going to the directory where you ran CMake
and running:

```
make check
```

The test suite will run using CTest and results will show how many
tests pass and fail.

Automated ParFlow testing is done with every pull request using the
Github Action system.  The results will appear on the pull request
page on GitHub.  You can view the CI results by selecting "Show all
checks".  The test system runs serial as well as parallel problems.
The current ParFlow regression test suite is limited but includes
tests on saturated and unsaturated subsurface flow and coupled ParFlow
CLM systems.  The GitHub Action test is run in a Linux image based on
Ubuntu.  The testing setup can be found in the
[linux.yml](/.github/workflows/linux.yml) file.

### Pull Requests

ParFlow uses a standard GitHub Fork-Branch-Pull workflow.  If you are
new to GitHub workflows the [quickstart
guide](https://docs.github.com/en/get-started/quickstart/contributing-to-projects)
will walk you through the process.

ParFlow uses this workflow to support several goals:

- Engage the community
- Ensure ParFlow does not break
- Maintain ParFlow's quality
- Fix problems that are important to users
- Provide a sustainable mechanism for accepting code contributions
