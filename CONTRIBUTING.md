# Contributing to ParFlow

This file is heavily inspired by Atom's CONTRIBUTING.md file.

:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

The following is a set of guidelines for contributing to ParFlow.
ParFlow is hosted on [ParFLow Github](https://github.com/parflow)
site. These are mostly guidelines, not rules. Use your best judgment,
and feel free to propose changes to this document in a pull request.

#### Table Of Contents

[I just have a question, where can I ask it?](#I have a question)

[How Can I Contribute?](#how-can-i-contribute)
  * [Reporting Bugs](#reporting-bugs)
  * [Suggesting Enhancements](#suggesting-enhancements)
  * [Your First Code Contribution](#your-first-code-contribution)
  * [Pull Requests](#pull-requests)

[What should I know before I get started?](#what-should-i-know-before-i-get-started)

[Styleguides](#styleguides)
  * [Git Commit Messages](#git-commit-messages)
  * [JavaScript Styleguide](#javascript-styleguide)
  * [CoffeeScript Styleguide](#coffeescript-styleguide)
  * [Specs Styleguide](#specs-styleguide)
  * [Documentation Styleguide](#documentation-styleguide)

[Additional Notes](#additional-notes)
  * [Issue and Pull Request Labels](#issue-and-pull-request-labels)

## I have a question

> **Note:** Please don't file an issue to ask a question. You'll get
faster results by using the resources below.  Issues are meant for bug
reports and feature requests.

The [ParFlow email
list](https://mailman.mines.edu/mailman/listinfo/parflow-users) is
seen by many ParFlow users and developers.  You can also subscribe to
the email list.  The traffic is minimal and is the platform for
discussing ParFlow questions.

The ParFlow GitHub wiki is where the team is starting to assemble a list of frequently asked questions:
- [Parflow FAQ](https://github.com/parflow/parflow/wiki/ParFlow-FAQ)

A ParFlow blog is available with notes from users on how to compile and use ParFlow:
- [Parflow Blog](http://parflow.blogspot.com/)

To report Parflow bugs or make change requests, please use the GitHub issue tracker for Parflow:
- [Parflow Issue Tracker](https://github.com/parflow/parflow/issues)

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report for
Parflow. Following these guidelines helps maintainers and the community
understand your report :pencil:, reproduce the behavior :computer:
:computer:, and find related reports :mag_right:.

Before creating bug reports, please check [this
list](#before-submitting-a-bug-report) as you might find out that you
don't need to create one. When you are creating a bug report, please
[include as many details as
possible](#how-do-i-submit-a-good-bug-report). Fill out [the required
template](ISSUE_TEMPLATE.md), the information it asks for helps us
resolve issues faster.

#### Before Submitting A Bug Report

* **Check the [FAQs](https://github.com/parflow/parflow/wiki/ParFlow-FAQ)** for a list of common questions and problems.
* **Perform a [cursory search](https://github.com/parflow/parflow/issues)** to see if the problem has already been reported. If it has **and the issue is still open**, add a comment to the existing issue instead of opening a new one.

> **Note:** If you find a **Closed** issue that seems like it is the
same thing that you're experiencing, open a new issue and include a
link to the original issue in the body of your new one.

#### How Do I Submit A (Good) Bug Report?

Bugs are tracked as [GitHub
issues](https://guides.github.com/features/issues/). Create an issue and provide the following
information by filling in [the template](ISSUE_TEMPLATE.md).

Explain the problem and include additional details to help maintainers reproduce the problem:

* **Use a clear and descriptive title** for the issue to identify the problem.
* **Describe the exact steps which reproduce the problem** in as many details as possible. 
* **Providing specific examples of Parflow input that demonstrate the problem helps enormously in debugging**. Include links to files or GitHub projects, or copy/paste-able snippets, which you use in those examples.  Developers unable to recreate a problem makes it very hard to fix.
* **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior.
* **Explain which behavior you expected to see instead and why.**
* **If you're reporting that Parflow crashed**, include a stack trace if available. 
* **If you're reporting that CMake doesn't configure or Parflow doesn't compile**, include the output from the CMake configure and the compiler output.  "It doesn't compile" is not enough information.
* **Large files** can be attached to the issue via a [file attachment](https://help.github.com/articles/file-attachments-on-issues-and-pull-requests/), or put it in a [gist](https://gist.github.com/) and provide link to that gist.

Provide more context by answering these questions:

* **Did the problem start happening recently** (e.g. after updating to a new version of Parflow) or was this always a problem?
* **Can you reliably reproduce the issue?** If not, provide details about how often the problem happens and under which conditions it normally happens.

Include details about your configuration and environment:

* **Which version of Parflow are you using?** You can get the exact version by running `parflow -v`.
* **What OS and version are you using**?
* **What compiler and version are you using**?
* **What OS and version are you using**?

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion
for Parflow, including completely new features and minor improvements to
existing functionality. Following these guidelines helps maintainers
and the community understand your suggestion :pencil: and find related
suggestions :mag_right:.

Before creating enhancement suggestions, please check [this
list](#before-submitting-an-enhancement-suggestion) as you might find
out that you don't need to create one. When you are creating an
enhancement suggestion, please [include as many details as
possible](#how-do-i-submit-a-good-enhancement-suggestion). Fill in
[the template](ISSUE_TEMPLATE.md), including the steps that you
imagine you would take if the feature you're requesting existed.

#### Before Submitting An Enhancement Suggestion

* **Perform a [cursory search](https://github.com/parflow/parflow/issues)** to see if the problem has already been reported. If it has **and the issue is still open**, add a comment to the existing issue instead of opening a new one.

* **Perform a [cursory search](https://github.com/parflow/parflow/issues)** to see if the enhancement has already been suggested. If it has, add a comment to the existing issue instead of opening a new one.

#### How Do I Submit A (Good) Enhancement Suggestion?

Enhancement suggestions are tracked as [GitHub issues](https://guides.github.com/features/issues/). Create an issue and provide the following information:

* **Use a clear and descriptive title** for the issue to identify the suggestion.
* **Provide a step-by-step description of the suggested enhancement** in as many details as possible.
* **Provide specific examples to demonstrate the steps**. Include copy/paste-able snippets which you use in those examples, as [Markdown code blocks](https://help.github.com/articles/markdown-basics/#multiple-lines).
* **Describe the current behavior** and **explain which behavior you expected to see instead** and why.
* **Explain why this enhancement would be useful** to most Parflow users.

## What should I know before I get started in the code?

### Code organization

The Parflow is split into two main parts.  The pftools directory
contains the input parsing (written using TCL) and the pfsimulator
directory which contains the simulator code.  Parflow is written as a
library so it can be embedded inside other applications; the code for
the library is in in the pfsimulator/parflow_lib directory.  For
stand-along execution an executable ('main') is also supplied; the
code for the stand-alone application is in pfsimulator/parflow_exe.
Parflow can be linked to CLM; the code for for CLM is in
pfsimulator/clm.  Parflow has an embedded a copy of KINSOL which is in
the pfsimulator/kinsol directory.

Parflow was written before the MPI standard was created.  In order to
easily port to vendor specific message passing libraries a
communication layer was created called Another Message Passing Sytem
(AMPS).  We continue to use AMPS since it has proved useful for easily
porting Parflow to use with other frameworks, like [OASIS 3](
https://journals.ametsoc.org/doi/10.1175/MWR-D-14-00029.1).  The AMPS
library is in pfsimulator/amps.

The pftools directory contains the TCL interface and a verity of tools
for pre and post processing.  

Additional documentation can be found in the [Parflow Users
Manaul](https://github.com/parflow/parflow/blob/master/parflow-manual.pdf)

### Parflow Input and Input Database

Parflow was written to be principally driven by an input file written in
TCL.  This was done to enable a much more feature rich input file than
a standard XML/JSON file.  Expressions can be used for computing
parameters and for loops can be used for easier setup up large numbers
of similar input elements.  The TCL was also meant to capture the
problem workflow in an executable script for repeat-ability.  For
example, data post-processing could be part of a standard input script.
Jupyter Notebooks or similar systems are examples of the intent;
Parflow was written before the current workflow technologies existed
so we used a scripting language.  TCL was chosen since it was widely
used at the time.

A Parflow input file is simply a set of key/value assignments which go
into a 'Parflow Input Database'.  The pfset commands in the TCL file
set the key/value pairs.  This database is saved to a simple text file
with the key/values when the pfrun command is executed in the TCL
script.  The '<runame>.pfidb' file is read by the main Parflow
executable to setup a problem.  The 'pfidb' file is similar in intent
to a JSON file (we would likely use JSON if starting today).

This input dataset file (pfidb file) is used by the main parflow
executable to configure and setup a problem.  The keys are read by
each ParFlow module to setup a problem.  The key names are a
structured hierarchy separated by '.'.  When adding new modules, new
keys can be added to the input database to configure the new module.

### Parflow Modules

Parflow modules are a way to provide a bit of object oriented
structure in C.  A Parflow module is simply a structures with
function pointers and data.  The module has a public and instance
structure.  The basic use for public is functions/data that are
instance independent.   Instance data is used for functions/data 
that are specific to an instance.  Frequently Parflow only has a
single instance of a module so this distinction does not matter.  It
is still a good idea to split things out to future proof the
implementation in case multiple instance are desired at some point.

The module pattern provides methods for constructing and destructing
the instances.  Different kinds of modules have specific methods for
the class of module that they are (e.g. linear solver).

For anyone with an object oriented background this should all sound
pretty familiar.  Since C doesn't have the language constructs it
makes the infrastructure for supporting OO more visible (and error
prone).

## ParFlow Conventions

Most of ParFlow is written in ANSI C.  ParFlow has been compiled with
C++ and used with a C++ library (the AMR framework SAMRAI) so should
be usable with C++.  CLM is written in FORTRAN.

### Indentation and whitespace

Parflow doesn't have a full coding style guide document; follow the
existing coding style to when doing a code submission.

Code contributions should NOT do indentation/whitespace changes
intermixed with bug fixes/feature additions.  Large amounts of white
spaces changes makes it hard for reviewers to see what is actually
being changed.  Looking at 20 changed files for a 1 line of a bug fix
is not reviewer friendly.

If you have indentation/whitespace changes, please submit those as a
separate pull request so they can be easily reviewed.

ParFlow uses Uncrustify to perform automatic code indentation/spacing.
The bin/pfuncrustify script is supplied to run the uncrustify tool on
the Parflow source code.  The Uncrustify setup is in the
bin/parflow.cfg file.  Code contributions should use the the
pfuncrustify tool to maintain consistency.

### Your First Code Contribution

See the [ParFlow Software Productivity and Sustainability
Plan](https://github.com/parflow/parflow/blob/master/pftools/docs/software_plan.md)
for notes on how to make code submissions (section 3.1) and
documentation (section 3.3).



