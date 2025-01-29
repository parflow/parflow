# ParFlow Software Productivity and Sustainability Plan

## Version 1.0

#### Carol S. Woodward(1), Reed M. Maxwell(2), Steve G. Smith(1), Stefan Kollet(3), Daniel Osei-Kuffuor(1), Carsten Burstedde(3)

(1) Lawrence Livermore National Laboratory

(2) Colorado School of Mines

(3) University of Bonn, Germany


## 1. Introduction

This document describes the development and change process used for the open-source, 
object-oriented, parallel watershed flow model, 
ParFlow (<https://inside.mines.edu/~rmaxwell/maxwell_software.shtml>).  ParFlow is an 
integrated, parallel watershed model that makes use of high-performance computing to
simulate surface and subsurface fluid flow. The goal of the ParFlow project is to
enable detailed simulations for use in the assessment and management of groundwater 
and surface water, to investigate system physics and feedbacks and to understand 
interactions at a range of scales.  This is a living document, and its markdown 
source is located in the docs directory of the ParFlow github repo: 
<https://github.com/parflow/parflow/tree/master/docs/software_plan.md>.

ParFlow includes fully-integrated overland flow, the ability to simulate complex 
topography, geology and heterogeneity and coupled land-surface processes including 
the land-energy budget, and biogeochemistry and snow (via CLM). It is multi-platform 
and runs with a common I/O structure from laptop to supercomputer. ParFlow is the 
result of a long, multi-institutional development history and is now a collaborative 
effort between CSM, FZ Juelich, Uni Bonn, LLNL, WSU, LBL, and LTHE.

Development on ParFlow was initiated in the early 1990s at Lawrence Livermore National 
Laboratory (LLNL).  ParFlow was initially designed to be a saturated flow code with 
transport capability.  In the mid 90s, a variably saturated flow module was added, 
and in 2006 overland flow was added.  At the same time, the Common Land Model was 
integrated as a subroutine simulating the terrestrial water and energy cycles at the 
land surface. Subsequently, ParFlow was interfaced with various atmospheric codes, 
including ARPS and WRF, and is also part of the Terrestrial Systems Modeling 
Platform (TerrSysMP) integrated platform.  In 2013 a semi-structured, terrain following 
transform was incorporated into the code.  The code was released from LLNL in the
mid-2000s, and development continues principally in Colorado at the School of Mines 
and in Germany at FZ-Jülich and the University of Bonn.

The following plan describes the processes adopted for development of ParFlow.  
It does not describe the technical work or details of the models and methods used 
within the simulator.


## 2. Process for New Features or Capability Enhancements

Before commencing development of new features or enhancing existing capabilities 
within the ParFlow code suite, requirements of the new code as well as the design 
approach for implementing it must be documented.  These Requirements and Design 
documents will be stored in the github repository so that all team members will 
be able to access them and contribute as appropriate.  The Requirements and Design 
documents are maintained in the ParFlow repository at 
<https://github.com/parflow/parflow/tree/master/docs/design_documents>.

The git issue tracker and pull request model will be used to facilitate and 
record discussions on these documents and the proposed changes.  Once discussion 
has taken place, designated Design Reviewers will determine whether the change 
will be accepted.  Once the new code is developed, the primary author will issue 
a git pull request.  A designated reviewer will evaluate the submitted code and, 
if acceptable, merge it into the repo.   If not, the author will be notified of 
any concerns for follow up.  The various steps in this process are detailed below.


### 2.1. Requirements and Design Document

The Requirements and Design (R&D) Document should include all goals of the new 
code, targeted use cases, and any constraints.  Furthermore, test problems that 
target the proposed new capability must also be described in the R&D document.
The Requirements and Design Document is viewed as a living document that 
evolves with the team’s understanding of its tasks and needs. 
A template for the R&D document is located at:
<https://github.com/parflow/parflow/blob/master/pftools/design_documents/0000-template.md>.

A requirement should be expressed succinctly in the form of some definite 
statement, and some set of victory conditions that specify when a requirement 
has been met. Requirements should be specific enough that team members can 
agree on what is meant in a technical sense, but need not articulate a specific 
approach if they can be met another way.  The victory conditions of a 
requirement should be comprehensible to anyone familiar with the nature of 
the project and should be revised until they are specific and concrete enough 
for everyone to agree on their meaning.

A good design describes a software application or system in terms of its duties 
and describes these duties in terms of the various pieces, or “entities” within 
the software (modules, functions, data structures, algorithms, and so forth). 
A design for ParFlow might consist of a set of diagrams of various modules and 
their relationships, accompanied by text descriptions of these modules, and a 
set of operations and interactions between them. 


### 2.2. Requirements and Design Review

Before fully reviewing the proposed changes to the code, the R&D document should 
be placed in the repo, and the proposed changes communicated to the development 
team.  Development team members should look through the document and discuss 
any concerns via the git issue tracker.  The issue tracker will enable the 
discussion to be referenced in the future.  

All proposed capabilities or enhancements will be reviewed by the ParFlow 
Capability Reviewers, Reed Maxwell or Stefan Kollet.  Criteria for review 
include the following.

* *General usefulness:* Will the proposed capability or enhancement be useful to 
more than one user?

* *Address single issue:* When possible enhancements should address single issues. 
Multiple orthogonal changes should be addressed in multiple contributions. This 
follows a common practice of open source teams; it makes reviewing changes 
easier and the code change history is cleaner.

* *Adherence to overarching goals of the ParFlow project:* Does the proposed 
capability or enhancement further the progress or the ParFlow code toward the 
goals set out above?

* *Impact to ParFlow code design:* Will the new capability or enhancement be 
added to the code without unreasonably altering the software architecture?  

* *Test problem specified:* Does the proposed enhancement or capability have 
an identified way of being tested that is discussed in the R&D document?


### 2.3. Pull Request Review

Once development of a new capability or enhancement is complete, the branch 
must be submitted as a pull request.  The new code will be evaluated and, 
if approved, merged into the main ParFlow repo.  Pull request evaluators 
are: Reed Maxwell, Stefan Kollet, Klaus Goergen, Steve Smith, and Carol Woodward. 
Criteria used for evaluation include the following.

* *Design:* Does the new code adhere to the design specified in the R&D document?  
If not, how is it different and does it still pass muster against the R&D review 
criteria?

* *Building:* Does the new code compile on a defined set of architectures/systems?

* *Testing:* Does the new code pass all regression tests successfully on multiple 
platforms?

* *Tests of new capability:* Does the new code include a test for correctness of 
the new capability or enhancement?

* *Documentation:* Is the new code itself documented and have there been additions 
to the User and/or Developer manuals that reflect the new capabilities?  Do the 
new manual versions go through LaTex without additional warnings


## 3. Source Code Management and Availability

This section describes the processes and tools used to develop and change the 
ParFlow simulator.


### 3.1. Use of git Version Control System

ParFlow development has been version controlled for the duration of the 
simulator’s evolution.  ParFlow was initiated using an RCS version control 
system on an LLNL server, and, in 2009, the ParFlow repository was moved 
from LLNL to a private repository on a public CVS server.  In 2016 the 
repository was moved to a public space on github (<https://github.com/parflow>), 
where it now resides.  

Parflow uses the standard Github Forking workflow.   The Forking workflow is a 
common workflow that originated with Github and is used by many open source 
projects.   The workflow enables any contributor to easily submit changes; no 
special permissions are required other than a Github account.   Contributors 
create a fork (copy) of the main repository, make changes, and submit a pull 
request for review and merging by the ParFlow team.  The workflow also enables 
contributors to keep their local forks up-to-date with the main repository. 
The workflow is supported by the Github web interface (for creating a fork 
and submitting a change for review) and any of the git command line or GUI 
tools.Short tutorials on the Github Forking workflow can be found in  
<https://gist.github.com/Chaser324/ce0505fbed06b947d962> and in 

<http://blog.scottlowe.org/2015/01/27/using-fork-branch-git-workflow/>.

The basic workflow process includes the following steps.  Note that details of 
how each of these are done can be found at the above links.

* Create a fork of the ParFlow repository.
* For each new feature, create a topic branch in the forked repository from 
the current trunk on which development for that feature will be done.  Branch 
names should reflect the feature being developed on the branch.

* Develop the new capability within the topic branch and conduct regular 
testing to verify all regression tests run successfully.

* Once development is done and the new feature or capability is ready to be 
submitted for a pull request, rebase the topic branch on the current master 
so that an ensuing merge will not have conflicts with the master in the pull 
request.

* It may be advised to merge the master branch into the topic branch whenever 
the master has been significantly updated to catch conflicts early. This requires 
to fetch and rebase/merge of the topic trunk branch on a regular basis.

* Once the master branch is merged into the topic branch, run the test suite 
along with any new test programs.

* If tests are ALL successful, push all changes to GitHub, go to the page for 
the forked repo on GitHub, select the topic branch, and click the pull request 
button.  Once the pull request is done, the ParFlow automated test  system will 
run through all regression tests in the test suite.  The success or failure of 
this run will be made known to the pull request reviewer.

When a pull request addresses one or more issues that are tracked in
the GitHub issue tracker, the final commit message should indicate
which issue(s) are being addressed by the pull request.  This
practices enables easy access to the reason for the change and any
discussion history.  Limit the first line of the commit message to 72
characters or less (this formats better when looking at the list of
commits).  In the commit message use one of the standard GitHub
keywords, for example:
 
Closes #issueid

Fixes #issueid

Resolves #issueid
 
For more than one issue each needs to be listed separately:

closes #issueid1, closes #issueid2, and closes #issueidN.  

For more details on good practices for commit messages, see 
<https://help.github.com/articles/closing-issues-via-commit-messages/>.

### 3.2. Long-Living Technical Tasks

Long-living technical tasks (in which work is done for weeks or months 
without merging work to the master branch) are strongly discouraged.  Because 
code changes in different branches are isolated from one another, they 
necessarily become less coherent with each other over time. If this decoherence 
is allowed to continue for a prolonged period, the process of merging changes 
into the master branch (and in doing so, of resolving incompatible differences) 
can become intractable. Therefore, it is desirable to minimize the amount of 
time spent developing a single feature or set of bugfixes in a branch. This 
means that features and fixes should be scoped and staged so that they can 
be performed in a relatively short period of time, on the order of days to 
a week, instead of weeks to months or years.


### 3.3. Documentation

Documentation of the ParFlow code is done in two main ways.  First, all 
source code is documented, and second, a number of manuals are distributed 
along with the code.


#### 3.3.1 Source Code Documentation

Every file of source code in ParFlow contains a header with license and copyright 
information.  Although the current source is not fully documented, we expect 
that any new source code file include in its header a listing of all routines 
or structures defined and the capability goals of the code in that file.  In 
addition, it is expected that code sections are documented with functionality 
information and explanations of any nonintuitive code.  Documentation information 
should be set off from the code through standard C language delimiters 
using /* and */.

While it is not a current practice on existing ParFlow code, using Doxygen 
formatted comments is highly encouraged for new functions that are added.   
The recommended format is:

/** @brief brief single line description of function.
 
  Longer description of function

  @param arg1 description of arg1

  @return description of return value

 */


#### 3.3.2 Documentation in Manuals

ParFlow also includes three main manuals for users and developers.  These 
manuals are located in the “pftools” directory under a docs 
subdirectory, <https://github.com/parflow/parflow/tree/master/docs>.  
All manuals are written in the LaTex word processing language.  **_It is expected 
that any new capability or feature be documented in these manuals as appropriate._**
The most critical manual is the ParFlow User Manual, which describes all input and 
output information needed to run ParFlow with all capabilities and to get state 
information out.  The Developer Manual includes information about the mathematical 
models within ParFlow and the algorithms used in their implementation.  It is 
expected that models underpinning new features be described in this manual. 

Of lesser importance, the AMPS Manual is a third manual documenting the message 
passing wrapping layer.   When ParFlow was originally developed, MPI did not exist; 
ParFlow followed the common practice of having an application-specific set of 
wrappers around each machine’s native message passing interfaces.  These wrappers 
comprise AMPS, Another Message Passing System.   


### 3.4. License and Copyright

ParFlow is released under the GNU Lesser General Public License (LGPL).  The 
full text of the license is included in the ParFlow LICENSE.txt file located 
at: <https://github.com/parflow/parflow/blob/master/LICENSE.txt>.   All 
contributions to ParFlow must be compatible with the LGPL.  It is preferred 
to use GNU General Public License version 2.1 for contributions since other 
licenses will have to be checked to see if including them is allowed.  If a 
contributor has a particular reason to provide new work with a license other 
than GPLv2, they should contact the code maintainers before contributing a 
pull request.    

All new source files must contain a license statement, and all modifications to 
source files must conform to the the license in the original file.   Contributors 
must make sure that the licensing attribution is correct and that the code is 
suitable for ParFLow inclusion. No contributions will be accepted if they 
include code (even snippets) from sources that have incompatible licenses.

In addition to the license contributors should include a copyright statement 
in the header, such as: Copyright 2009 John Doe.   If the contributor works 
for an employer or a university, the contributor should check with their 
employer as to how to properly identify the copyright holder (which may not 
be the contributor as an individual).  As an example the LLNL copyright 
statement is shown here:

Copyright (c) 1995-2009, Lawrence Livermore National Security LLC.

Produced at the Lawrence Livermore National Laboratory.

Written by the Parflow Team (see the CONTRIBUTORS file)

CODE-OCEC-08-103. All rights reserved.

Parflow is released under the GNU General Public License version 2.1

For details and restrictions, please read the LICENSE.txt file.


### 3.5. Bug Reporting, Issue Management, and Feature Requests

The Github issue tracker for ParFlow should be used for reporting bugs and 
also for proposing new features or enhancements, 
<https://github.com/parflow/parflow/issues>.  When possible, please refer 
to specific files and and modules when reporting bugs.

The issue workflow for developers follows the standard GitHub Forking model 
outlined previously with minor additions to link the pull request with an issue 
using a tag in the commit message: 

When a committer decides to start work on resolving an issue in the project, 
they should first assign the issue to themselves to record that the issue is 
being worked on.  The standard  GitHub Forking model is used to prepare the 
code to address the issue. The last commit in the history should indicate 
the issue number (such as Fixed #issueid).  A standard pull request is made. 
When the pull request is accepted the issue will be automatically closed 
by GitHub.


### 3.6. Code Contributions from Outside the Development Team

Contributions from outside the core development team are welcome.  However, we 
expect any contributor to adhere to the following guidelines:

* Contributed code should be useful to more than one user.

* Contributed code should be in support of the goals of the ParFlow simulation system.

* Contributed code should not require unreasonable alterations of the code 
architecture.    

* Contributed code compiles cleanly with the -Wall compiler option.

* A test program that verifies correct results of the functionality of any 
contributed code must be included with the contribution.  

* A new test case to be included in the ParFlow regression test suite must be provided.

* Contributed code must pass all current ParFlow regression tests plus any 
additional ones provided with the code on all targeted computational platforms.

* All contributed code must be documented, including additions to the User 
and Developer Manuals as appropriate.  Modified Developer and User manuals 
must run through LaTex cleanly.

* When possible, code contributions should address a single change in each 
pull request.   This is a common practice on open source teams to make 
reviewing easier and makes code history easier to follow.

When possible, new contributors should supply a Requirements and Design 
document describing any proposed code changes or additions.  This document 
should be placed in the ParFlow git repository, and an issue filed in the 
ParFlow issue tracker (see above) for discussion among the ParFlow developer 
team and user community.  


## 4. ParFlow Software Testing

Automated ParFlow testing is done with every pull request using the TravisCI 
continuous integration system.  This system runs tests on serial as well as 
parallel problems.  The current ParFlow regression test suite is limited but 
includes tests on saturated and unsaturated subsurface flow and coupled 
ParFlow CLM systems.  The TravisCI systems uses Ubuntu and a Ubuntu-based 
build done at the time the testing commences.  The testing setup is included 
in the .travis.yaml file located in the root ParFlow directory in the repository.

Currently, an infrequent manual test is done to ensure the ParFlow code compiles 
cleanly with the -Wall compiler flag.

While this testing system is useful, it could be improved through greater test 
case coverage of the code and expansion of the various ParFlow configurations 
tested (with/without hypre, with/without silo/HDF5, different I/O models, etc).
In addition, the team would like to set up a regular regression testing capability.


## 5. External Software Package Processes

ParFlow currently has a number of external package dependencies. Most of 
them are compiled externally, and ParFlow links to them. Right now, each 
one is included in a package-specific manner.  The packages and main handling 
of the dependence is:

* KINSOL nonlinear solver - source code from 1999 is contained in the ParFlow 
source. Plans include updating interfaces to the current version, removing 
this source, and setting up the build to link to the current library.

* KINSOL nonlinear solver - the adaptive mesh refinement module in ParFlow that 
is based on the SAMRAI library uses the current version of KINSOL through a 
link to an externally built library.

* Hypre PFMG and SMG preconditioners - link to an externally built version of 
the hypre library.   Specific version requirements should be stated in the 
main ParFlow README..

* Silo - link to an externally built library.  ParFlow does not depend on a 
specific version.

* HDF5 - link to an externally built library.  ParFlow does not depend on a 
specific version.

* MPI (MPI1) - link to an externally built library.  ParFlow does not depend 
on a specific version beyond that it provides MPI1 capabilities.

* SAMRAI -  this software library is compiled externally and ParFlow optionally 
links to it in the master branch. SAMRAI use was placed into ParFlow in an 
experimental mode, and current viability is unknown.

* p4est - this software library is compiled externally and ParFlow links to 
it in the branch, adaptive, that is under consideration for being merged. 
p4est is free software under the GNU GPL version 2 or later, which means that 
the ParFlow executable when linked against p4est may be distributed under the 
terms of the GNU GPL. Work on and distribution of the ParFlow source code is 
not affected.

* Crunchflow- this code can be called as a library for solving reaction network 
kinetics for a specified geochemical database.  This code is needed for the 
reactive transport ParFlow branch.


## 6. User Support

The ParFlow development team actively engages the user community.  We 
encourage  users to submit bugs and feature requests via the Github issue 
tracker for ParFlow: Parflow Issue Tracker.  In addition, the ParFlow team 
maintains a blog dedicated to use and development of ParFlow.  The blog is 
located here: <http://parflow.blogspot.com/>.  The blog is designed to 
take commonly asked 
questions from the listserve (below) and the user community to provide 
extended answers, how-to type tutorials and additional code information 
(including discussion of recent papers, etc).  There is also a ParFlow 
website under development that will be located at: <http://www.parflow.org>.
The website will be hosted by the Juelich Research Centre.

Lastly, the team maintains a user email list where questions or
concerns can be posted.  The list is available here:
[Parflow-Users](https://groups.google.com/g/parflow).  The list is
archived on the Google Groups site for searching pevious questions.

## 7. ParFlow Developer Training

Currently, there exist a number of different training materials, such as 
tutorials, short courses, the developers manual, and the ParFlow blog 
(<http://parflow.blogspot.com/>). In addition, live trainings are offered during 
conferences such as MODFLOW & More and in dedicated summer and fall schools. 
The materials will be made available via the ParFlow website (parflow.org), 
github, and the mailing list. In the future, a dedicated virtual training 
centre will be established as a one-stop shop, where all the training materials 
will be available including YouTube documents and videos. Additionally, 
yearly ParFlow developer meetings are planned with the next meeting at 
Colorado School of Mines in 2018. 
