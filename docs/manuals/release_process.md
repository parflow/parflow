# ParFlow release Process

The release process for ParFlow follows the a standard GitHub
release process.  The following steps should be followed for creating a new
ParFlow release.

* Create branch/fork for generating the release
* Edit the RELEASE-NOTES.md file
* Regenerate the ParFlow User Manual
* Commit release and version file changes
* Create a pull request for the branch/fork
* Merge the pull request
 Generate a release on GitHub

## Create branch

Use standard Git/GitHub commands to create a branch for editing some files for the release.

```shell
VERSION=v3.13.0
git clone git@github.com:parflow/parflow.git
git checkout -b $VERSION
```

## Edit files

Edit the RELEASE-NOTES.md file to add notes about what was changed in
this release.  Notes should should be appended.  PF has changed to appending
release changes to the RELEASE notes.

Edit `./VERSION` file with current version.

Edit `./docs/user_Manual/conf.py` and update version number.

### Update pftools version number 

Edit `./pftools/python/pyproject.toml` and increment the version number.

## Commit release file changes

Use standard git add and git commit commands to add the modified files
to the release branch/fork.

```shell
git add -u :/
git commit -m "Update release files"
```

## Create a pull request for the branch/fork

Use GitHub to create a pull request for the release branch.
  
## Merge the pull request

Use GitHub to create a pull request for the release branch.
  
## Generate a release on GitHub

On the GitHub [Parflow Releases](https://github.com/parflow/parflow/releases)
page use ``Draft a new release'' to create the release.

Version tag should have format of `vX.Z.Z` version.  Release title
should have format of `ParFlow Version X.Y.Z`.  The GitHub release
description can be copied from the release notes markdown file that
was created in a prior step.

## Generate Docker

```shell
  docker build -t <hub-user>/<repo-name>[:<tag>]
  docker push <hub-user>/<repo-name>:<tag>
```

Example using podman to build and push

```shell
  VERSION=3.13.0
  podman build -t docker.io/parflow/parflow:version-${VERSION} .
  podman login docker.io
  podman push docker.io/parflow/parflow:version-${VERSION}
  podman tag docker.io/parflow/parflow:version-${VERSION} docker.io/parflow/parflow:latest
  podman push  docker.io/parflow/parflow:latest
```

## Update pftools on PiPy

### Build 

Make sure that Python is enabled through the `PARFLOW_ENABLE_PYTHON` option.

```shell
mkdir build
cd build
cmake .. -D PARFLOW_ENABLE_PYTHON=TRUE
```

### Create Python Package

Run the following command to create and test a source archive and a wheel
distribution of the package. Make sure you are running this command in an 
environment with the `twine` Python package installed.

```shell
make PythonCreatePackage
```
### Publish package

If the distributions pass, run the following command to publish the
distributions. In order to run this command successfully, you must first set the
`TWINE_USERNAME` and `TWINE_PASSWORD` environment variables to the username
and password that you will use to authenticate with PyPI.

```shell
make PythonPublishPackage
```

### Check PyPI

Check PyPI to make sure your package update was published correctly.

## Check ParFlow User Manual

The manual update should be automated by ReadTheDocs.   Check that a new version has been uploaded after the 
release has been merged on GitHub.
  


