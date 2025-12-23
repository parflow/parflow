# parflow/docker

Docker is a container technology that allows us to bundle the full software stack
into a virtual image of a Linux installation that can then be run into various
operating systems. In order to use Docker, you will need to have Docker installed
on your computer. For more information you can follow [this guide](https://docs.docker.com/get-docker/).

This directory captures the definition of various Docker images that could be
used for development, testing, and runtime.

These builds of ParFlow include Hypre, NetCDF, CLM, and Silo and Python 3
inside an Ubuntu system.

The __development__ images contain the source along with the build tree of all the
required components. This typically allows the user to use them for doing
incremental builds and testing new code changes when their machines don't allow
them to build ParFlow.

The __runtime__ image only provides the executables of the code, so simulations
can be run using those images without the extra weight of the build tree or
development tools required to compile the software.

To enable that directory in your build environment,
you will need to set `PARFLOW_ENABLE_DOCKER=ON`

## Building

We created convenient targets for you to use that will build those Docker images
for you. Depending on which CMake generator (__make/ninja__) you are using, you
should be able to run the following set of targets:

 - __DockerBuildDevelopment__
 - __DockerBuildRuntime__

## Testing

We also have helper targets to run the tests inside docker to see if everything
is working as expected.

 - __DockerTestDevelopment__
 - __DockerTestRuntime__

 For the runtime testing, we just picked a simple test but you could run one on
 your own with a command line similar to the one below:

        docker run --rm -v ~/Documents/code/Intern/parflow/test/python:/tests -it parflow-runtime /tests/base/van-genuchten-file/van-genuchten-file.py

## Running

Once the __runtime__ image is built, you can run the Docker image with the following
command which will automatically execute python3, which means you can provide the
python script that you want to run as argument (assuming the path is valid inside
the container):

        docker run --rm -it parflow-runtime

For example, if you want to run a script that exists in your current directory, you
could run the following:

        docker run --rm -v $PWD:/run -it parflow-runtime /run/my_script.py

If you want to have a shell and just run python3 manually inside that container,
you can execute the following in the command line:

        docker run --rm --entrypoint /bin/bash -it parflow-runtime
