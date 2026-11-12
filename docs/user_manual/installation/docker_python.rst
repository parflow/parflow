Docker Container with Python Packages
=====================================

This is a step-by-step guide for running ParFlow in a Docker container that includes Python packages (pftools and HydroFrame packages subsettools and hf_hydrodata).

1. Make sure you have the latest version of Docker Desktop. Go to `Docker <https://www.docker.com/products/docker-desktop/>`_ and download Docker Desktop. Make sure you download the correct version based on your computer’s operating system and architecture. Follow the instructions to install Docker Desktop.

.. image:: https://raw.githubusercontent.com/hydroframe/parflow_short_course_updated/main/Docker.png
   :alt: Screenshot

2. Launch Docker Desktop. The application should be running while you do the next steps. (You might need to create a Docker account and sign in if you don’t already have one.)

3. Start a new terminal session (outside Docker) to type the commands in the next two steps:

4. Pull the parflow/subsettools image from DockerHub. Choose the correct version based on your computer’s architecture.

**NOTE:** *If your Docker is running out of space, you might need to use `docker system prune <https://docs.docker.com/engine/reference/commandline/system_prune/>`_ with the appropriate options to clear out old containers and make space for the new one.*

* For the x86_64/amd64 (Intel Chip for Mac or Windows) architecture:

.. code-block:: bash

   docker pull george135/subsettools_amd64-june2025

* For the arm64 (M1/M2 chip for Mac) architecture:

.. code-block:: bash

   docker pull george135/subsettools_arm64-june2025

5. Once the image has finished downloading, you can run the container with:

* For the x86_64/amd64 architecture:

.. code-block:: bash

   docker run -dp 8888:8888 george135/subsettools_amd64-june2025:latest start-notebook.sh --NotebookApp.token=''

* For the arm64 architecture:

.. code-block:: bash

   docker run -dp 8888:8888 george135/subsettools_arm64-june2025:latest start-notebook.sh --NotebookApp.token=''

6. Use a browser to navigate to your `JupyterLab container <http://localhost:8888/lab?>`_ or use the link that will appear next to your container on the Docker Desktop application:

.. image:: https://raw.githubusercontent.com/hydroframe/parflow_short_course_updated/main/Docker2.png
   :alt: Screenshot

7. You should see a JupyterLab environment like this:

.. image:: https://raw.githubusercontent.com/hydroframe/parflow_short_course_updated/main/Docker3.png
   :alt: Screenshot

8. Click on the Terminal application to start a terminal session *inside* the container.

9. Clone the ParFlow short course GitHub repository:

.. code-block:: bash

   git clone https://github.com/hydroframe/parflow_short_course_updated.git

10. Navigate to the ``gridding`` folder and click on the ``box_domain_setup_full.ipynb`` notebook.

11. Make sure the notebook runs successfully without errors.

.. image:: https://raw.githubusercontent.com/hydroframe/parflow_short_course_updated/main/Docker4.png
   :alt: Screenshot

12. Congratulations, you're all setup!
