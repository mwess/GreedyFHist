.. _installation:

============
Installation
============

At the moment we have verified installation for GreedyFHist only on linux system. Once we get ahold of Mac and Windows, we will test this as well.

------------
Requirements
------------

GreedyFHist is dependent on several Python packages which are all listed in ``pyproject.toml``. Otherwise, the only external dependency is the greedy software ``greedy``. 

--------------------
Installing using pip
--------------------

GreedyFHist can be installed by running the following command:

.. code-block:: bash

    pip install --user git+https://github.com/mwess/greedyfhist@v0.0.2-rc2


----------------------------
Building GreedyFHist locally
----------------------------

GreedyFHist can be build locally using ``poetry``.

.. code-block:: bash

    poetry build
    pip install dist/greedyfhist-*.tar.gz

------------
Using Docker
------------

We also provided a Dockerfile for installing ``GreedyFHist``.


-----------------
Installing Greedy
-----------------

GreedyFHist has one external dependency, which is ``greedy`` (https://sites.google.com/view/greedyreg/about).

``greedy`` can be installed using the official instructions. It also comes with ITK-SNAP (http://www.itksnap.org). 

Installing Greedy via Docker
============================

If ``greedy`` cannot be installed using the official documentation, we have also provided a Dockerfile. 

Either build the Dockerfile locally and set an alias (for instance in your ``~/.bash.rc`` file).

.. code-block:: bash

    cd docker
    docker build -f Dockerfile_greedy -t greedy .

Note, that currently GreedyFHist cannot be used from the commandline if `greedy` is accessed as an external docker container.
For now, please refer to the interactive API.

Alternatively, a docker-image for ``greedyfhist`` can be downloaded:

.. code-block:: bash

    docker pull mwess89/greedyfhist:0.0.2-rc2

Or the docker image can be built as well:

.. code-block:: bash

    cd docker
    docker build -t greedyfhist -f Dockerfile_greedyfhist .
