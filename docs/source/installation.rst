.. _installation:

============
Installation
============

At the moment we have verified installation for GreedyFHist only on linux system. Once we get ahold of Mac and Windows, we will test this as well.


Requirements
============

GreedyFHist is dependent on several Python packages which are all listed in ``pyproject.toml``. Otherwise, one external dependency needs to be provided, ``greedy``. 


Installing 
==========

GreedyFHist can be installed locally by running the following command::

    pip install --user git+https://github.com/mwess/greedyfhist@master


Installing GreedyFHist locally
==============================

To build GreedyFHist locally, we use ``poetry``::

    poetry build
    pip install dist/greedyfhist-*.tar.gz

Docker
======

We also provided a Dockerfile for installing ``GreedyFHist``.



Installing Greedy
=================

GreedyFHist has one external dependency, which is ``greedy`` (https://sites.google.com/view/greedyreg/about).

``greedy`` can be installed using the official instructions. It also comes with ITK-SNAP (http://www.itksnap.org). 

Installing Greedy via Docker
----------------------------

If ``greedy`` cannot be installed using the official documentation, we have also provided a Dockerfile. 

Either build the Dockerfile locally and set an alias (for instance in your ``~/.bash.rc`` file)::

    cd docker
    docker build -f Dockerfile_greedy -t greedy .

Setting the alias::

    alias greedy='docker run greedy'

Alternatively, a docker-image for ``greedy`` can be downloaded::

    docker ...
