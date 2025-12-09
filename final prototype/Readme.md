# ISSIP Synthetic Data final prototype

This is meant to run on a tabular dataset, producing synthetic data ready for machine learning usage. It will work for other purposes, but the scoring is assuming a machine learning application for measuring accuracy of produced data.

This project was tested primarily on Ubuntu, however other operation systems are supported with the docker image which was briefly tested on both Windows and Mac

The general steps to run the project are as follows:

## 1. Installing docker
### Mac
with brew, run
> brew install docker

Then install the desktop application for docker on mac

### Windows
Download Docker Desktop for Windows from the official docker website 

### Linux
For Ubuntu
> sudo apt install docker

For other distributions, use own package manager

## 2. Build docker image
In the terminal, cd into this final prototype directory, and run (as administrator if needed)
> docker build -t synth-docker .

## 3. Running project with docker
Use config.yaml to configure surrounding settings such as iterations, path to dataset, and so on, it should be relatively self-explanatory.

Notably, the project needs a target column(s) input, which it will train a model to predict as a method of testing utility of synthetic data.

Once configured, run (as administrator if needed)
> docker run -v ./output:/output --rm synth-docker

To save the output when running in docker, the output directory is mounted by default to /output in the docker image, configure the synthetic data to write to that directory
Depending on operating system you may have to make slight changes to the format for the path being mounted to.

Currently, the projecr is set up to use a sample dataset, which can be used as an example for setup.

## 4. Making changes
If needed, hyperparameter optimization can be configured in optimize.py, where the ranges of values can be tweaked if needed.

Optimizer should be able to narrow down the best fit for the dataset, however during the training a higher scoring configuration might have been found in the process, these ranges can be adjusted to forcibly test a configuraton if desired.
