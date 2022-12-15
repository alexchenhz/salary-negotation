# Applying Multi Agent Reinforcement Learning to Candidate/Employer Job Matching and Salary Negotiations

Alexander H. Chen

Yale University

CSEC 491 Senior Project

December 15, 2022

Thank you to Dr. James Glenn for advising me on this project.

## Project Description

## Abstract

## Final Project Report PDF

## Original Project Description

## Repository README

### Getting started

Activate the virtual environment.

```bash
source venv/bin/activate

```

Install the required packages.

```bash
pip install -r requirements.txt
```

Train the agents. See the `job_search.py` file for all CLI flags and args. Note this assumes the training is run on the Zoo with 16 CPU cores (using 4 workers + 1 local worker per trial, so requires 5 CPU cores per trial running in parallel). I also had issues running Ray on my M1 MacBook, so best to stick to x86 for now.

```bash
python job_search.py --num-candidates <int> --num-employers <int> --max-budget <int> --max-num-iters <int>
```

This will create a new directory `ray_results` which will store the information from training the reinforcement learning policies.

By default, this will use the TensorFlow job search model. Use TensorBoard to view the training metrics.

```bash
tensorboard --logdir ray_results/<path to results>
```
### File structure

├── archive
├── environment
│   ├── environment.py
│   ├── job_search_environment.py
│   └── __pycache__
├── example_1.py
├── example_2.py
├── index.md
├── job_search_agent_training.ipynb
├── job_search.py
├── job_search_simulation.py
├── models
│   ├── job_search_model.py
│   └── __pycache__
├── __pycache__
│   └── environment.cpython-38.pyc
├── ray_results
│   ├── job_search_env
│   └── tf2
├── README.md
├── requirements.txt
├── stdouts
│   ├── 10.out
│   ├── 11.out
│   ├── 12.out
│   ├── 13.out
│   ├── 14.out
│   ├── 15.out
│   ├── 16.out
│   ├── 17.out
│   ├── 18.out
│   ├── 19.out
│   ├── 1.out
│   ├── 20.out
│   ├── 21.out
│   ├── 22.out
│   ├── 23.out
│   ├── 24.out
│   ├── 25.out
│   ├── 26.out
│   ├── 27.out
│   ├── 2.out
│   ├── 3.out
│   ├── 4.out
│   ├── 5.out
│   ├── 6.out
│   ├── 7.out
│   ├── 8.out
│   ├── 9.out
│   ├── job_search2.out
│   ├── job_search.out
│   ├── nohup.out
│   └── tf_job_search3.out
└── venv
    ├── bin
    ├── etc
    ├── include
    ├── lib
    ├── pyvenv.cfg
    └── share

### Contributing