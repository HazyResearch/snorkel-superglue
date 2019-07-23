# snorkel-superglue
Applying Snorkel to SuperGLUE

This repository includes a demonstration of how to use the [Snorkel](https://github.com/HazyResearch/snorkel) library to achieve a state-of-the-art score on the [SuperGLUE](https://super.gluebenchmark.com/) benchmark.
The specific code used to create the submission on the leaderboard is hosted in the [emmental-tutorials](https://github.com/SenWu/emmental-tutorials/tree/master/superglue) repository.
This repository contains a refactored version of that code made compatible with the Snorkel API for general exploration.

Best Reference:
**[Blog post](https://hazyresearch.github.io/snorkel/blog/superglue.html)**

## Installation
To use this repository:
1. [Optional] Create a virtual environment and activate it:

    ```
    virtualenv .env
    source .env/bin/activate
    ```

2. Install requirements:

    ```
    pip install -r requirements.txt
    ```

3. Set the environment variable $SUPERGLUEDATA that points to the directory where the data will be stored. We recommend using a directory called `data/` at the root of the repo) by running:

    ```
    export SUPERGLUEDATA=$(pwd)/data/
    ```

4. Download the SuperGLUE data by running:

    ```
    bash download_superglue_data.sh $SUPERGLUEDATA
    ```

This will download the data for the primary SuperGLUE tasks as well as the SWAG dataset used for pretraining COPA.
To obtain the MNLI dataset for pretraining RTE and CB, we recommend referring to the starter code for the [GLUE](https://gluebenchmark.com/) benchmark.

## Usage
- Tutorials for using Slicing Functions (SFs), Transformation Functions (TFs), or doing pre-training with an auxiliary task are included under `tutorials/`. Start with the `WiC_augmentation_tutorial` and `WiC_slicing_tutorial` for gentler introductions to those concepts.
- To train a model for one of the SuperGLUE tasks, use `run.py` with settings you specify or `run.sh` to use general defaults we recommend. (e.g., `bash run.sh CB`)
- See `run.py` for an example of how to add slicing functions to a run.
- Note that the first training run will automatically download the pretrained BERT module, and that training will be very slow in general without a GPU.
