# snorkel-superglue
Applying Snorkel to SuperGLUE

## Installation
To use this repository:
1. Install snorkel (see [snorkel](https://github.com/HazyResearch/snorkel) repo for details)

2. In that virtual environment (or a copy of it if you want to keep them separate), move back to this directory and run:

    ```pip install -r requirements.txt```

3. Set the environment variable SUPERGLUEDATA that points to the directory where the SUPERGLUE data will be stored (we recommend using a directory called `data/` at the root of the repo) by running:

    ```export SUPERGLUEDATA=$(pwd)/data/```
   
4. Download the SuperGLUE data by running: 

    ```python download_superglue_data.py --data_dir $SUPERGLUEDATA --tasks all```