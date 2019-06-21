# snorkel-superglue
Applying Snorkel to SuperGLUE

## Installation
To use this repository:
1. Install snorkel (see [snorkel](https://github.com/HazyResearch/snorkel) repo for details)

2. Create a virtual environment (optionally the same environment as `snorkel`) that includes the same required packages as `snorkel`. Then in this environment, from your `snorkel` directory run:

    ```pip -e install .```

3. Move back to this directory and run:

    ```pip install -r requirements.txt```

4. Set the environment variable SUPERGLUEDATA that points to the directory where the SUPERGLUE data will be stored (we recommend using a directory called `data/` at the root of the repo) by running:

    ```export SUPERGLUEDATA=data/```
   
5. Download the SuperGLUE data by running: 

    ```python download_superglue_data.py --data_dir $SUPERGLUEDATA --tasks all```