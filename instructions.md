## Python environment installation instructions
- Make sure you have conda installed in your system. [Instructions link here](https://conda.io/projects/conda/en/latest/user-guide/install/index.html#regular-installation).
- Then, get the `conda_env.yml` file, and from the same directory, run `conda env create -f conda_env.yml`. 
- Activate the environment, conda activate ddrl.

## Running the code
- Make sure you have the environment activated, and you are in the `policy` directory.
- Update the `root_dir` variable in `cfgs/config.yml` to the absolute path of the `Tutorial_1` directory.
- Command to run code: `python train.py agent=<agent_name> dataset_type=<dataset_type> experiment=<exp_name>` where `agent_name` is one of `['bc', 'gcbc', 'bet']`, `dataset_type` is one of `['fixed_goal', 'changing_goal', 'multimodal']` and `exp_name` is the name of the experiment.