# Planning Through Contact

This repository was used to generate all simulated datasets in our paper on the [Empirical Analysis of Sim-and-Real Cotraining of Diffusion Policies for Planar Pushing from Pixels](https://arxiv.org/abs/2503.22634). It forks the original [planning-through-contact](https://github.com/bernhardpg/planning-through-contact) repository and provides wrappers for data generation.

---

## Installation (Linux and MacOS)
If you run into Git LFS errors while attempting to clone the repository, try setting `GIT_LFS_SKIP_SMUDGE=1`.
```
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:adamw8/planning-through-contact.git
```

This repo uses Poetry for dependency management. To setup this project, first
install [Poetry](https://python-poetry.org/docs/#installation) and, make sure
to have Python3.10 installed on your system.

```python
poetry env use python3.10
poetry install -vvv
```

### (Optional) Installing customized Drake version

If you are using this repository to generate planar pushing data, you will need 
a [customized version](https://github.com/bernhardpg/drake/tree/sample-paths-and-sdp-options) of Drake. To install, follow these instructions:

Navigate to a desired installation location and run:

```console
git clone git@github.com:bernhardpg/drake.git 
cd drake
git checkout sample-paths-and-sdp-options
```

To build Drake and the python bindings, run:

```console
cd ../
mkdir drake-build
cd drake-build
cmake -DWITH_MOSEK=ON -DWITH_SNOPT=ON ../drake
make install
export PYTHONPATH={DRAKE_BUILD_DIR_PATH}/install/lib/python3.10/site-packages:${PYTHONPATH}
```
where `{DRAKE_BUILD_DIR_PATH}` should be replaced with the absolute path to the `drake-build` directory above.

If you are a member of the RLG, run:

```console
cd ../
mkdir drake-build
cd drake-build
cmake -DWITH_MOSEK=ON -DWITH_ROBOTLOCOMOTION_SNOPT=ON ../drake
make install
export PYTHONPATH={DRAKE_BUILD_DIR_PATH}/install/lib/python3.10/site-packages:${PYTHONPATH}
```

See [the docs](https://drake.mit.edu/from_source.html) for more information on building Drake.

### Installing the Matplotlib backend for visualization

This can't be installed with poetry.

```
pip install PyQt5
```

#### Activating the environment

To activate the environment, run:

```console
poetry shell
```

### (Optional) Additional packages

Make sure to have graphviz installed on your computer. On MacOS, run the following
command:

```python
brew install graphviz
```

## Automated Data Generation
To generate data, please ensure that `PYTHONPATH` points to the custom version of Drake. You may also need to configure `QT`.
```
export PYTHONPATH={DRAKE_BUILD_DIR_PATH}/install/lib/python3.10/site-packages:${PYTHONPATH}
export QT_QPA_PLATFORM=offscreen
```

Both the simulation environment and the data generation process are specified by a config file. Here is an example usage of the data generation script:
```
python scripts/planar_pushing/run_data_generation.py \
    --config-dir config/sim_config \
    --config-name example_sim_config.yaml
```

`config/sim_config/example_sim_config.yaml` is a good starting template.

The data generation runs in 3 phases (which can all be configured and ran independently).
1. **Plan Generation** (if `generate_plans: true`): [Generates plans](https://arxiv.org/abs/2402.10312) and saves them to `plans_dir`.
2. **Plan Rendering** (if `render_plans: true`): Renders the plans in `plans_dir` and saves them to `rendered_plans_dir`.
3. **Zarr Conversion**(if `convert_to_zarr: true`): Compresses the rendered plans in `rendered_plans_dir` into a zarr dataset.

If you run into LCM issues with rendering plans, see [Rendering batched trajectories](#rendering-batched-trajectories).


## Data Collection & Eval in *Target Sim* 

**Note:** For historical reasons, the *target sim* is referred to as *sim-sim* in some parts of the codebase.

When using *target sim*, do not use the custom version of Drake. Instead, use `drake 1.27.0`.

All *target sim* config files are located in `config/sim_config/sim_sim`. The environment used in the [paper](https://arxiv.org/abs/2503.22634) was `gamepad_teleop_carbon.yaml`.

### Data Collection
The data collection can be run using:
```
python scripts/planar_pushing/run_gamepad_teleop.py --config-dir <dir> --config-name <file>
```
After running the script, open the meshcat. You will be prompted in the terminal to press any key to connect the gamepad controller.

Press `A` to start data collection. While data collection is active, press `A` to end data collection and save the trajectory. Alternatively, press `B` to end data collection and discard the trajectory. Press `X` at anytime to terminate the script.

**Note:** The button mappings on the gamepads vary from controller to controller! To test your mapping, visit https://hardwaretester.com/gamepad. You may need to edit `self.button_index` in `planning_through_contact/simulation/planar_pushing/gamepad_controller.py`.

### Policy Evaluation in Simulation
`scripts/launch_eval.py` evaluates 'groups' of policy checkpoints in parallel.

Example usage::
```
python scripts/planar_pushing/launch_eval.py \
    --csv-path config/example_launch_eval.txt \
    --max-concurrent-jobs 8 \
    --num-trials 50 50 100 \
    --drop-threshold 0.05
```
The structure of `example_launch_eval.txt` is as follows. Each line represents a 'group' of checkpoints. In each line:

* The first argument provides a path to the group. If the path is a directory, the group will contain all checkpoints in the `checkpoints` subdirectory. If the path is a checkpoint (ends with `.ckpt`), the group will only contain the single checkpoint.

* The second argument is an output directory where the logs for each group will be written.

* The third argument provides the name (as opposed to the path) of the simulation config file in `config/sim_config`. The evaluation configuration (length of trials, success criteria, etc) are specified in the `multi_run_config` field of the simulation config.

`--num-trials` specifies the number of rounds to evaluate each group of checkpoins and the number of trials per round. In the example above, we will evaluate all groups for 3 rounds with 50, 50, and 100 trials.

At the end of each round, the poorest performing checkpoints in each group are discarded before beginning the next round. This avoids wasting compute on poor-performing policies.

 `--drop-threshold` controls how policies are dropped. In the example, if a checkpoint is less than 5% likely to be best in the group, it will be dropped.

When all evaluations are complete, the script outputs a summary of the evaluations. This includes the paths to the best checkpoints in each directory and their success rates. More evaluation information is available in the logging output directories (raw logs, plots, summaries, meshcat html recordings, etc).

If you run into this issue,
```
qt.qpa.plugin: Could not load the Qt platform plugin "xcb" in "/home/adam/.cache/pypoetry/virtualenvs/planning-through-contact-5FVglCdX-py3.10/lib/python3.10/site-packages/cv2/qt/plugins" even though it was found.
```
run:
```
export QT_QPA_PLATFORM=offscreen
```

## Utility Scripts
### Rendering batched trajectories
When rendering large numbers of plans with the data generation script (150+), 
you may run into LCM issues. If so, split the plans into batches of 150 or less 
and move each batch into a subdirectory at `{saved_trajectories_dir}/run_{i}`,
where `i` is the batch index, i.e. your directory structure should look like this:
```
{saved_trajectories_dir}
|-- run_0
|-- run_1
...
|-- run_{num_batches}
```

`render_batched_trajectories.py` renders each batch into the desired output directory. The following command runs the script. Ensure that `render_plans` is `true` in the config file.
```
python scripts/planar_pushing/util/render_batched_trajectories.py \
    --start-index <start-index> \
    --end-index  <end-index> \
    --config-dir <config-dir> \
    --config-name <config-name> \
    --plans-root <saved_trajectories_dir> \
    --suppress-output # set to suppress output
```

**Note:**: This script is a temporary solution for the LCM issue. The need for this script will be eliminated in the future.

### Generating Custom Slider SDFs
This repository can generate plans for custom sliders.

Here is an example of config values for using a custom slider:
```
slider_type: 'arbitrary'
arbitrary_shape_pickle_path: arbitrary_shape_pickles/small_t_pusher.pkl
arbitrary_shape_rgba: [0.1, 0.1, 0.1, 1.0]
arbitrary_shape_visual_mesh_path: null # use collision geometry
```

The repository can generate custom shape pickles (which get converted to SDF during data generation). Modify the `boxes` dictionary in `create_arbitrary_shape_pickle.py` to specify your desired shape. Then run:
```
python scripts/create_arbitrary_shape_pickle.py
```
To visualize the shape, modify `loaded_boxes` in `visualize_arbitrary_shape_pickle.py` and run"
```
python scripts/visualize_arbitrary_shape_pickle.py
```
