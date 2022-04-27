### References
This work is based out of an implementation of Planet from: https://github.com/Kaixhin/PlaNet


### Running Some of the Stuff
#### Install Pre-commit configs
python -m pre_commit install

#### Tests
python -m unittest

#### Code Quality
python -m pylint src

#### How to run Models
Once every dependencies are installed (following the steps of the next section), you can run the following command to train the a model:

```
python src/main.py disable_cuda=True \
                    algorithm="dreamer" \
                    env="Pendulum-v0" \
                    action_repeat=2 \
                    episodes=100 \
                    collect_interval=50 \
                    hidden_size=32 \
                    belief_size=32 \
                    test_interval=10 \
                    log_video_freq=10
```

Notes:
Use `algorithm="dreamer"` to run the Dreamer algorithm.
Use `algorithm="planet"` to run the Planet algorithm.
Use `algorithm="dreamerV2"` to run the DreamerV2 algorithm (not complete yet).

## Installation Procedures from Homeworks

### Install mujoco:
```
mkdir ~/.mujoco
cd ~/.mujoco
wget https://www.roboti.us/download/mujoco200_linux.zip
unzip mujoco200_linux.zip
mv mujoco200_linux mujoco200
rm mujoco200_linux.zip
wget -O mjkey.txt https://github.com/milarobotlearningcourse/ift6163_homeworks/blob/master/hw1/mjkey.txt?raw=true
```
The above instructions download MuJoCo for Linux. If you are on Mac or Windows, you will need to change the `wget` address to either
`https://www.roboti.us/download/mujoco200_macos.zip` or `https://www.roboti.us/download/mujoco200_win64.zip`.

Finally, add the following to bottom of your bashrc:
```
export LD_LIBRARY_PATH=~/.mujoco/mujoco200/bin/
```

### Install other dependencies

There are two options:

A. (Recommended) Install with conda:

	1. Install conda, if you don't already have it, by following the instructions at [this link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

	```

	This install will modify the `PATH` variable in your bashrc.
	You need to open a new terminal for that path change to take place (to be able to find 'conda' in the next step).

	2. Create a conda environment that will contain python 3:
	```
	conda create -n big-dreamer python=3.7
	```

	3. activate the environment (do this every time you open a new terminal and want to run code):
	```
	source activate big-dreamer
	```

	4. Install the requirements into this conda environment
	```
	pip install --user -r requirements.txt
	```

	5. Allow your code to be able to see 'src'
	```
	cd <path_to_hw1>
	$ pip install -e .
	```


### Debugging issues with installing `mujoco-py`

If you run into issues with installing `mujoco-py` (especially on MacOS), here are a few common pointers to help:
  1. If you run into GCC issues, consider switching to GCC7 (`brew install gcc@7`)
  2. [Try this](https://github.com/hashicorp/terraform/issues/23033#issuecomment-543507812) if you run into developer verification issues (use due diligence when granting permissions to code from unfamiliar sources)
  3. StackOverflow is your friend, feel free to shamelessly look up your error and get in touch with your classmates or instructors
  4. If nothing works and you are frustrated beyond repair, consider using the Colab version of the homework!
