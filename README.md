<h1 align="center">ECAM: A Contrastive Learning Approach to Avoid Environmental Collision in Trajectory Forecasting</h1>
<p align="center">
  <a href="https://jackroi.github.io/"><strong>Giacomo Rosin</strong></a>
  ·
  <a href="https://github.com/ram95d"><strong>Muhammad Rameez Ur Rahman</strong></a>
  ·
  <a href="https://www.sebastianovascon.it/"><strong>Sebastiano Vascon</strong></a>
  <br>
  IJCNN 2025
  <br>
  <a href="https://arxiv.org/abs/2506.09626">Paper</a>
</p>

ECAM (Environmental Collision Avoidance Module) is
a contrastive learning-based module to enhance collision
avoidance ability with the environment, of trajectory forecasting models.
It can be integrated into existing models, improving their ability to generate
collision-free trajectories, with zero overhead during inference.

<div align='center'>
  <br>
  <img src="assets/ECAM-comparison.png" alt="ECAM qualitative comparison" width=60%>
  <br>
  <img src="assets/ECAM-diagram.png" alt="ECAM diagram" width=90%>
</div>

## Setup

### Clone the repository

```shell
git clone https://github.com/CVML-CFU/ECAM.git
cd ECAM
```

### Install the required packages

First create a virtual environment with Python 3.11, eg. with `uv`:

```shell
uv venv --python 3.11
source .venv/bin/activate
```

Then install the required packages. We provide two different requirements files,
one for CPU only and one for GPU (CUDA). Choose the one that fits your setup.

#### CPU only

```shell
pip install -r requirements-cpu.txt
```

#### GPU (cuda)

```shell
pip install -r requirements-cuda.txt
```

### Download the additional data

Download the additional data required for the experiments.

```shell
./SingularTrajectory/script/download_extra.sh
```

## Run the model

First, change into the `SingularTrajectory` directory:

```shell
cd SingularTrajectory
```

### Train
To train the model, run the following command:

```shell
./script/train.sh -p CONFIG_PREFIX -t TAG -d {eth|hotel|univ|zara1|zara2|sdd|pfsd|thor} -v {orig|map|ecam} -g {cpu|gpu}"
```

Example:

```shell
./script/train.sh -p stochastic/singulartrajectory -t SingularTrajectory-stochastic -d "eth" -v ecam -g gpu
```

### Test
To test the model, run the following command:

```shell
./script/test.sh -p CONFIG_PREFIX -t TAG -d {eth|hotel|univ|zara1|zara2|sdd|pfsd|thor} -v {orig|map|ecam} -g {cpu|gpu}"
```

Example:
```shell
./script/test.sh -p stochastic/singulartrajectory -t SingularTrajectory-stochastic -d "eth" -v ecam -g gpu
```

<!-- ## Citation -->

<!-- If you find our work useful in your research, please cite our paper ECAM: -->

<!-- ```bibtex -->
<!-- @misc{rosin2025ecamcontrastivelearningapproach, -->
<!--       title={ECAM: A Contrastive Learning Approach to Avoid Environmental Collision in Trajectory Forecasting}, -->
<!--       author={Giacomo Rosin and Muhammad Rameez Ur Rahman and Sebastiano Vascon}, -->
<!--       year={2025}, -->
<!--       eprint={2506.09626}, -->
<!--       archivePrefix={arXiv}, -->
<!--       primaryClass={cs.CV}, -->
<!--       url={https://arxiv.org/abs/2506.09626}, -->
<!-- } -->
<!-- ``` -->

## Acknowledgements

This work builds upon code from the
[SingularTrajectory](https://github.com/InhwanBae/SingularTrajectory),
[EigenTrajectory](https://github.com/InhwanBae/EigenTrajectory),
[Social-NCE](https://github.com/vita-epfl/social-nce), and
[AgentFormer](https://github.com/Khrylx/AgentFormer) repositories.
We sincerely thank the respective authors for making their implementations
and models available.
