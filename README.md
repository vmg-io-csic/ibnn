# IBNN

## Aim
This project contains the implementation of the Implicit Bias Neural Networks (IBNN) presented in [[1]](#ibnn-reference).
More specifically, it contains the implementation of layers, based on PyTorch, according to the IBNN paradigm,
and provides utilites for training them and for comparing them to analogous
traditional convolutional layers (based on the Standard Model, SM, of deep learning, based on artificial neurons
according to the point-neuron model) in image classification tasks.

The IBNN layers, as detailed in [[1]](#ibnn-reference), are composed of a number of $D$ neurons, wherein
each neuron $i$ produces a response $v_i$ to the input $(x_{j})_{j=1}^{N}$ of the layer according to   

$$
    \begin{split}
        \displaystyle v_i &= \phi(z_i) \\
        \displaystyle z_i &=
            \displaystyle \sum_{j=1}^N m_{ij} x_j - b_i
            - \lambda \sum_{k=1}^D w_{ik}  \sigma \left( z_k  -  z_i\right) ,
    \end{split}
$$

which corresponds to system of coupled implicit equations and a multidimensional fixed-point problem:
the present implementation relies on the library [TorchDEQ](#torchdeq-reference) for the aspects regarding
implicit operations. Further details can be found in [[1]](#ibnn-reference).

## Structure
In particular, the IBNN layer is implemented by two specific
classes/PyTorch [modules](https://docs.pytorch.org/docs/2.12/generated/torch.nn.Module.html)
that correspond to the same formulation but carry it out with different degree of precision:
- ``IBNNLayer``, which corresponds to the full-version layer performing a detailed calculation of the
  fixed-point solution of the system of coupled implicit equations describing IBNN; and
- ``IBNNLiteLayer``, which is a lighter version of the above ``IBNNLayer`` performing one single
  iteration of the fixed-point search and thus implemented as a purely forward, instead of implicit, layer.

The code of the project (in the folder ``src``) has the following is based on the created packages, subpackages,
and modules:
- ``modified_rf``, containing the code related to the creation, use, and analysis of the ANN layers
  of this project, and containing the following subpackages:
  - ``nn_layers``, which contains the
    PyTorch classes/[modules](https://docs.pytorch.org/docs/2.12/generated/torch.nn.Module.html) corresponding
    to ANN layers (including the class ``SMLayer``, which corresponds, in its convolutional version,
    to the module [Conv2D](https://docs.pytorch.org/docs/2.12/generated/torch.nn.Conv2d.html) of PyTorch, and ``IBNNLayer`` and its light version ``IBNNLiteLayer``),
  - ``fixed_point``,
  - ``calculus``,
  - ``memory_handling``,
- ``applications``, containing the code related to the creation complete functional networks, using (among other blocks)
    the layers from the ``modified_rf`` package, and containing for now the following subpackage:
  - ``classifiers``,
- ``experimental_evaluation``, containing the code related to the evaluation of the implemented layers and networks
    on the core computer vision tasks, and containing for now the following modules:
  - ``experiment_utils.py``, containing the functions for the training, evaluation, logging, and log analysis
    of the implemented layers and networks, and
  - ``adversarial_attacks.py``.

Help and documentation for the existing code can be found in [docs/build/html/index.html](docs/build/html/index.html).

## Setup/install

1. Clone the repo in your local:<br> `git clone https://github.com/vmg-io-csic/ibnn.git`

In the repo folder ``ibnn``:

2. OPTIONAL BUT ADVISABLE: create an environment for the dependencies using one of two options:
    * Using `virtualenv`:
      - Create a `virtualenv`:<br> `virtualenv -p python3.10 ibnn-env`
      - Activate the virtualenv:<br> `source ibnn-env/bin/activate`
    * Using Conda:
      - Create a Conda environment:<br> `conda create -n ibnn-env python=3.10`
      - Activate the Conda environment:<br> `conda activate ibnn-env`
3. Install dependencies:<br> `pip install -r requirements.txt`

Once the dependencies are satisfied, the implemented layers are ready to be used. 

## Run: training and validation

In addition to the implemented layers (and their supporting methods and classes),
the repository contains utilities embodying the architectures and training processes
used to generate the results in [[1]](#ibnn-reference).
In particular, (``/src/``)``main_train_classifiers.py`` allows the creation, training, and validation of image classification experiments for, among others, the datasets
MNIST, FashionMNISTS, SVHN, and CIFAR-10. This script accepts configuration files whose formats can be checked,
detailed, in the file [CLASSIFIER_TRAINING_CONFIG_FILE_FORMAT.md](CLASSIFIER_TRAINING_CONFIG_FILE_FORMAT.md).

### Demo

An illustrative example of the use of the above training utility, which illustrates
how to compare analogous architectures based on different types of hidden layers and
which uses one of the implemented IBNN layers, can be executed using the line

```
python ./src/main_train_classifiers.py ./experiment_configuration_files/demo_sm_vs_ibnn_lite.toml
```

The above demo takes few minutes and addresses the same architecture executed, in the first run, based on usual convolutional hidden layers
(i.e. according to the SM) and, in the second run, based on the lightweight version ``IBNNLiteLayer``
of the proposed artificial neuron model IBNN.

For the analogous example using the complete ``IBNNLayer`` with higher precision calculation of the
fixed-point solutions using multiple iterations (and lasting longer than the above), use the configuration file
``./experiment_configuration_files/demo_ibnn.toml`` instead.

## License

Distributed under the Apache License 2.0. See [LICENSE](./LICENSE) for more information.


[//]: # (## FAQ)

[//]: # ()
[//]: # (Soon...)

## Citation

<a id="ibnn-reference">[1]</a>
Raul Mohedano, Thomas Batard, Erik Velasco-Salido, Ramsses De Los Santos Mendoza, Jorge H. Martínez, Stacey Levine, Marcelo Bertalmío.
*Updating the standard neuron model in artificial neural networks.*
Preprint as [arXiv:2605.30370](https://arxiv.org/abs/2605.30370), 2026.

## Acknowledgements

This work has been supported by:
- Programa Fundamentos (ref. VIS4NN) of Fundación BBVA 2022 (Spain);
- Programa Tecnologías (ref. TEC 2024/COM-322) of Comunidad de Madrid (Spain);
- Proyectos de Generación de Conocimiento 2024 (ref. PID2024-161841NB-I00) of Ministerio de Ciencia,
Innovación y Universidades (Spain); and
- Programa Momentum (ref. MMT24-IO-02) of Consejo Superior de Investigaciones Científicas (Spain),
funded by European Commission–NextGenerationEU through the Spanish Recovery, Transformation and Resilience Plan.


## Other references and acknowledgements

<a id="torchdeq-reference">[2]</a>
Zhengyang Geng and J. Zico Kolter. *TorchDEQ: A Library for Deep Equilibrium Models.*
GitHub repository ([link](https://github.com/locuslab/torchdeq)), 2023.

Moreover, we would like to specially thank [PyTorch](https://pytorch.org/) for their open source environment,
allowing this and many other research teams to develop and test their ideas.