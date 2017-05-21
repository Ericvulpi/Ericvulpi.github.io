---
layout: page
title: TFBenchmarks
permalink: /TFBenchmarks/
---

#### Tensorflow use cases of varying complexity to benchmark hardware, hyperparameters and optimizers

#### [See it on GitHub](https://github.com/Ericvulpi/MusicBar)

## Target

Provide callable or executable scripts of standard functions and neural nets example with
the following characteristics :

- Parameters (all with default value provided) :
  - Hardware : CPU / GPU
  - Convergence criteria : max iterations / max time / target precision, with a failsafe
  - Hyperparameters : batch size, dropout, learning rate, other tbd
  - Optimizer : Gradient Descent / Adadelta / Adagrad / Momentum / Adam / FTRL /
  Proximal Gradient / Proximal Adagrad / RMSProp
  - Custom optimizer : option to use instead an optimzer provided by the user as a function
  of the gradient
  - Output options : see below
- Outputs :
  - Precision or result on the validation set
  - Time to perform the benchmark
  - Optional :
    - tensorboard output
    - convergence file
    - tbd

## List of functions and neural nets

- Rosenbrock function
- MNIST tensorflow example
- tbd
