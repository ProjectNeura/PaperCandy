# PaperCandy

**PaperCandy** is a loosely coupled lightweight framework for deep learning papers. It provides a series of auxiliary tools for rapid rebuilding and writing papers. So far support PyTorch as the only front-end framework.

## Installation

**NOTICE: This version is built with PyTorch. Consider reimplementing the abstract classes in package `papercandy.core` to adapt to the framework that you use if you are not a PyTorch user.**

### Optional Modules

#### CUPY

PaperCandy is capable to use CUPY as an optional replacement for Numpy, which means you need to manually install it. You can find their documentation 👉[here](https://cupy.dev/).

#### COOTA

PaperCandy provides some extra functions based on COOTA, which are only available having installed COOTA. Documentation 👉[here](https://github.com/ATATC/COOTA).

Check if COOTA is available:

```python
from papercandy.core import optional_modules as om
print(om.coota_is_available())
```

### Repository

[Github](https://github.com/ATATC/PaperCandy) | [Pypi](https://pypi.org/project/paper-candy/)

### Package 

#### For Windows

```shell
pip install paper-candy
```

#### For Linux & macOS

```shell
pip3 install paper-candy
```

## Quick Start

***You can find classes and methods annotated under `papercandy.core`.***

Some preparation needs to be done before the demo works.

1. Run this script to create the dataset.

   ```python
   import os
   NUM_ITEMS = 6
   os.mkdir("./data")
   for i in range(1, NUM_ITEMS + 1):
       with open(f"./data/{i}.txt", "w") as f:
           f.write(f"[{2 * i - 1}, {2 * i}]")
   ```

2. Create file `config.txt` and leave it empty.

3. Make sure you have the structure like this:

   data
   
   --------1.txt
   
   --------2.txt
   
   --------3.txt
   
   --------4.txt
   
   --------5.txt
   
   --------6.txt
   
   config.txt
   
   main.py

```python
from papercandy import *
from torch import nn


class ExampleDataset(Dataset):
    def __init__(self, src: Union[str, PathLike]):
        self.src: Union[str, PathLike] = src
        self.file_list: list = _listdir(self.src)

    def __len__(self) -> int:
        return len(self.file_list)

    def cut(self, i: slice) -> Self:
        o = ExampleDataset(self.src)
        o.file_list = self.file_list[i]
        return o

    def get(self, i: int) -> _network.DataCompound:
        with open("%s/%s" % (self.src, self.file_list[i]), "r") as f:
            return _network.DataCompound(_Tensor([int(self.file_list[i])]), _Tensor(eval(f.read())))
          

if __name__ == "main":
    CONFIG().CURRENT = new_config("./config.txt")
    dataset = ExampleDataset("./data")
    # `num_works`: the number of processes
    # `batch_size`: batch size
    dataloader = Dataloader(dataset, num_works=2, batch_size=4)
    trainer = Trainer(dataloder)
    
    torch_network = YOUR_NETWORK()
    network_container = NetworkC(torch_network)
    trainer.set_network(network_container)
    
    torch_loss_function = YOUR_LOSS_FUNCTION()
    loss_function_container = LossFunctionC(torch_loss_function)
    trainer.set_loss_function(loss_function_container)
    
    torch_optimizer = YOUR_OPTIMIZER()
    optimizer_container = OptimizerC(torch_optimizer)
    trainer.set_optimizer(optimizer_container)
    
    # the monitor is a callback interface for trainer
    tariner.train(monitor=TrainingMonitor())	# optional kwargs: `num_batches`, `monitor`
    
    drawer = draw(trainer, 1920, 1080)	# width, height
    drawer.save("./training_loss").show()
```

## Predefined Configuration

| Name               | Required Type               | Default Value | Usage                                                       |
| ------------------ | --------------------------- | ------------- | ----------------------------------------------------------- |
| `gpu_acceleration` | papercandy.core.config.Bool | False         | Whether to enable GPU acceleration in the training process. |
| `device`           | int                         | 0             | The GPU device.                                             |

## FAQ

1. ### macOS, Unexpected Ending

   #### Description

   The program unexpectedly ends.

   #### Cause

   It seems like PyTorch has some kind of bug on macOS that makes the program end when this happens when the network is forwarding.

   #### Solution

   There hasn't been a solution but to change to another system.

2. ### GPU Acceleration

   #### Description

   Cannot enable GPU acceleration even though `gpu_acceleration` has been set to True.

   #### Cause

   This is mostly because you didn't install the correct version of PyTorch. The CPU version will be automatically installed if any version of PyTorch was not found when you were installing PaperCandy, which is not capable of GPU acceleration, so the configuration will be compulsively changed to False.

   #### Solution

   Uninstall PyTorch and reinstall the correct version. See 👉[here](https://pytorch.org/get-started/locally/)
