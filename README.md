# PaperCandy

**PaperCandy** is a loosely coupled lightweight framework for deep learning papers. It provides a series of auxiliary tools for rapid rebuilding and writing papers. So far support PyTorch as the only front-end framework.

## Installation

**NOTICE: This version is built with PyTorch. Consider reimplementing the abstract classes in package `papercandy.universal` to adapt to the framework that you use if you are not a PyTorch user.**

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

## FAQ

1. ### macOS, Unexpected Ending

   #### Description

   The program unexpectedly ends.

   #### Cause

   It seems like PyTorch has some kind of bug on macOS that makes the program end when this happens:

   ```python
   network: torch.nn.Module = ......
   network(data)	# The program mostly ends here.
   ```

   #### Solution

   There hasn't been a solution but to change to another system.

2. ### GPU Acceleration

   #### Description

   Cannot enable GPU acceleration even though `gpu_acceleration` has been set to True.

   #### Cause

   This is mostly because you didn't install the correct version of PyTorch. The CPU version will be automatically installed if any version of PyTorch was not found when you were installing PaperCandy, which is not capable of GPU acceleration, so the configuration will be compulsively changed to False.

   #### Solution

   Uninstall PyTorch and reinstall the correct version. See 👉[here](https://pytorch.org/get-started/locally/)
