# FuseMedML Utils Package

The fuse.utils package contains general purpose utilities and helper tools which were not specific to any one of the other existing packages.
The package consists of the following modules:

## config_tools
Contains configuration file utilities for reading config arguments from a Python dict, Python script file or a callable with given signature.

## cpu_profiling
Contains Timer and Profiler classes for timing code execution, and finding CPU bottlenecks, respectively.

## data
Contains a collate function that converts a list of dictionaries with common keys to a dictionary with these keys whose values become a list of the original dictionary's values:
```dict[str, Any] -> dict[str, list[Any]]``` 

## file_io
Contains a set of utilities related to reading and writing files. These include saving and loading Pickle, text, hdf5 files, extracting zip files, path manipulation, file and directory deletion and more.

## misc
Miscellaneous functions and tools without a particular common theme. Some examples: Pretty-printing a Pandas DataFrame for display purposes, user prompt for asking a yes/no question, squeeze batch dimension of different object types, and more.

## multiprocessing
Contains tools and helper functions related to running workers in parallel using multiple processes.

## rand
Contains a class for setting a given random seed in common libraries, and several classes for various kinds of random sample drawing.

## remote_execution
Contains utilities for executing code on remote machines.

## gpu
GPU related utilities, including finding which GPUs are available, setting `$CUDA_VISIBLE_DEVICES` environment variable, and more.

## ndict
Nested dictionary implementation. Extends Python's `dict` type to allow accessing keys with a '.' notation, i.e `d.abc.def` instead `d['abc']['def']`.

## utils_debug
Contains a class implementing debugging utilities. In "debug" mode it automatically disables multi-threading and multiprocessing. 

## utils_hierarchical_dict
Implements a class similar to Ndict described above. It is kept for backwards compatibility.

## utils_logger
Implements logging and console output formatting functions.