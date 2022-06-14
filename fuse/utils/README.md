# FuseMedML Utils Package

The fuse.utils package contains general purpose utilities and helper tools which were not specific to any one of the other existing packages.
The package consists of the following modules:

## config_tools
Contains a function for reading config arguments from a Python dict, Python script file or a callable with given signature.

## cpu_profiling
Contains Timer and Profiler classes for timing code execution, and finding CPU bottlenecks, respectively.

## data
Contains a collate function that converts a list of dictionaries with common keys to a dictionary with these keys whose values become a list of the original dictionary's values:
```dict[str, Any] -> dict[str, list[Any}``` 

## file_io
Contains a set of utilities related to reading and writing files. These include saving and loading Pickle, text, hdf5 files, extracting zip files, path manipulation, file and directory deletion and more.

## misc
Miscellaneous functions and tools without a particular common theme. Some examples: Pretty-printing a Pandas DataFrame for display purposes, user prompt for asking a yes/no question, squeeze batch dimension of different object types, and more.

## multiprocessing

## rand

## remote_execution

## gpu

## ndict

## utils_debug

## utils_hierarchical_dict

## utils_logger
