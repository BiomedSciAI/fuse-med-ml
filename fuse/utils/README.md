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
