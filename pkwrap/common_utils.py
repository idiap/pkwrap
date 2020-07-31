# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

import subprocess

def copy_file(src, dest):
    """Copy a file from source to destination
    
    This function calls subprocess to use the 'cp' command
    in the shell. In the future we will just use the python
    function.

    Args:
        src: path to source file, a string
        dest: path to destination file, a string
    """
    subprocess.run(["cp", src, dest])

def copy_folder(src, dest):
    """Copy src folder to destination

    This function calls subprocess.run to run 'cp -r' command.
    In the future, we will just use the python function in the
    standard library to do this.

    Args:
        src: source folder
        dest: destination folder
    """
    subprocess.run(["cp", "-r", src, dest])

def read_single_param_file(src, typename=int):
    """Read a file with one value

    This function can be used to read files in Kaldi which
    has parameter values stored in them. E.g. egs/info/num_archives.
    Pass the typename in advance to return the value without errors.

    Args:
        src: the path to file that contains the parameter value
        typename (type): The type of value to be expected in the file.
            Default is int. Any custom value that can take a string
            can be passed.

    Returns:
        Value in the "src" file casted into type "typename"
         
    Raises:
        AssertionError if parameter value is empty.
    """
    param = None
    with open(src) as ipf:
        param = typename(ipf.readline().strip())
    assert param is not None
    return param
