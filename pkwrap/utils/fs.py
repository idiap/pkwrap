# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>

"""
 wrap scripts in kaldi utils directory
"""

import subprocess
from .. import script_utils
import os

def split_data(dirname, num_jobs=0):
    """Call's Kaldi's utils/split_data.sh script

    Args:
        dirname: feature directory to be split
        num_jobs: number of splits
    """
    if num_jobs == 0:
        spk2utt_filename = os.path.join(dirname, "spk2utt")
        num_jobs = num_lines(spk2utt_filename)
    script_utils.run(["utils/split_data.sh", dirname, str(num_jobs)])
    return int(num_jobs)

def num_lines(filename):
    """Find out the number of lines in a file using wc

    This function uses subprocess to run the wc command.
    It quits if the return code from subprocess is non-zero.

    Args:
        filename: a string containing the path of the file
    
    Returns:
        An integer: the number of lines in the file. 
    """
    try:
        p = subprocess.check_output(["wc", "-l", filename])
        return int(p.decode().strip().split()[0])
    except subprocess.CalledProcessError as cpe:
        quit(cpe.returncode)

def make_soft_link(src, dst, relative=False, extra_opts=[]):
    """Create soft link using ln -r -s command

    The function calls quit(1) if execution fails

    Args:
        src: source file
        dst: destination file
        relative: set to True if link is relative
        extra_opts: other options to be passed to ln
    """
    try:
        cmd = ["ln"]
        if relative:
            cmd.append("-r")
        cmd.append("-s")
        if extra_opts:
            cmd += extra_opts
        cmd += [src, dst]
        p = subprocess.run(cmd)
        if p.returncode != 0:
            quit(p.returncode)
    except:
        quit(1)

def touch_file(file_path):
    """Touch a file
    
    This function calls the touch command and quits if the call fails.
    """
    try:
        subprocess.run(["touch", file_path])
    except:
        quit(1)
