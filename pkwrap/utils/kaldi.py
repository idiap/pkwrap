"""Wrappers to call kaldi's utils/ scripts"""
from ..script_utils import run
import subprocess

def split_data(data_folder, num_jobs):
    run([
        'utils/split_data.sh',
        data_folder,
        f'{num_jobs}',
    ])
    sdata = '{}/split{}'.format(data_folder, num_jobs)
    return sdata

def get_frame_shift(data_folder):
    process = run([
        'utils/data/get_frame_shift.sh',
        data_folder
    ])
    return float(process.stdout.decode('utf-8'))

def get_utt2dur(data_folder):
    run([
        'utils/data/get_utt2dur.sh',
        data_folder,
    ])

def gen_utt2len(data_folder):
    run([
        'feat-to-len',
        f"{data_folder}/feats.scp",
        "ark,t:{data_folder}/utt2len",
    ])

def filter_scp(id_list=None, exclude=False, input_list=None, output_list=None):
    with open(output_list, 'w') as opf:
        cmd = ['utils/filter_scp.pl']
        if exclude:
            cmd.append('--exclude')
        cmd += [id_list, input_list]
        subprocess.run(cmd, stdout=opf)
        opf.close()

def shuffle(input_file, output_file):
    with open(output_file, 'w') as opf:
        subprocess.run([
            'utils/shuffle_list.pl',
            input_file],
            stdout=opf)
        opf.close()

def split_scp(input_file, prefix='', suffix='', num_splits=-1):
    if num_splits < 1:
        raise ValueError("num splits should be positive integer")
    if num_splits == 1:
        raise NotImplementedError
    out_scp = [f'{prefix}{i}{suffix}' for i in range(1, num_splits+1)]

    subprocess.run([
        'utils/split_scp.pl',
        input_file,
        *out_scp
    ])
