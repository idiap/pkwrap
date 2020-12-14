"""This module contains training related classes and functions"""
from dataclasses import dataclass
from . import script_utils
from _pkwrap import kaldi
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# TODO: Use ConfigLoaderMixin in Opts
class ConfigLoaderMixin:
    """A mixin for classes wishing to load member values from a dict type object"""
    def load_from_config(self, cfg):
        for key, value in cfg.items():
            if hasattr(self, key):
                type_of_value = type(getattr(self, key))
                setattr(self, key, type_of_value(value))
        return self

@dataclass
class TrainerOpts:
    num_jobs_initial:int = 1
    num_jobs_final: int = 6
    lr_initial: float = 0.01
    lr_final: float = 0.001
    iter_no: int = 0
    num_epochs: int = 6
    train_stage: int = 0
    frames_per_iter: int = 120000
    chunk_width: str = "140"
    cmd: str = 'queue.pl -l q_gpu -V'
    diagnostics_interval: int = 10
    online_ivector_dir: str =  ''
    minibatch_size:str = "32"
    srand: int = 1
    
    def load_from_config(self, cfg):
        for key, value in cfg.items():
            if hasattr(self, key):
                type_of_value = type(getattr(self, key))
                setattr(self, key, type_of_value(value))
        return self

    def load_from_config_file(self, cfg, name="trainer"):
        raise NotImplementedError

@dataclass
class ModelOpts:
    model_file: str = ''
    dirname: str = './'
    left_context: int = 0
    right_context: int = 0
    egs_dir: str = './egs'
    den_graph: str = './den.fst'
    frame_subsampling_factor: int = 3

    def set_dirname(self, dirname, reset_paths=True):
        self.dirname = dirname
        if reset_paths:
            self.egs_dir = os.path.join(self.dirname, 'egs')
            self.den_graph = os.path.join(self.dirname, 'den.fst')

    def load_from_config(self, cfg):
        for key, value in cfg.items():
            if hasattr(self, key):
                type_of_value = type(getattr(self, key))
                setattr(self, key, type_of_value(value))
        return self

@dataclass
class JobOpts:
    job_id: int =  1
    lr: int = 0.01
    l2_regularize = 0.01
    minibatch_size:str = "32"
    model_opts: ModelOpts = ModelOpts()

    @staticmethod
    def create_jobs(num_jobs=1, lr=0.01):
        """Create an array of jobs"""
        return [
            JobOpts(job_id=1, lr=lr)
            for j in range(1, num_jobs+1)
        ]


class SGETrainer:
    def __init__(self, trainer_opts, model_opts, chain_opts=None):
        self.trainer_opts = trainer_opts
        self.model_opts = model_opts
        if chain_opts is None:
            self.chain_opts = kaldi.chain.CreateChainTrainingOptionsDefault()
        else:
            self.chain_opts = chain_opts

    def train(self):
        """run training"""
        num_archives_to_process = num_archives*self.trainer_opts.num_epochs*self.model_opts.frame_subsampling_factor
        num_iters = (num_archives_to_process*2) // (trainer_opts.num_jobs_initial + trainer_opts.num_jobs_final)
        num_archives_processed = 0

        for iter_no in range(0, num_iters):
            self.train_opts.iter_no = iter_no
            num_jobs = script_utils.get_current_num_jobs(
                iter_no, 
                num_iters,
                self.model_opts.num_jobs_initial,
                1, # we don't play with num-jobs-step
                self.model_opts,
            )
            assert num_jobs>0
            if iter_no < self.trainer_opts.train_stage:
                num_archives_processed += num_jobs
                continue
            lr = script_utils.get_learning_rate(
                iter_no, 
                num_jobs, 
                num_iters, 
                num_archives_processed, 
                num_archives_to_process,
                args.lr_initial, 
                args.lr_final,
                schedule_type='exponential'
            )
            logging.info("lr={}\n".format(lr))
            with ThreadPoolExecutor(max_workers=num_jobs) as executor:
                    job_pool = []
                    sys.stderr.write("Num jobs = {}\n".format(num_jobs))
                    sys.stderr.flush()
                    for job_id in range(1, num_jobs+1):
                        frame_shift = num_archives_processed%model_opts.frame_subsampling_factor
                        job_opts = JobOpts(job_id=job_id, num_jobs=num_jobs, lr=lr, frame_shift=frame_shift)
                        p = executor.submit(self.run_job, job_opts)
                        num_archives_processed += 1
                        job_pool.append(p)
                    for p in as_completed(job_pool):
                        result_code = p.result()
                        if result_code != 0:
                            quit(result_code)

        def run_job(self, job_opts):
            dirname = self.model_opts.dirname
            model_file = os.path.join(dirname, f"{iter_no}".pt)
            egs_dir = self.model_file.egs_dir

            iter_no = self.train_opts.iter_no
            cmd = self.train_opts.cmd
            
            num_jobs = job_opts.num_jobs
            job_id = job_opts.job_id
            lr = job_opts.lr
            l2_regularize = job_opts.l2_regularize
            frame_shift = job_opts.frame_shift
            minibatch_size = job_opts.minibatch_size
            log_file = "{}/log/train.{}.{}.log".format(dirname, iter_no, job_id)
            egs_idx = num_archives_processed%num_archives+1
            process_out = subprocess.run([
                *cmd.split(),
                log_file,
                model_file,
                "--dir", dirname,
                "--mode", "training",
                "--cell-dim", cell_dim,
                "--lr", f"{lr}",
                "--frame-shift", f"{frame_shift}",
                "--egs", f"ark:{egs_dir}/cegs.{egs_idx}.ark",
                "--l2-regularize", f"{l2_regularize}",
                "--l2-regularize-factor", f"{1.0/num_jobs}",
                "--minibatch-size", minibatch_size,
                "--new-model", os.path.join(dirname, f"{iter_no}.{job_id}.pt"),
                os.path.join(dirname, f"{iter_no}.pt"),
            ])
            return process_out.returncode
