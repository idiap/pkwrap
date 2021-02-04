#!/usr/bin/env python3
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
#  Written by Srikanth Madikeri <srikanth.madikeri@idiap.ch>
#             Juan Pablo Zuluaga <juan-pablo.zuluaga@idiap.ch>

description = """
  This script trains and tests chain models.
  It takes a config file, if none is provided the script will look at configs/default.
"""

import os
import sys
import shutil
import datetime
import torch
import pkwrap
import argparse
import subprocess
import configparser
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
logging.basicConfig(level=logging.DEBUG, format='pkwrap %(levelname)s: %(message)s')


def run_diagnostics(dirname, model_file, iter_no, egs_file, job_cmd, diagnostic_name='valid'):
    """
        sub a single diagnostic job and let ThreadPoolExecutor monitor its progress
    """
    log_file = "{}/log/compute_prob_{}.{}.log".format(dirname, diagnostic_name, iter_no)
    process_out = subprocess.run([*job_cmd.split(),
                log_file,
                model_file,
                "--dir", dirname,
                "--mode", "diagnostic",
                "--egs", "ark:{}".format(egs_file),
                os.path.join(dirname, "{}.pt".format(iter_no))])
    return process_out.returncode


def submit_diagnostic_jobs(dirname, model_file, iter_no, egs_dir, job_cmd):
    job_pool = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        for diagnostic_name in ['train', 'valid']:
            egs_file = os.path.join(egs_dir, '{}_diagnostic.cegs'.format(diagnostic_name))
            p = executor.submit(
                run_diagnostics, 
                dirname, 
                model_file,
                iter_no,
                egs_file,
                job_cmd,
                diagnostic_name
            )
            job_pool.append(p)
    return job_pool

def run_job(num_jobs, job_id, dirname, iter_no, model_file, lr, frame_shift, egs_dir,
    num_archives, num_archives_processed, minibatch_size, job_cmd):
    """
        sub a single job and let ThreadPoolExecutor monitor its progress
    """
    log_file = "{}/log/train.{}.{}.log".format(dirname, iter_no, job_id)
    process_out = subprocess.run([*job_cmd.split(),
                log_file,
                model_file,
                "--dir", dirname,
                "--mode", "training",
                "--lr", str(lr),
                "--frame-shift", str(frame_shift),
                "--egs", "ark:{}/cegs.{}.ark".format(egs_dir, num_archives_processed%num_archives+1),
                "--l2-regularize-factor", str(1.0/num_jobs),
                "--minibatch-size", minibatch_size,
                "--new-model", os.path.join(dirname, "{}.{}.pt".format(iter_no, job_id)),
                os.path.join(dirname, "{}.pt".format(iter_no))])
    return process_out.returncode

def train():
    parser = argparse.ArgumentParser(description="Acoustic model training script")
    pkwrap.script_utils.add_chain_recipe_opts(parser)
    # the idea behind a test config is that one can run different configurations of test
    parser.add_argument("--test-config", default="test", help="name of the test to be run")
    parser.add_argument("--decode-iter", default="final")
    parser.add_argument("--config", default="configs/default")
    args = parser.parse_args()
    
    logging.info("Reading config")
    cfg_parse = configparser.ConfigParser()
    cfg_parse.read(args.config)
    cmd = cfg_parse["cmd"]
    cpu_cmd = cmd['cpu_cmd']
    cuda_cmd = cmd['cuda_cmd']

    exp_cfg = cfg_parse["exp"]
    assert exp_cfg is not None

    stage = args.stage
    model_file = exp_cfg["model_file"]
    data = exp_cfg["data"] if "data" in exp_cfg else "data"
    exp = exp_cfg["exp"] if "exp" in exp_cfg else "exp"
    chain_affix = exp_cfg["chain_affix"] if "chain_affix" in exp_cfg else ""
    chain_dir = os.path.join(exp, f"chain{chain_affix}")
    dirname = os.path.join(chain_dir, exp_cfg["dirname"])
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    egs_dir = os.path.join(dirname, "egs")
    if "e2e" in exp_cfg:
        is_e2e = bool(exp_cfg["e2e"])
    else:
        is_e2e = False
    if not is_e2e:
        gmm_dir = exp_cfg["gmm_dir"]
        ali_dir = exp_cfg["ali_dir"]
        lat_dir = exp_cfg["lat_dir"]
        lores_train_set = exp_cfg["lores_train_set"]
    tree_dir = exp_cfg["tree_dir"]
    train_set = exp_cfg["train_set"]
    lang = exp_cfg["lang"] if "lang" in exp_cfg else "lang"
    lang_chain = exp_cfg["lang_chain"] if "lang_chain" in exp_cfg else "lang_chain"

    l2_regularize = args.l2_regularize
    
    model_opts = pkwrap.trainer.ModelOpts().load_from_config(exp_cfg)
    frame_subsampling_factor = model_opts.frame_subsampling_factor
    trainer_opts = pkwrap.trainer.TrainerOpts().load_from_config(exp_cfg)
    
    # create lang folder
    if stage <= 0:
        logging.info("Creating lang_chain folder from lang")
        try:
            rc = subprocess.run([
                "shutil/chain/check_lang.sh", 
                lang, 
                lang_chain,
            ]).returncode
        except Exception as e:
            #           TODO: should use logging
            sys.stderr.write(e)
            sys.stderr.write("ERROR: copying lang failed")
        logging.info(f"Created {lang_chain} folder")
    # create lats
    if stage <= 1:
        logging.info("Create supervision lattices")
        try:
            subprocess.run([
                "steps/align_fmllr_lats.sh", "--nj", "{}".format(args.decode_nj),
                "--cmd", "{}".format(cpu_cmd),
                lores_train_set,
                "{}".format(lang),
                gmm_dir,
                lat_dir,
            ])
        except Exception as e:
            logging.error(e)
            logging.error("Lattice creationg failed")
        logging.info("Finished creating supervision lattices")
    # build tree
    if stage <= 2:
        tree_file = os.path.join(tree_dir, "tree")
        if os.path.isfile(tree_file):
            logging.error(
                f"Tree file {tree_file} already exists."
                " Refusing to overwrite".format(tree_file)
            )
            quit(1)
        tree_size = exp_cfg["tree_size"] if "tree_size" in exp_cfg else 7000
        logging.info(f"Using tree_size={tree_size}")
        if not os.path.isfile(os.path.join(tree_dir, '.done')):
            cmd = [
                "steps/nnet3/chain/build_tree.sh",
                "--frame-subsampling-factor", f"{frame_subsampling_factor}",
                '--context-opts', "--context-width=2 --central-position=1",
                "--cmd", "{}".format(cpu_cmd),
                tree_size,
                f"{lores_train_set}",
                f"{lang_chain}",
                f"{ali_dir}",
                f"{tree_dir}",
            ]
            pkwrap.script_utils.run(cmd)
            subprocess.run(["touch", "{}/.done".format(tree_dir)])
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
        logging.info(f"Created {dirname}")
    if not os.path.isfile(os.path.join(dirname, 'tree')):
        shutil.copy(
            os.path.join(tree_dir, "tree"),
            dirname
        )
    learning_rate_factor = 0.5/args.xent_regularize

#   create den.fst
    if stage <= 3:
        logging.info("Creating den.fst")
        pkwrap.script_utils.run([
            "shutil/chain/make_den_fst.sh",
            "--cmd", f"{cpu_cmd}",
            tree_dir, 
            gmm_dir, 
            dirname,
        ])

    if not os.path.isfile(os.path.join(dirname, 'num_pdfs')):
        logging.info(f"Creating num_pdfs file in {dirname}")
        num_pdfs = subprocess.check_output([
            "tree-info", 
            os.path.join(tree_dir, "tree")
        ]).decode().strip().split('\n')[0].split()[1]
        with open(os.path.join(dirname, 'num_pdfs'), 'w') as opf:
            opf.write(num_pdfs)
            opf.close()
    if not os.path.isfile(os.path.join(dirname, "0.trans_mdl")):
        pkwrap.script_utils.run([ 
                "copy-transition-model",
                os.path.join(tree_dir, "final.mdl"),
                os.path.join(dirname, "0.trans_mdl"),
        ])
#   create or copy the egs folder
    context = None
    if stage <= 4 and not ("egs_dir" in exp_cfg and exp_cfg["egs_dir"]):
        logging.info("Creating egs")
        # first check the context
        process_out = subprocess.run([
            model_file,
            "--mode", "context",
            "--dir", dirname,
            "0.pt", # use a dummy model. 
        ])
        if process_out.returncode != 0:
            quit(process_out.returncode)
        with open(os.path.join(dirname, 'context')) as ipf:
            context = int(ipf.readline())
        pkwrap.script_utils.run([
            "steps/chain/get_egs.sh",
            "--cmd", cpu_cmd,
            "--cmvn-opts", "--norm-means=false --norm-vars=false",
            "--left-context", str(context),
            "--right-context", str(context),
            "--frame-subsampling-factor", str(frame_subsampling_factor),
            "--alignment-subsampling-factor", str(frame_subsampling_factor),
            "--frames-per-iter", str(trainer_opts.frames_per_iter),
            "--frames-per-eg", str(trainer_opts.chunk_width),
            "--srand", str(trainer_opts.srand),
            "--online-ivector-dir", trainer_opts.online_ivector_dir,
            train_set, dirname, lat_dir, egs_dir
        ])
    elif "egs_dir" in exp_cfg:
        egs_dir = exp_cfg["egs_dir"]
        if not os.path.exists(os.path.join(dirname, 'context')):
            shutil.copy(
                os.path.join(egs_dir, 'info', 'left_context'), 
                os.path.join(dirname, 'context')
            )
    if context is None:
        with open(os.path.join(dirname, 'context')) as ipf:
            context = int(ipf.readline())
    model_opts.load_from_config({
        'left_context': context,
        'right_context': context
    })
    if not os.path.isfile(os.path.join(dirname, "feat_dim")):
        shutil.copy(os.path.join(egs_dir,"info","feat_dim"), dirname)
    # we start training with 
    num_archives = pkwrap.script_utils.get_egs_info(egs_dir)
    num_epochs = trainer_opts.num_epochs
#           we don't use num of jobs b/c it is 1 for now
    num_archives_to_process = num_archives*num_epochs*frame_subsampling_factor
    num_iters = (num_archives_to_process*2) // (trainer_opts.num_jobs_initial + trainer_opts.num_jobs_final)

#   TODO: for stages 5 and 6 (and possibly 7), use ChainTrainer
#   start the training    
    if stage <= 5:
        logging.info("Initializing model")
        process_out = subprocess.run([
            *cuda_cmd.split(),
            os.path.join(dirname, "log", "init.log"),
            model_file,
            "--mode", "init",
            "--dir", dirname,
            os.path.join(dirname, "0.pt")
        ])
        if process_out.returncode != 0:
            quit(process_out.returncode)

    if stage <= 6:
        train_stage = trainer_opts.train_stage
        logging.info(f"Starting training from stage={train_stage}")
        assert train_stage >= 0
        num_archives_processed = 0
        for iter_no in range(0, num_iters):
            num_jobs = pkwrap.script_utils.get_current_num_jobs(
                iter_no, 
                num_iters,
                trainer_opts.num_jobs_initial,
                1, # we don't play with num-jobs-step
                trainer_opts.num_jobs_final
            )
            if iter_no < train_stage:
                num_archives_processed += num_jobs
                continue
            assert num_jobs>0
            lr = pkwrap.script_utils.get_learning_rate(
                iter_no, 
                num_jobs, 
                num_iters, 
                num_archives_processed, 
                num_archives_to_process,
                trainer_opts.lr_initial, 
                trainer_opts.lr_final,
                schedule_type='exponential'
            )
            diagnostic_job_pool = None
            if iter_no % trainer_opts.diagnostics_interval == 0:
                diagnostic_job_pool = submit_diagnostic_jobs(dirname, model_file, iter_no, egs_dir, cuda_cmd)
            logging.info("{} Running iter={} of {} with {} jobs and lr={:.6f}".format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                iter_no,
                num_iters,
                num_jobs,
                lr
            ))
            with ThreadPoolExecutor(max_workers=num_jobs) as executor:
                    job_pool = []
                    for job_id in range(1, num_jobs+1):
                        frame_shift = num_archives_processed%frame_subsampling_factor
                        p = executor.submit(run_job,num_jobs, job_id, dirname, iter_no,
                                        model_file, lr, frame_shift, 
                                        egs_dir, num_archives, num_archives_processed,
                                        exp_cfg["minibatch_size"], cuda_cmd)
                        num_archives_processed += 1
                        job_pool.append(p)
                    for p in as_completed(job_pool):
                        if p.result() != 0:
                            quit(p.result())
            if num_jobs>1:
                model_list = [os.path.join(dirname,  "{}.{}.pt".format(iter_no, job_id)) for job_id in range(1, num_jobs+1)]
                process_out = subprocess.run([
                    *cuda_cmd.split(),
                    "{}/log/merge.{}.log".format(dirname, iter_no+1),
                    model_file,
                    "--dir", dirname,
                    "--mode", "merge",
                    "--new-model", os.path.join(dirname, "{}.pt".format(iter_no+1)),
                    ",".join(model_list)])
                for mdl in model_list:
                    pkwrap.script_utils.run(["rm", mdl])
            else:
                pkwrap.script_utils.run([
                    "mv",
                    os.path.join(dirname, "{}.1.pt".format(iter_no)),
                    os.path.join(dirname, "{}.pt".format(iter_no+1)),
                ])
            # remove old model
            if iter_no >= 20:
                mdl = os.path.join(dirname, "{}.pt".format(iter_no-10))
                pkwrap.script_utils.run(["rm", mdl])
        # do final model combination
        model_list = [
                os.path.join(dirname, f"{i}.pt")
                for i in range(num_iters, num_iters-10, -1)
        ]
        logging.info("Final model combination...")
        diagnostic_name = 'valid'
        egs_file = os.path.join(egs_dir, '{}_diagnostic.cegs'.format(diagnostic_name))
        process_out = subprocess.run([
            *cuda_cmd.split(),
            "{}/log/combine.log".format(dirname),
            model_file,
            "--dir", dirname,
            "--mode", "final_combination",
            "--new-model", os.path.join(dirname, "final.pt"),
            "--egs", "ark:{}".format(egs_file),
            ",".join(model_list)
        ])

    graph_dir = ""
    decode_params = cfg_parse[args.test_config]
    if "graph_dir" in exp_cfg:
        graph_dir = exp_cfg["graph_dir"]
    if "graph_dir" in decode_params:
        graph_dir = decode_params["graph_dir"]
    if not graph_dir:
        graph_dir = os.path.join(dirname, 'graph')
    if stage <= 7:
        if not os.path.isfile(os.path.join(graph_dir, 'HCLG.fst')):
            logging.info("Making graph with {}".format(exp_cfg["lang"]))
            pkwrap.script_utils.run([
                'utils/mkgraph.sh',
                '--self-loop-scale', '1.0',
                exp_cfg["lang"],
                tree_dir,
                graph_dir
            ])

    if stage <= 8:
        final_iter = num_iters-1
        data_dir = decode_params["test_set"]
        data_name = os.path.basename(data_dir)
        decode_iter = decode_params["iter"] if "iter" in decode_params else "final"
        decode_affix = decode_params["suffix"] if "suffix" in decode_params else ""
        decode_suff= "_iter{}{}".format(decode_iter, decode_affix)
        out_dir = os.path.join(dirname, f"decode_{data_name}{decode_suff}")
        graph = "{}/HCLG.fst".format(graph_dir)
        if "num_jobs" in decode_params:
            num_jobs = pkwrap.utils.split_data(
                data_dir, 
                int(decode_params["num_jobs"]),
            )
        else:
            num_jobs = pkwrap.utils.split_data(data_dir)
        logging.info(f"Decoding with {num_jobs} jobs...")

        ivector_opts = []
        if "ivector_dir" in decode_params and len(decode_params["ivector_dir"])>0:
            ivector_opts = ["--ivector-dir", decode_params["ivector_dir"]]

        if "apply_cmvn" in decode_params and bool(decode_params["apply_cmvn"]):
            use_cmvn = True
            cmvn_opts = decode_params["cmvn_opts"]
            utt2spk_name = "ark:{}/split{}/JOB/utt2spk".format(data_dir, num_jobs)
            feats_name = "scp:{}/split{}/JOB/feats.scp".format(data_dir, num_jobs)
            cmvn_name = "scp:{}/split{}/JOB/cmvn.scp".format(data_dir, num_jobs)
            feats_scp = "ark,s,cs:apply-cmvn {} --utt2spk={} {} {} ark:- |".format(cmvn_opts, utt2spk_name, cmvn_name, feats_name)
        else:
            feats_scp = "scp:{}/split{}/JOB/feats.scp".format(data_dir, num_jobs)
        pkwrap.script_utils.run([
            *cpu_cmd.split(),
            "JOB=1:{}".format(num_jobs),
            os.path.join(out_dir, "log", "decode.JOB.log"),
            model_file,
            "--dir", dirname,
            "--mode", "decode",
            *ivector_opts,
            "--decode-feats", feats_scp,
            os.path.join(dirname, "{}.pt".format(decode_iter)),
            "|",
            "shutil/decode/latgen-faster-mapped.sh",
            os.path.join(graph_dir, "words.txt"),
            os.path.join(dirname, "0.trans_mdl"),
            graph,
            os.path.join(out_dir, "lat.JOB.gz")
        ])
        opf = open(os.path.join(out_dir, 'num_jobs'), 'w')
        opf.write('{}'.format(num_jobs))
        opf.close()
        logging.info(f"Scoring...")
        if not os.path.isfile(os.path.join(out_dir, '../final.mdl')) and \
            os.path.isfile(os.path.join(out_dir, '../0.trans_mdl')):
            pkwrap.script_utils.run([
                    "ln",
                    "-r",
                    "-s",
                    os.path.join(out_dir, '../0.trans_mdl'),
                    os.path.join(out_dir, '../final.mdl'),
            ])
        pkwrap.script_utils.run([
            "local/score.sh",
            "--cmd", cpu_cmd,
            data_dir,
            graph_dir,
            out_dir
        ])

if __name__ == '__main__':
    train()
