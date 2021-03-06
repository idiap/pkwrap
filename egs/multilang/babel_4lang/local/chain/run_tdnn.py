#!/usr/bin/env python

description = """
  This script trains and tests a simple tdnn system.
"""

import torch
import torch.nn as nn
import pkwrap
import argparse
import subprocess
import configparser
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_job(num_jobs, job_id, dirname, iter_no, model_file, lr, frame_shift, egs_dir,
    num_archives, num_archives_processed, minibatch_size, context_offset ):
    """
        sub a single job and let ThreadPoolExecutor monitor its progress
    """
    log_file = "{}/log/train.{}.{}.log".format(dirname, iter_no, job_id)
    process_out = subprocess.run(["queue.pl", "-l", "q_short_gpu", "-l", "h=!vgng*", "-V", 
                log_file,
                model_file,
                "--dir", dirname,
                "--mode", "training",
                "--lr", str(lr),
                "--frame-shift", str(frame_shift),
                "--egs", "ark:{}/cegs.{}.ark".format(egs_dir, num_archives_processed%num_archives+1),
                "--l2-regularize-factor", str(1.0/num_jobs),
                "--minibatch-size", "64,32",
                "--context-offset", str(context_offset),
                "--new-model", os.path.join(dirname, "{}.{}.pt".format(iter_no, job_id)),
                os.path.join(dirname, "{}.pt".format(iter_no))])
    return process_out.returncode

def run_diagnostic(arg_arr):
    process_out = subprocess.run(arg_arr)
    return process_out.returncode

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    pkwrap.script_utils.add_chain_recipe_opts(parser)
    parser.add_argument('--decode-data', default='')
    parser.add_argument('--graph-dir', default='')
    parser.add_argument('--left-context', default=0)
    parser.add_argument('--right-context', default=0)
    parser.add_argument('model_file')
    args = parser.parse_args()
    cfg_parse = configparser.ConfigParser()
    cfg_parse.read("config")
    cmd = cfg_parse["cmd"]
    affix = args.affix
    model_file = args.model_file
    if args.decode_data:
        decode_data = args.decode_data

# gmm_dir=exp/$gmm
    data = args.data
    exp = args.exp
    stage = args.stage
    gmm_dir = os.path.join(exp, args.gmm)
    ali_dir = os.path.join(exp, "_".join([args.gmm, "ali", args.train_set, "sp"]))
    tree_dir = os.path.join(exp, "chain", "tree_sp{}".format(args.tree_affix))
    lat_dir = os.path.join(exp, "chain", "{}_{}_sp_lats".format(args.gmm, args.train_set))
    dirname = os.path.join(exp, "chain", "tdnn{}_sp".format(affix))
    train_data_dir = os.path.join(data, "{}_sp_hires".format(args.train_set))
    lores_train_data_dir = os.path.join(data, "{}_sp".format(args.train_set))
    train_cmd = cmd['train_cmd']
    cuda_cmd = cmd['cuda_cmd']

    if stage <= 0:
        old_lang = os.path.join(data, "lang")
        new_lang = os.path.join(data, "lang_chain")
        try:
            rc = subprocess.run(["shutil/chain/check_lang.sh", old_lang, new_lang]).returncode
        except Exception as e:
            #           TODO: should use logging
            sys.stderr.write(e)
            sys.stderr.write("ERROR: copying lang failed")
    if stage <= 1:
        try:
            subprocess.run(["steps/align_fmllr_lats.sh", "--nj", "{}".format(args.decode_nj),
                            "--cmd", "{}".format(train_cmd),
                            lores_train_data_dir,
                            "{}/lang".format(data),
                            gmm_dir,
                            lat_dir])
            #subprocess.run(["rm {}/fsts.*.gz".format(lat_dir)])
        except Exception as e:
            sys.stderr.write(e)
            sys.stderr.write("ERROR: lattice creationg failed")
    if stage <= 2:
        tree_file = os.path.join(tree_dir, "tree")
        if os.path.isfile(tree_file):
            sys.stderr.write("ERROR: tree file {} already exists."
                " Refusing to overwrite".format(tree_file))
            quit(1)
        if not os.path.isfile(os.path.join(tree_dir, '.done')):
            cmd = ["steps/nnet3/chain/build_tree.sh",
                "--frame-subsampling-factor",
                "{}".format(args.frame_subsampling_factor),
                '--context-opts',
                "--context-width=2 --central-position=1",
                "--cmd",
                "{}".format(train_cmd),
                "3500",
                "{}".format(lores_train_data_dir),
                "{}/lang_chain".format(data),
                "{}".format(ali_dir),
                "{}".format(tree_dir)]
            subprocess.run(cmd)
            subprocess.run(["touch", "{}/.done".format(tree_dir)])

    config_dir = os.path.join(dirname, "configs")
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    if not os.path.isdir(config_dir):
        os.mkdir(config_dir)
    init_dir = os.path.join(dirname, "init")
    if not os.path.isdir(init_dir):
        os.mkdir(init_dir)
    learning_rate_factor = 0.5/args.xent_regularize
#   create den.fst
    if stage <= 3:
        subprocess.run([
            "shutil/chain/make_den_fst.sh",
            "--cmd",
            "{}".format(train_cmd),
            tree_dir, gmm_dir, dirname])
    egs_dir = os.path.join(dirname, "egs")
    if stage <= 4:
        process_out = subprocess.run(["steps/chain/get_egs.sh",
        "--cmd", train_cmd,
        "--cmvn-opts", "--norm-means=false --norm-vars=false",
        "--left-context", args.left_context,
        "--right-context", args.right_context,
        "--frame-subsampling-factor", str(args.frame_subsampling_factor),
        "--alignment-subsampling-factor", str(args.frame_subsampling_factor),
        "--frames-per-iter", str(args.frames_per_iter),
        "--frames-per-eg", str(args.chunk_width),
        "--srand", str(args.srand),
        train_data_dir, dirname, lat_dir, egs_dir])
        if process_out.returncode != 0:
            quit(process_out.returncode)
    # we start training with 
    num_archives = pkwrap.script_utils.get_egs_info(egs_dir)
    fpi = args.frames_per_iter
#           we don't use num of jobs b/c it is 1 for now
    num_archives_to_process = num_archives*args.num_epochs*args.frame_subsampling_factor
    num_iters = (num_archives_to_process*2) // (args.num_jobs_initial + args.num_jobs_final)
    if stage <= 5:
        num_pdfs = subprocess.check_output(["cat", os.path.join(egs_dir, "info", "num_pdfs")]).decode().strip()
        subprocess.run(["cp", os.path.join(egs_dir,"info","feat_dim"), dirname])
        with open(os.path.join(dirname, 'num_pdfs'), 'w') as opf:
            opf.write(num_pdfs)
            opf.close()
        process_out = subprocess.run(["queue.pl", "-l", "q_gpu", "-V",
            os.path.join(dirname, "log", "init.log"),
            model_file,
            "--mode", "init",
            "--dir", dirname,
            os.path.join(dirname, "0.pt")])
        if process_out.returncode != 0:
            quit(process_out.returncode)
    if stage <= 6:
        train_stage = args.train_stage
        # get contexts
        egs_info = os.path.join(egs_dir, "info")
        left_context = int(subprocess.check_output("cat {}".format(os.path.join(egs_info, "left_context")), shell=True).decode().strip())
        left_context_intial = int(subprocess.check_output("cat {}".format(os.path.join(egs_info, "left_context_initial")), shell=True).decode().strip())
        right_context = int(subprocess.check_output("cat {}".format(os.path.join(egs_info, "right_context")), shell=True).decode().strip())
        right_context_final = int(subprocess.check_output("cat {}".format(os.path.join(egs_info, "right_context_final")), shell=True).decode().strip())
        context_offset = left_context+right_context+left_context_intial+right_context_final-1

        assert train_stage >= 0
        num_archives_processed = 0
        for iter_no in range(0, num_iters):
            num_jobs = pkwrap.script_utils.get_current_num_jobs(iter_no, 
                                                                num_iters,
                                                                args.num_jobs_initial,
                                                                1, # we don't play with num-jobs-step
                                                                args.num_jobs_final)
            if iter_no < train_stage:
                num_archives_processed += num_jobs
                continue
            assert num_jobs>0
            sys.stderr.flush()        
            lr = pkwrap.script_utils.get_learning_rate(iter_no, 
                                                       num_jobs, 
                                                       num_iters, 
                                                       num_archives_processed, 
                                                       num_archives_to_process,
                                                       args.lr_initial, 
                                                       args.lr_final,
                                                       schedule_type='exponential')
            sys.stderr.write("Running iter={} of {} lr={} \n".format(iter_no, num_iters, lr))
            with ThreadPoolExecutor(max_workers=num_jobs+1) as executor:
                    job_pool = []
                    sys.stderr.write("Num jobs = {}\n".format(num_jobs))
                    sys.stderr.flush()
                    for job_id in range(1, num_jobs+1):
                        frame_shift = num_archives_processed%args.frame_subsampling_factor
                        p = executor.submit(run_job,num_jobs, job_id, dirname, iter_no,
                                        model_file, lr, frame_shift, egs_dir, num_archives, num_archives_processed,
                                        "64,32", context_offset)
                        num_archives_processed += 1
                        job_pool.append(p)
                    if iter_no <= 5 or iter_no % 10 == 0:
                        p = executor.submit(run_diagnostic, ["queue.pl", "-l", "q_short_gpu", "-l", "h=!vgng*","-V",
                                                    f"{dirname}/log/diagnostics_valid.{iter_no}.log",
                                                    model_file,
                                                    "--dir", dirname,
                                                    "--mode", "diagnostic",
                                                    "--egs", f"ark:{egs_dir}/valid_diagnostic.cegs",
                                                    os.path.join(dirname, "{}.pt".format(iter_no))
                                                    ])
                        job_pool.append(p)
                    for p in as_completed(job_pool):
                        if p.result() != 0:
                            quit(p.result())
            model_list = [os.path.join(dirname,  "{}.{}.pt".format(iter_no, job_id)) for job_id in range(1, num_jobs+1)]
            process_out = subprocess.run(["queue.pl", "-l", "q_short_gpu", "-l", "h=!vgng*", "-V", 
                "{}/log/merge.{}.log".format(dirname, iter_no+1),
                model_file,
                "--dir", dirname,
                "--mode", "merge",
                "--new-model", os.path.join(dirname, "{}.pt".format(iter_no+1)),
                ",".join(model_list)])
            if process_out.returncode != 0:
                quit(process_out.returncode)
            else:
                for mdl in model_list:
                    subprocess.run(["rm", mdl])
            if process_out.returncode != 0:
                quit(process_out.returncode)
        src = os.path.join(dirname, "{}.pt".format(num_iters-1))
        dst = os.path.join(dirname, "final.pt")
        subprocess.run(['ln', '-r', '-s', src, dst])
    if stage <= 7:
        final_iter = num_iters-1
        data_name = decode_data
        decode_dir = "data/{}_hires".format(data_name)
        decode_iter = "final"
        decode_suff= "_iter{}".format(decode_iter)
        out_dir = os.path.join(dirname, "decode_{}{}".format(decode_data,decode_suff))
        graph_dir = args.graph_dir
        graph = "{}/HCLG.fst".format(graph_dir)
        num_jobs = pkwrap.utils.split_data(decode_dir)
        subprocess.run(["queue.pl", "-l", "q1d", "-l", "io_big", "-V",
            "JOB=1:{}".format(num_jobs),
            os.path.join(out_dir, "log", "decode.JOB.log"),
            model_file,
            "--dir", dirname,
            "--mode", "decode",
            "--decode-feat", "scp:{}/split{}/JOB/feats.scp".format(decode_dir, num_jobs),
            os.path.join(dirname, "{}.pt".format(decode_iter)),
            "|",
            "shutil/decode/latgen-faster-mapped.sh",
            "data/lang_chain/words.txt",
            os.path.join(dirname, "0.trans_mdl"),
            graph,
            os.path.join(out_dir, "lat.JOB.gz")
            ])
        opf = open(os.path.join(out_dir, 'num_jobs'), 'w')
        opf.write('{}'.format(num_jobs))
        opf.close()
        subprocess.run(["shutil/decode/score.sh",
        "--cmd", "queue.pl -l q1d -l io_big -V",
        decode_dir,
        graph_dir,
        out_dir
        ])
