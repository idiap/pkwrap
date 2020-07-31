#!/usr/bin/env python3

# Results with this script
# 2 layer TDNN, no bn, Adam, lr=0.001: valid loss was 2.8642 after 6 epochs
# 3 layer TDNN, no bn, Adam, lr=0.001: valid loss was 2.4447  with <4 epochs. Stopped training as the trend was clear.
# 3 layer TDNN, bn, Adam, lr=0.001: valid loss was 2.15 with <4 epochs

import argparse
import os
import sys
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init
import pkwrap
import numpy as np

class Net(nn.Module):
    def __init__(self, output_dim, feat_dim):
        super(Net, self).__init__()
        self.input_dim = feat_dim
        self.output_dim = output_dim
        self.tdnn1 = nn.Conv1d(feat_dim, 512, kernel_size=5, stride=1)
        self.tdnn1_bn = nn.BatchNorm1d(512, affine=False)
        self.tdnn2 = nn.Conv1d(512, 512, kernel_size=3, stride=1)
        self.tdnn2_bn = nn.BatchNorm1d(512, affine=False)
        self.tdnn3 = nn.Conv1d(512, 512, kernel_size=3, stride=1)
        self.tdnn3_bn = nn.BatchNorm1d(512, affine=False)
        self.xent_prefinal_layer = nn.Linear(512, 512)
        self.xent_layer = nn.Linear(512, output_dim)
        self.initialize()

    def initialize(self):
        init.xavier_normal_(self.tdnn1.weight)
        init.xavier_normal_(self.tdnn2.weight)
        init.xavier_normal_(self.tdnn3.weight)

    def forward(self, input): 
        mb, T, D = input.shape

        x = input.permute(0,2,1)
        x = self.tdnn1(x)
        x = F.relu(x)
        x = self.tdnn1_bn(x)

        x = self.tdnn2(x)
        x = F.relu(x)
        x = self.tdnn2_bn(x)

        x = self.tdnn3(x)
        x = F.relu(x)
        x = self.tdnn3_bn(x)

        x = x.permute(0,2,1)
        x = x.reshape(-1,512)
        pxx = F.relu(self.xent_prefinal_layer(x))
        xent_out = self.xent_layer(pxx)
        return F.log_softmax(xent_out, dim=1)

class Mls(torch.utils.data.Dataset):
    def __init__(self, feat_dict, target_dict, egs_file):
        self.feat_dict = feat_dict
        self.target_dict = target_dict
        self.chunks = []
        with open(egs_file) as ipf:
            for ln in ipf:
                self.chunks.append(ln.strip().split()[0])

    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        s = chunk.split('-')
        start_frame = int(s[-1])
        n = '-'.join(s[:-1])
        f = self.feat_dict[n]
        chunk_size = 8
        t = target_dict[n][start_frame:start_frame+chunk_size]
        fnp = np.pad(f.numpy(), [(20,20), (0,0)], mode='edge')
        f = torch.tensor(fnp)
        start_frame += 20
        x = f[start_frame-4:start_frame+chunk_size+4]
        t = torch.tensor(t)
        return x, t
        


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--mode")
    args = parser.parse_args()

    dirname = './'
    if args.mode == "init":
        model = Net(2016, 40)
        torch.save(model.state_dict(), os.path.join(dirname, "0.pt"))
    if args.mode == "train":
        num_iters = 20
        feat_file = 'scp:data/train_clean_5_sp_hires/feats.scp'
        feat_dict = {}
        target_dict = {}
        r = pkwrap.script_utils.feat_reader(feat_file)
        while not r.Done():
            k = r.Key()
            feat_dict[k] = pkwrap.kaldi.matrix.KaldiMatrixToTensor(r.Value())
            m = feat_dict[k].mean(0)
            v = feat_dict[k].var(0)
            feat_dict[k] = (feat_dict[k]-m)
            r.Next()
        print("Read all features")
        for i in range(1,21):
            ali_name = 'exp/tri3b_ali_train_clean_5_sp/ali.{}.txt'.format(i)
            with open(ali_name) as ipf:
                for ln in ipf:
                    lns = ln.strip().split()
                    n = lns[0]
                    v = list(map(int, lns[1:]))
                    target_dict[n] = v
        lr = 0.001
        model = Net(2016, 40)
        base_model = '{}.pt'.format(0)
        model.load_state_dict(torch.load(base_model))
        model = model.cuda()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        for i in range(0, 6):
            #logf = open('log/{}.log'.format(i),'w')
            print("Starting iter={}".format(i))
            dataset = Mls(feat_dict, target_dict, 'exp/chain/tdnn_sp/egs_ce/egs.scp')
            loader = torch.utils.data.DataLoader(dataset, batch_size=128)
            lr = lr/2
            model.train()
            for idx, (feat, target) in enumerate(loader):
                # feat_i = feat.permute(0,2,1)
                feat_i = feat
                feat_i = feat_i.cuda()
                #target_t = torch.cat(target)
                target_t = target.reshape(-1).cuda()
                x = model(feat_i)
                loss = F.nll_loss(x, target_t)
                if idx%20 == 0:
                    print(idx, loss.item())
                    sys.stdout.flush()
                if idx%100 == 0:
                    prediction = x.argmax(1).cpu()
                    acc = torch.eq(prediction, target_t.cpu()).sum()
                    #acc = acc/float(target_t.shape[0])
                    print("Accuracy={}".format(acc))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if idx>0 and idx%100 == 0: # validate
                    norms = []
                    for p in model.parameters():
                        if p.grad.data is not None:
                            norms.append(p.grad.data.norm(2))
                    print("Norms", norms)
                    with torch.no_grad():
                        model.eval()
                        # acc = 0
                        nt = 0
                        utt_list = [utt.strip() for utt in open('exp/chain/tdnn_sp/egs/valid_uttlist')]
                        valid_loss = None
                        for utt in utt_list:
                            utt = utt.strip()
                            feat = feat_dict[utt] #.numpy()
                            # add left and right context before testing
                            #feat = np.pad(feat, [(2,2), (0,0)], mode='edge')
                            d = feat.shape[-1]
                            left_context = feat[0,:].repeat(4).reshape(-1,d)
                            right_context = feat[-1,:].repeat(4).reshape(-1,d)
                            feat = torch.cat([left_context, feat, right_context])
                            #feat = torch.tensor(feat)
                            feat = feat.unsqueeze(0).cuda()
                            tgt = target_dict[utt]
                            nt += len(tgt)
                            y = model(feat) #.cpu().squeeze(0)
                            valid_loss_ = F.nll_loss(y, torch.tensor(tgt).cuda())
                            if valid_loss is None:
                                valid_loss = valid_loss_
                            else:
                                valid_loss += valid_loss_
                            # predictions = y.argmax(1)
                            # acc_ = torch.eq(predictions, torch.tensor(tgt)).sum()
                            # acc += acc_.tolist()
                            # sys.stdout.flush()
                            # pred_list = set([x.tolist() for x in predictions[:]])
                    model.train()
                    # print("Validation acc=", float(acc)/float(nt))
                    print("Valid loss=", valid_loss/len(utt_list))


            # model = model.cpu()
            # torch.save(model.state_dict(), os.path.join(dirname, "{}.pt".format(i+1)))
    if args.mode == "test":
        feat_file = 'scp:data/train_clean_5_sp_hires/feats.scp'
        feat_dict = {}
        target_dict = {}
        r = pkwrap.script_utils.feat_reader(feat_file)
        while not r.Done():
            feat_dict[r.Key()] = pkwrap.kaldi.matrix.KaldiMatrixToTensor(r.Value())
            r.Next()
        print("Read all features")
        for i in range(1,21):
            ali_name = 'exp/tri3b_ali_train_clean_5_sp/ali.{}.txt'.format(i)
            with open(ali_name) as ipf:
                for ln in ipf:
                    lns = ln.strip().split()
                    n = lns[0]
                    v = list(map(int, lns[1:]))
                    target_dict[n] = v
        model = Net(2016, 40)
        base_model = '{}.pt'.format(0)
        model.load_state_dict(torch.load(base_model))
        model.eval()
        dataset = Mls(feat_dict, target_dict, 'exp/chain/tdnn_sp/egs_ce/egs.scp')
        loader = torch.utils.data.DataLoader(dataset, batch_size=128)
        for idx, (feat, target) in enumerate(loader):
            feat_i = feat
            #target_t = torch.cat(target)
            #target_t = target
            x = model(feat_i)
            print(x.shape)
            print(F.nll_loss(x, target.reshape(-1)))
            quit(0)
