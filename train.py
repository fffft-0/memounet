import os
import numpy as np
import torch
from matplotlib.colors import BoundaryNorm, Normalize
from torch import optim
from torch.utils.data import DataLoader
import time
from torch.cuda.amp import autocast, GradScaler
from tool import *
import json
from dataprovider.nextdata import datancMuti
from torch import optim
from einops import rearrange, repeat
import logging as log
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.transforms import RandomErasing
from model import MemoUnet
from metrics import metric, R2
from torcheval.metrics.functional import r2_score
from mpl_toolkits.basemap import Basemap
torch.cuda.empty_cache()

std2 = 1.3902251
mean2 = 3.3787999


std1 = 1.04575
mean1 = 2.63117

realstd = std1
realmean = mean1
class trainframe:
    def __init__(self, batch_size=48, en_channels=8, de_channels=1, image_size=256, workpath="./",
                 device=0, name="?", epochs=6, train=True, branch=1, var_num=7, var_loss_weight=None,
                 patiences=10, jsonpath=None, period=20, lr=0.001, selector=None, criterion=None,
                 dropout=0.1, model=None, itrlossprint=True, iterprintnum=10, embedtime=False,
                 memory=None, halftrain=True, percept_path="./percept.pth", accumulate_step=1, percept=False):
        self.device = torch.device('cuda', device)
        if model is not None:
            self.model = model(in_channels=en_channels, var_num=var_num, out_channels=de_channels,
                               image_size=image_size, batch_size=batch_size)
            self.model.to(self.device)
        self.discriminator = None
        self.perceptual_loss = None
        self.image_size = image_size
        # self.discriminator = discri.to(self.device)
        self.en_channels = en_channels
        self.de_channels = de_channels
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.iterprintnum = iterprintnum
        self.itrlossprint = itrlossprint
        self.accumulate_step = accumulate_step
        self.var_num = var_num
        self.selector = selector
        self.period = period
        self.criterion = criterion
        self.patiences = patiences
        self.embedtime = embedtime
        self.halftrain = halftrain
        self.percept = percept
        self.disc_start = None
        self.dataset = None
        self.data = None
        self.setting = name + "_in{}_out{}_b{}_ep{}_v{}_lr{}_s{}".format(en_channels, de_channels, batch_size,
                                                                         epochs, var_num, lr, accumulate_step)
        self.name = name
        self.old_bs = batch_size
        self.checkpath = None
        self.init_loss = None
        self.jsonpath = None
        self.outputpath = None
        self.modelpath = None
        self.paraldata = None
        self.inparal = False
        self.drawtrainbin = None
        self.drawvalibin = None
        self.drawpath = None
        self.memorypath = None
        self.percept_path = percept_path
        self.augmentation = RandomErasing(0.6, (0.05, 0.2), (0.33, 3), 0, True)
        self.buildPath(workpath)

    def get_percept(self):
        percept_loss = percept(in_channels=8, image_size=self.image_size, batch_size=self.batch_size)
        percept_loss.load_state_dict(torch.load(self.percept_path))
        for param in percept_loss.parameters():
            param.requires_grad = False
        return percept_loss

    def modelgrouptrain(self, models, device_ids=None, main_device=0, inparal=True, itrlossprint=False, train=True):
        if device_ids is None:
            device_ids = [0]
        ls = len(models)
        ls_lr = len(lrs)
        ls_bss = len(bss)
        ls_eps = len(eps)
        i = 0
        self.device = torch.device('cuda', main_device)
        while i < ls:
            torch.cuda.empty_cache()
            self.init_loss = torch.inf
            if i < ls_lr:
                self.lr = lrs[i]
            if i < ls_bss:
                self.batch_size = bss[i]
            if i < ls_eps:
                self.epochs = eps[i]
            self.setting = (f"{models[i].__name__}{i}_{self.name}_in{self.en_channels}_out{self.de_channels}_"
                            f"b{self.batch_size}_ep{self.epochs}_v{self.var_num}_lr{self.lr}_s{self.accumulate_step}")
            self.jsonpath = f"{self.checkpath}{self.setting}_record.json"
            self.modelpath = f"{self.checkpath}{self.setting}.pth"
            self.itrlossprint = itrlossprint
            print(f"transfer to new model setting: {self.setting}")
            self.model = models[i](in_channels=self.en_channels, var_num=self.var_num, out_channels=self.de_channels,
                                   image_size=self.image_size, batch_size=self.batch_size)
            if self.percept:
                self.perceptual_loss = self.get_percept()
            # self.discriminator = discriminator()
            if inparal:
                self.paralleltrain(device_ids=device_ids, main_device=main_device, train=train)
            elif train is True:
                self.model = self.model.to(self.device)
                if self.percept:
                    self.perceptual_loss = self.perceptual_loss.to(self.device)
                    # self.discriminator = self.discriminator.to(self.device)
                self.train()
            i = i + 1

    def specifyload(self, modestr):
        dataset = datancMuti(en_channels=self.en_channels, de_channels=self.de_channels, mode_str=modestr,
                             selector=self.selector, embedtime=self.embedtime)
        return DataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=6,
            drop_last=False)

    def lazyload(self):
        if self.dataset is None:
            self.dataset = {
                'train': datancMuti(en_channels=self.en_channels, de_channels=self.de_channels, mode_str="train",
                                    selector=self.selector, embedtime=self.embedtime),
                'valid': datancMuti(en_channels=self.en_channels, de_channels=self.de_channels, mode_str="valid",
                                    selector=self.selector, embedtime=self.embedtime),
                'test': datancMuti(en_channels=self.en_channels, de_channels=self.de_channels, mode_str="test",
                                   selector=self.selector, embedtime=self.embedtime)
            }
        if self.data is None or self.old_bs != self.batch_size:
            self.data = {
                'train': DataLoader(
                    dataset=self.dataset['train'],
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=6,
                    drop_last=True),
                'valid': DataLoader(
                    dataset=self.dataset['valid'],
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=6,
                    drop_last=True),
                'test': DataLoader(
                    dataset=self.dataset['test'],
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=6,
                    drop_last=True)
            }
            self.old_bs = self.batch_size

    def buildPath(self, workpath):
        if workpath[-1] != '/':
            workpath += '/'
        self.checkpath = mkdir(f"{workpath}checkpoints/")
        self.jsonpath = f"{self.checkpath}{self.setting}_record.json"
        self.outputpath = mkdir(f"{workpath}hiddenFeature/")
        self.modelpath = f"{self.checkpath}{self.setting}.pth"
        self.drawpath = f"{self.checkpath}{self.setting}_loss.png"
        self.memorypath = f"{self.checkpath}{self.setting}_memory.npy"

    def loadModelInitLoss(self, path, best=None):
        self.model.load_state_dict(torch.load(path))
        self.init_loss = best if best is not None else self.vali(self.data["valid"], self.criterion)
        print(f"init best loss: {self.init_loss}")

    def paralleltrain(self, device_ids=None, main_device=0, train=True):
        if device_ids is None:
            device_ids = [0, 1]
        device_ids.remove(main_device)
        device_ids.insert(0, main_device)
        print(f"main device: {self.device} | devices list: {device_ids}")
        self.model = torch.nn.DataParallel(self.model.to(self.device), device_ids=device_ids)
        # self.discriminator = torch.nn.DataParallel(self.discriminator.to(self.device), device_ids=device_ids)
        if self.percept:
            self.perceptual_loss = torch.nn.DataParallel(self.perceptual_loss.to(self.device), device_ids=device_ids)
        self.inparal = True
        self.train()

    def ddptrain(self, device_ids=None):
        self.lazyload()
        if device_ids is None:
            device_ids = [0, 1]
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://'
        )
        train_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset["train"])
        valid_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset["valid"])
        test_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset["test"])
        self.data = {
            'train': torch.utils.data.DataLoader(self.dataset["train"], batch_size=self.batch_size,
                                                 num_workers=6, drop_last=False, sampler=train_sampler),
            'valid': torch.utils.data.DataLoader(self.dataset["valid"], batch_size=self.batch_size,
                                                 num_workers=6, drop_last=False, sampler=valid_sampler),
            'test': torch.utils.data.DataLoader(self.dataset["test"], batch_size=self.batch_size,
                                                num_workers=6, drop_last=False, sampler=test_sampler),
        }
        self.device = torch.device('cuda', 0)
        model = torch.nn.parallel.DistributedDataParallel(self.model.to(self.device), device_ids=[device_ids],
                                                          output_device=device_ids)
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.inparal = True
        self.train()

    def adopt_weight(self, disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    # def calculate_lambda(self, perceptual_loss, gan_loss):
    #     last_layer = self.model.out
    #     last_layer_weight = last_layer.weight
    #     perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]
    #     gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]
    #
    #     ratio = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-6)
    #     ratio = torch.clamp(ratio, 0, 1e3).detach()
    #     return ratio*0.5

    def prehandle(self, data):
        for i in range(self.batch_size):
            for j in range(self.var_num):
                for k in range(self.en_channels):
                    self.augmentation(data[i, j, k].unsqueeze(0))

    def get_perceptual_loss(self, fake, real):
        fakes = self.perceptual_loss(fake)
        reals = self.perceptual_loss(real)
        loss = 0
        for i in range(len(fakes)):
            loss = loss + self.criterion(fakes[i], reals[i])
        return loss

    def updatememoinfo(self, memoinfo):
        indexes = memoinfo["index"]
        memodecode_loss = memoinfo["memodecode_loss"]
        origin = memodecode_loss.mean().detach()
        indices_num = indexes.numel()
        rank = torch.bincount(indexes)
        count, countindex = torch.sort(rank, descending=True)
        boundary = count[:-1] - count[1:]
        if boundary.numel() > 1:
            boundary = boundary.argmax() + 1
            diff = count[boundary - 1] - count[boundary]
        else:
            boundary = torch.tensor(1).to(self.device)
            diff = count[0]
        bounds = countindex[:boundary]
        mask = torch.ones_like(indexes, dtype=torch.bool)
        for bound in bounds:
            mask = mask & (indexes != bound)
        mask = mask.view(memodecode_loss.shape[:3] + (1,))
        less_attn_num = mask.sum()
        code_num = (count != 0).sum()
        weight = mask.float()
        val = 0.6
        diff = diff / indices_num
        weight[weight == 0] = val
        memodecode_loss = memodecode_loss * weight
        memodecode_loss = memodecode_loss.mean()
        memoinfo.update(
            {
                "memodecode_loss": memodecode_loss,
                "boundary": boundary,
                "code_num": code_num,
                "diff_percent": diff,
                "more_wwight_val": val
            }
        )

    def train(self):
        self.lazyload()
        countParas(self.model, "generator")
        # countParas(self.discriminator, "discriminator")
        time_train_start = time.time()
        print(f"in parallel is {self.inparal}")
        print('>>>>>>>start training : {} | time : {}>>>>>>>>>>>'.format(self.setting,
                                                                         time.strftime('%Y-%m-%d %H:%M:%S',
                                                                                       time.localtime(
                                                                                           time_train_start))))
        train_steps = len(self.data['train'])
        early_stopping = EarlyStopping(verbose=True, patience=self.patiences, best_score=self.init_loss)
        criterion = self.criterion
        train_epochs = self.epochs
        scaler = GradScaler()
        best_dict = {}
        best_vali = torch.inf if self.init_loss is None else self.init_loss
        generate_optim = torch.optim.AdamW(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), weight_decay=0)
        dataloader = self.data['train']
        steps_per_epoch = len(dataloader)
        num_steps = steps_per_epoch * self.period
        gan_lr = 0.0001
        # discriminate_optim = torch.optim.AdamW(self.discriminator.parameters(), lr=gan_lr, betas=(0.9, 0.999),
        #                                        weight_decay=0)
        # self.disc_start = 1000
        print(f"num_steps: {num_steps} | warmup iterations: {int(num_steps * 0.02)}")
        for epochi in range(train_epochs):
            iter_count = 0
            mse_loss_bin = []
            final_loss_bin = []
            memo_loss_bin = []
            bound_bin = []
            num_bin = []
            diff_bin = []
            gan_loss_bin = []
            self.model.train()
            epoch_time = time.time()
            time_now = time.time()
            for i, (x, y, t) in enumerate(dataloader):
                """
                batchsize, var, time, width, height
                """
                iter_count += 1
                if self.halftrain:
                    x = x.to(self.device)
                    with autocast():
                        pred = self.model(x)
                    y = y[:, 0].to(self.device)
                    loss = criterion(pred, y)
                    # nearloss = criterion(near, y)
                    allloss = loss
                    # allloss = (loss*self.var_loss_weight).sum()
                    scaler.scale(allloss).backward()
                    # 4.1 update parameters of net
                    scaler.step(generate_optim)
                    scaler.update()
                else:
                    # self.prehandle(x)
                    x = x.float().to(self.device)
                    t = t[:, 0].long().to(self.device)
                    pred, memo_info = self.model(x, t)
                    self.updatememoinfo(memo_info)
                    q_loss = memo_info["memodecode_loss"]
                    boundary = memo_info["boundary"]
                    num = memo_info["code_num"]
                    diff = memo_info["diff_percent"]
                    y = y[:, 0].float().to(self.device)
                    loss = criterion(pred, y)
                    allloss = 0
                    if self.percept:
                        perceptual_loss = self.get_perceptual_loss(pred, y)
                        while perceptual_loss >= loss:
                            perceptual_loss = perceptual_loss * 0.5
                        allloss = perceptual_loss

                    allloss = loss + allloss + q_loss
                    # if epochii == 0:
                    #     allloss = q_loss*(0.0006/self.lr) + loss*0
                    #     epochi = 0
                    # else:
                    #     allloss = loss + allloss + q_loss*(0.0006/self.lr)
                    #     epochi = epochii - 1
                    # allloss = loss + q_loss.mean()

                    # disc_factor = self.adopt_weight(1, epochi * steps_per_epoch + i, threshold=self.disc_start)
                    # g_loss = 0
                    # gan_loss = 0
                    # if disc_factor != 0:
                    #     disc_real = self.discriminator(y)
                    #     disc_fake = self.discriminator(pred)
                    #     d_loss_real = torch.mean(F.relu(1. - disc_real))
                    #     d_loss_fake = torch.mean(F.relu(1. + disc_fake))
                    #     gan_loss = 0.5 * (d_loss_real + d_loss_fake)
                    #     g_loss = -torch.mean(disc_fake)
                    #     # ratio = self.calculate_lambda(allloss, g_loss)
                    #     # g_loss = ratio * g_loss

                    # allloss = allloss + g_loss*0.5
                    allloss = allloss / self.accumulate_step
                    allloss.backward()

                    if (i + 1) % self.accumulate_step == 0:
                        generate_optim.step()
                        generate_optim.zero_grad()
                    # if disc_factor != 0:
                    #     allloss.backward(retain_graph=True)
                    #     discriminate_optim.zero_grad()
                    #     gan_loss.backward()
                    #     discriminate_optim.step()
                    #     gan_loss_bin.append([d_loss_fake.item(), d_loss_real.item()])
                    # else:
                    #     allloss.backward()
                    # generate_optim.step()
                memo_loss_bin.append(q_loss.item())
                diff_bin.append(diff.item())
                num_bin.append(num.item())
                bound_bin.append(boundary.item())
                mse_loss_bin.append(loss.item())
                final_loss_bin.append(allloss.item() * self.accumulate_step)
                adjust_lr_cos_warm(generate_optim, epochi * len(dataloader) + i + 1, num_steps, warmup=True,
                                   lr_max=self.lr)
                # adjust_lr_cos_warm(memo_optim, epochi * len(dataloader) + i + 1, num_steps, warmup=False,
                #                    lr_max=memo_lr)
                # adjust_lr_cos_warm(discriminate_optim, epochi * len(dataloader) + i + 1, num_steps, warmup=False,
                #                    lr_max=gan_lr)
                if i % (len(dataloader) // self.iterprintnum) == 0 and self.itrlossprint:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f} all_loss: {3:.7f} memo_loss: {4:.7f} bound: {5: d} num: {6: d} diff: {7: .5f}".format(
                            i + 1, epochi + 1,
                            loss.item(), allloss.item() * self.accumulate_step, q_loss.item(),
                            boundary.item(), num.item(), diff.item()))
                    # if disc_factor != 0:
                    #     print("\t\tgan | real_loss: {0:.7f} fake_loss: {1:.7f}".format(d_loss_real.item(), d_loss_fake.item()))
                    # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epochi + 1,
                    #                                                         loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((train_epochs - epochi) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    print('\tlearning rate: {:.3e}'.format(generate_optim.state_dict()['param_groups'][0]['lr']))
                    time_now = time.time()
                    iter_count = 0
            print("Epoch: {} cost time: {}".format(epochi + 1, time.time() - epoch_time))
            # adjust_lr_exp(optimizer, epochi, self.lr)
            mse_loss_bin = np.average(mse_loss_bin)
            final_loss_bin = np.average(final_loss_bin)
            memo_loss_bin = np.average(memo_loss_bin)
            bound_bin = np.average(bound_bin)
            num_bin = np.average(num_bin)
            diff_bin = np.average(diff_bin)
            # gan_loss_bin = np.average(gan_loss_bin, axis=0)
            vali_loss = self.vali(self.data["valid"], criterion)
            test_loss = self.vali(self.data["test"], criterion)
            # test_all_loss = torch.inf
            if vali_loss < best_vali:
                best_vali = vali_loss
                best_dict.update({
                    'train': "{:.5f}".format(mse_loss_bin),
                    'vali': "{:.5f}".format(vali_loss),
                    'test': "{:.5f}".format(test_loss),
                    # 'allloss': "{:.5f}".format(all_loss),
                    'epoch': epochi + 1
                })
                self.record_json({
                    'best': best_dict,
                })
            else:
                print(f"best loss: {best_dict}")
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f} all loss: {5:.7f} memo_loss: {6:.7f} bound: {7:.7f} num: {8:.7f} diff: {9:.7f}".format(
                    epochi + 1, train_steps, mse_loss_bin, vali_loss, test_loss,
                    final_loss_bin, memo_loss_bin, bound_bin, num_bin, diff_bin))
            # if disc_factor != 0:
            #     print("gan | real_loss: {0:.7f} fake_loss: {1:.7}".format(gan_loss_bin[1], gan_loss_bin[0]))
            # print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
            #     epochi + 1, train_steps, mse_loss_bin, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, self.modelpath)
            self.record_json({
                'epoch' + str(epochi + 1) + "_mseloss": {
                    'train': "{:.5f}".format(mse_loss_bin),
                    'vali': "{:.5f}".format(vali_loss),
                    'test': "{:.5f}".format(test_loss),
                    # 'allloss': "{:.5f}".format(all_loss),
                }
            })
            if early_stopping.early_stop:
                print("Early stopping")
                break
        # self.model.load_state_dict(torch.load(self.modelpath))
        print('>>>>>>>end training : {} | time : {}>>>>>>>>>>>\n\n\n'.format(self.setting,
                                                                             time.strftime('%Y-%m-%d %H:%M:%S',
                                                                                           time.localtime(
                                                                                               time.time()))))

    def vali(self, vali_loader, criterion):
        self.lazyload()
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (x, y, t) in enumerate(vali_loader):
                if self.halftrain:
                    x = x.to(self.device)
                    with autocast():
                        pred = self.model(x)
                    loss = criterion(pred, y[:, 0].to(self.device))
                else:
                    x = x.float().to(self.device)
                    t = t[:, 0].long().to(self.device)
                    pred, memo_info = self.model(x, t)
                    pred = pred * realstd + realmean
                    y = y * realstd + realmean
                    loss = criterion(pred, y[:, 0].float().to(self.device))
                total_loss.append(loss.item())
                # all_loss.append((loss.sum().item()))
        total_loss = np.average(total_loss)
        # all_loss = np.average(all_loss)
        self.model.train()
        return total_loss

    def saveModel(self, path):
        torch.save(self.model.state_dict(), path)

    def record_json(self, dict):
        writeJson(self.jsonpath, dict)

    def read_one_json(self, key: str):
        with open(self.jsonpath, "a+", encoding='utf-8') as f:
            f.seek(0, os.SEEK_SET)
            try:
                data = json.load(f)
                return data[key]
            except:
                return None

    def test(self, dataload=None, lossf=None, modelp=None):
        self.lazyload()
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(self.setting))
        self.model.eval()
        path = self.modelpath if modelp is None else modelp
        self.model.load_state_dict(torch.load(path))
        criterion = lossf if lossf is not None else self.criterion
        loader = dataload if dataload is not None else self.data["valid"]
        total_loss = []
        for i, (x, y) in enumerate(loader):
            x = x.to(self.device)
            with autocast():
                pred = self.model(x)
            loss = criterion(pred.detach().cpu(), y[:, 0]).mean().item()
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        # # dict = metric(preds, trues, True)
        # b = {
        #     'test_result': {
        #         'time': time.time().strftime('%Y%m%d-%H%M'),
        #         'result': dict
        #     }
        # }
        # self.record_json(b)
        # print(dict)
        # self.model.train()
        return total_loss

    def seeparams(self):
        for name, parameters in self.model.named_parameters():
            print(name, ':', parameters.size())

    @torch.no_grad()
    def outhidden(self, modestr='test', hFName=None, modelp=None, paral=False):
        path = modelp if modelp is not None else self.modelpath
        if paral:
            self.load_paral(path)
        else:
            self.model.load_state_dict(torch.load(path))
        self.model.eval()
        print('>>>>>>>start output result : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(self.setting))
        dataloader = self.specifyload(modestr)
        if hFName is None:
            hFName = self.name
        # mkdir(f"{self.outputpath}{hFName}")
        hfname = f"{self.outputpath}{hFName}"
        train_steps = len(dataloader)
        criterion = self.criterion
        prin = printFeature()
        iter_count = 0
        time_now = time.time()
        std = 1.3902251
        mean = 3.3787999
        std = 1.04575
        mean = 2.63117
        preds = []
        trues = []
        min = torch.inf
        total = 0
        a = printFeature()
        for i, (x, y, t) in enumerate(dataloader):
            """
            batchsize, var, time, width, height
            """
            (x, y, t) = dataloader.dataset[282]
            x = x.float().to(self.device).unsqueeze(0)
            y = y.float().to(self.device).unsqueeze(0)
            t = t.to(self.device).unsqueeze(0)
            if self.halftrain:
                x = x.to(self.device)
                y = y[:, 0].to(self.device)
                with autocast():
                    pred = self.model(x)
            else:
                x = x.float().to(self.device)
                t = t[:, 0].long().to(self.device)
                pred, memo_info = self.model(x, t)
                pred = pred * std + mean
                pred = pred.unsqueeze(2)
                y = y[:, 0].unsqueeze(2).float()
                y = y * std + mean
                min = 0.1612
                max = 5.5402
                # min = 0.5601
                # max = 5.8808
                diffminr2 = -1.2
                diffmaxr2 = 1.3
                diffminr1 = -1.2
                diffmaxr1 = 1.1
                preds.append(pred.detach().cpu().numpy())
                trues.append(y.numpy())
                loss = self.criterion(pred, y.to(self.device))
                total = total + loss.item()
                x = x *std + mean

                # if i == 282:
                #     label = y[0,:,0]
                #     min = 0.5601
                #     max = 5.8808
                #     print(min, max)
                #     print(loss)

                # if i == 887:
                #     label = y[0,:,0]
                #     min = 0.3531
                #     max = 6.5312
                #     input = x[0, 0].detach().cpu()
                #     pred = pred[0, :, 0].detach().cpu()
                #     self.drawmore(pred, label.min(), label.max(), "region2pred.png")
                #     print(min, max)
                    # true = torch.cat([input, label], dim=1)
                    # print(loss)
                # if loss < min:
                #     min = loss
                #     print(i)
                #     print(loss)
            # indexip = f"{self.outputpath}{hFName}/index{i}"
            # mkdir(indexip)
            # for k in range(self.var_num):
            #     prin.out(x[0, k].detach().cpu(), f"{indexip}/input{k}.png",
            #              title=[f"{selectedVar[k]}-t{j}" for j in range(self.en_channels)]).close()
            # prin.out(y[0].detach().cpu(),
            #          f"{indexip}/true.png",
            #          title=[f"true-t{k}" for k in range(self.de_channels)]).close()
            # iter_count += 1
            # prin.out(pred[0].detach().cpu(),
            #          f"{indexip}/pred.png",
            #          title=[f"pred-t{k}" for k in range(self.de_channels)]).close()
            # prin.out(torch.cat([y[0, -1].unsqueeze(0), pred[0, -1].unsqueeze(0), x[0, 0, -1].unsqueeze(0)],
            #                    dim=0).detach().cpu(),
            #          f"{indexip}/contrast.png", title=["true", "pred", "input"]).close()
            # loss = criterion(pred, y)
            # print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
            # speed = (time.time() - time_now) / iter_count
            # left_time = speed * (train_steps - i)
            # print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
            # time_now = time.time()
            # iter_count = 0
        total = total / train_steps
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        mse = np.mean((preds - trues)**2, axis=(0,2,3,4))
        mae = np.mean(np.abs(preds - trues), axis=(0,2,3,4))
        print(mse, mae, np.sqrt(mse))
        print(mse.mean(),mae.mean())
        print(R2(torch.tensor(preds), torch.tensor(trues)))
        print(r2_score(torch.tensor(preds).reshape(preds.shape[0], -1), torch.tensor(trues).reshape(preds.shape[0], -1)))

    def draw(self, x, min_val, max_val, title = None):
        fig = plt.figure(figsize=(10, 8), dpi=300, facecolor='none')
        m = Basemap(llcrnrlon=52.6, llcrnrlat=-61, urcrnrlon=103.8, urcrnrlat=-9.8, projection='cyl', resolution='l')

        # 创建一个网格来表示数据的经纬度范围
        lon = np.linspace(52.6, 103.8, 256)
        lat = np.linspace(-61, -9.8, 256)
        lon, lat = np.meshgrid(lon, lat)

        # 输入数据
        data = x

        # 绘制地图
        m.drawcoastlines()
        m.drawcountries()

        # 设置经纬度标签的格式
        m.drawmeridians(np.arange(52.6, 103.8, 10), labels=[1, 0, 0, 1], fontsize=8, dashes=[1, 1])
        m.drawparallels(np.arange(-61, -9.8, 10), labels=[1, 0, 0, 1], fontsize=8, dashes=[1, 1])
        cmap = plt.get_cmap('viridis')
        levels = np.linspace(min_val, max_val, 13)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        im = m.imshow(data, cmap=cmap, origin='upper', extent=[lon.min(), lon.max(), lat.min(), lat.max()], norm=norm)
        cax = fig.add_axes([0, 0, 0, 0])
        cbar = plt.colorbar(im, cax=cax, cmap='Set3', ticks=np.linspace(min_val, max_val, 13))
        cbar.ax.axis('off')
        # plt.colorbar(im, cmap='Set3',ticks=np.linspace(min, max,13))
        # 去掉色卡
        # plt.colorbar(im).remove()
        if title is not None:
            print(title)
            plt.savefig(title, transparent=True, bbox_inches=None, dpi=300)
        # 显示图像
        plt.show()

    def drawdiff(self, inputs, title = None, color = "viridis", vmin = None, vmax = None):
        fig, axes = plt.subplots(1, inputs.shape[0], figsize=(10, 3), dpi=300)
        cmap = plt.get_cmap(color)
        norm = Normalize(vmin=vmin, vmax=vmax)
        for i, ax in enumerate(axes):
            # ax.imshow(inputs[i], cmap=color, vmin=min_val, vmax=max_val)
            ax.imshow(inputs[i], cmap=cmap)
            ax.axis('off')

        plt.subplots_adjust(wspace=0.01)  # 调整子图之间的间距
        if title is not None:
            plt.savefig(title, transparent=True, bbox_inches='tight', dpi=300)
        plt.show()

    def drawmore(self, inputs, min_val, max_val, title = None, color = "viridis"):
        fig, axes = plt.subplots(1, inputs.shape[0], figsize=(10, 3), dpi=300)
        levels = np.linspace(min_val, max_val, 13)
        cmap = plt.get_cmap(color)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
        for i, ax in enumerate(axes):
            # ax.imshow(inputs[i], cmap=color, vmin=min_val, vmax=max_val)
            ax.imshow(inputs[i], cmap=cmap, norm=norm)
            ax.axis('off')

        plt.subplots_adjust(wspace=0.01)  # 调整子图之间的间距
        if title is not None:
            plt.savefig(title, transparent=True, bbox_inches='tight', dpi=300)
        plt.show()

    def readloss(self, epoch=None):
        if epoch is None:
            epoch = self.epochs
        vali = []
        train = []
        for i in range(epoch):
            mse = self.read_one_json('epoch' + str(i + 1) + "_mseloss")
            vali.append(float(mse["vali"]))
            train.append(float(mse["train"]))
        return train, vali

    def load_paral(self, path):
        state_dict = torch.load(path, "cpu")
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():  # k为module.xxx.weight, v为权重
            name = k[7:]  # 截取`module.`后面的xxx.weight
            new_state_dict[name] = v
        # load params
        self.model.load_state_dict(new_state_dict)


def torchMSE(pred, true):
    a = pred.float() - true.float()
    return torch.mean(a ** 2)


epoch = 30
selector = []
varnum = 1
nc_NAN = -32768
models = []
lrs = [0.0003]
bss = []
eps = []
selectedVar = ["VHM0", "VHM0_SW1", "VHM0_WW", "VTM01_WW", "VHM0_SW2", "VTM02", "VSDY"]
conv = trainframe(batch_size=16, epochs=epoch, name="memounet", en_channels=8, de_channels=8,
                  device=1, period=epoch, lr=0.0003, criterion=torchMSE, patiences=epoch, var_num=7,
                  selector=None, embedtime=True, itrlossprint=True, model=MemoUnet,
                  halftrain=False, accumulate_step=1, percept=False, iterprintnum=3)
devices = [4,5,6,7]
devicestr = ""
for device in devices:
    devicestr = devicestr + f"{device},"
os.environ['CUDA_VISIBLE_DEVICES'] = devicestr[:-1]
temp = []
for i in range(len(devices)):
    temp.append(i)
devices = temp
conv.modelgrouptrain([MemoUnet,MemoUnet], inparal=True, device_ids=devices,
                     itrlossprint=True, main_device=devices[0], train=True)
# conv.outhidden(modelp="/kulang/fangteng/weight/nc/first/409/besta31_bestbest1_in8_out8_b16_ep22_v7_lr0.0003_s1.pth", paral=True)
