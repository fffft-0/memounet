import numpy as np
import torch
import matplotlib.pyplot as plt
import os,json
import logging as log
from matplotlib import colors
from math import cos, pi
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

def adjust_lr_exp(optimizer, epoch, learning_rate, type = 'type1', decay = 0.5):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if type=='type1':
        lr_adjust = {epoch: learning_rate * (decay ** ((epoch-1) // 1))}
    elif type=='type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))

class EarlyStopping:
    def __init__(self, patience=3, verbose=False, delta=0, best_score = None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = best_score
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score > self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true, is_dict = False):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    corr = CORR(pred, true)
    if is_dict:
        return {
            'mae': "{:.5f}".format(mae),
            'mse': "{:.5f}".format(mse),
            'rmse': "{:.5f}".format(rmse),
            'mape': "{:.5f}".format(mape),
            'mspe': "{:.5f}".format(mspe),
            'corr': "{:.5f}".format(corr)
        }
    return mae, mse, rmse, mape, mspe, corr

def countParas(model, name = '?'):
    params = list(model.parameters())
    num_params = 0
    for param in params:
        curr_num_params = 1
        for size_count in param.size():
            curr_num_params *= size_count
        num_params += curr_num_params
    print(f"{name} total number of parameters: " + str(num_params))

def show_nc_graph(x, y, data):
    [X, Y] = np.meshgrid(x, y)
    plt.contourf(X, Y, data)
    plt.colorbar()
    plt.show()

def init_log(path = 'log1.log'):
    logger = log.getLogger(name='r')  # 不加名称设置root logger
    logger.setLevel(log.DEBUG)
    formatter = log.Formatter(
        '%(asctime)s: - %(message)s',
        datefmt='%m%d %H%M')

    # 使用FileHandler输出到文件
    fh = log.FileHandler(path)
    fh.setLevel(log.DEBUG)
    fh.setFormatter(formatter)

    # 使用StreamHandler输出到屏幕
    ch = log.StreamHandler()
    ch.setLevel(log.DEBUG)
    ch.setFormatter(formatter)

    # 添加两个Handler
    logger.addHandler(ch)
    logger.addHandler(fh)

def adjust_lr_cos_warm(optimizer, current_epoch, max_epoch, lr_min=0, lr_max=0.1, warmup=True):
    """

    :param optimizer:
    :param current_epoch: >=1,int
    :param max_epoch: half period
    :param lr_min:
    :param lr_max:  warmup max lr
    :param warmup:  bool
    :return:
    """
    it = max_epoch*0.02
    it = int(it) if it > 10 else 10
    warmup_epoch = it if warmup else 0
    if current_epoch < warmup_epoch:
        lr = lr_max * current_epoch / warmup_epoch
    elif current_epoch < max_epoch:
        lr = lr_min + (lr_max - lr_min) * (
                1 + cos(pi * (current_epoch - warmup_epoch) / (max_epoch - warmup_epoch))) / 2
    else:
        lr = lr_min + (lr_max - lr_min) * (
                1 + cos(pi * (current_epoch - max_epoch) / (max_epoch))) / 2
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def writeJson(path, d):
    with open(path, "a+", encoding='utf-8') as f:
        f.seek(0, os.SEEK_SET)
        try:
            data = json.load(f)
        except:
            data = {}
    deepwrite(d, data)
    with open(path, "w", encoding='utf-8') as f:
        json.dump(data, f)

def deepwrite(new, old):
    for key in new.keys():
        if type(new[key]) == dict:
            if old.get(key) is None:
                old[key] = new[key]
            else:
                deepwrite(new[key], old[key])
        else:
            old[key] = new[key]


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print(f"---  create: {path}  ---")
    else:
        print(f"---  exist: {path}  ---")
    return path

class printFeature():
    def __init__(self, vmin = None, vmax = None):
        self.vmin = vmin
        self.vmax = vmax
        if vmin is not None:
            self.norm = colors.Normalize(vmin=vmin, vmax=vmax)
        else:
            self.norm = None

    def out(self, x, path = None, title = None, unify = True, num = 5, normtype = 0, normsize = None):
        image_width = x.shape[-1]/100
        image_height = x.shape[-2]/100
        while image_width < 1 or image_height < 1:
            image_width *= 2
            image_height *= 2
        if unify:
            if self.norm is None:
                self.vmin = torch.min(x).item()
                self.vmax = torch.max(x).item()
                if normtype == 0:
                    norm = colors.Normalize(vmin=self.vmin, vmax=self.vmax)
                elif normtype == 1:
                    norm = colors.BoundaryNorm(np.arange(self.vmin, self.vmax, (self.vmax - self.vmin) / normsize),
                                               ncolors=256)
            else:
                norm = self.norm
        l = len(x.shape)
        space = 0.6
        if l == 3:
            size = x.shape[0]
            plt.subplots_adjust(wspace=space)
        elif l == 2:
            size = 1
            x = x.reshape(1, x.shape[-2], -1)
        else:
            return
        # fig, ax = plt.subplots(1, size, figsize=(size*2.5+space*size + 1, 2.5))
        fig = plt.figure(figsize=(size*image_width+space*(size+1), image_height+1), dpi=100)
        gs = gridspec.GridSpec(2, size, width_ratios=[1]*size, height_ratios=[image_height, 0.2], hspace=0.2, wspace=0.6)  # 指定各个子图的宽比例。
        title = [f"index: {i + 1}" for i in range(size)] if title is None or len(title) != size else title
        for i in range(size):
            if not unify:
                vmin = torch.min(x[i]).item()
                vmax = torch.max(x[i]).item()
                if normtype == 0:
                    norm = colors.Normalize(vmin=vmin, vmax=vmax)
                elif normtype == 1:
                    norm = colors.BoundaryNorm(np.arange(vmin, vmax, (vmax - vmin) / normsize),
                                               ncolors=256)
            ax = plt.subplot(gs[0, i])
            ax.set_title(title[i])
            a = ax.imshow(x[i], norm=norm)
            if not unify:
                cax = plt.subplot(gs[1, i])
                cb1 = plt.colorbar(a, cax = cax, orientation='horizontal')
                tick_locator = ticker.MaxNLocator(nbins=num)  # colorbar上的刻度值个数
                cb1.locator = tick_locator
                ticks = [round(vmin, 2)]
                for i in range(num - 1):
                    ticks.append(round((vmax - vmin)*(i+1)/num + vmin, 2))
                cb1.locator = tick_locator
                cb1.set_ticks(ticks + [round(vmax, 2)])
                cb1.update_ticks()
        if unify:
            cax = plt.subplot(gs[1,:2])
            cb1 = plt.colorbar(a, cax = cax, orientation='horizontal')
            tick_locator = ticker.MaxNLocator(nbins=num)  # colorbar上的刻度值个数
            cb1.locator = tick_locator
            ticks = [round(self.vmin, 2)]
            for i in range(num - 1):
                ticks.append(round((self.vmax - self.vmin)*(i+1)/num + self.vmin, 2))
            cb1.locator = tick_locator
            cb1.set_ticks(ticks + [round(self.vmax, 2)])
            cb1.update_ticks()
        if path is not None:
            plt.savefig(path)
        return plt

    def change(self, vmin, vmax):
        self.vmin = vmin
        self.vmax = vmax
        self.norm = colors.Normalize(vmin=vmin, vmax=vmax)

def model_structure(model):
    blank = ' '
    print('-' * 90)
    print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
          + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
          + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 90)
    num_para = 0
    type_size = 1  # 如果是浮点数就是4

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        if len(key) <= 30:
            key = key + (30 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 40:
            shape = shape + (40 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank

        print('| {} | {} | {} |'.format(key, shape, str_num))
    print('-' * 90)
    print('The total number of parameters: ' + str(num_para))
    print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
    print('-' * 90)

