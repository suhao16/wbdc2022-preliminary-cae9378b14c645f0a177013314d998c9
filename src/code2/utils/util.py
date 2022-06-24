import logging
import random

import numpy as np
from sklearn.metrics import f1_score, accuracy_score
import torch
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup

from dataset.category_id_map import lv2id_to_lv1id

from pathlib import Path
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


def setup_device(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.n_gpu = torch.cuda.device_count()


def setup_seed(args,i):
    random.seed(args.seed+i)
    np.random.seed(args.seed+i)
    torch.manual_seed(args.seed+i)


def setup_logging():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger


def build_optimizer(args, model):
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    return optimizer, scheduler


def evaluate(predictions, labels):
    # prediction and labels are all level-2 class ids

    lv1_preds = [lv2id_to_lv1id(lv2id) for lv2id in predictions]
    lv1_labels = [lv2id_to_lv1id(lv2id) for lv2id in labels]

    lv2_f1_micro = f1_score(labels, predictions, average='micro')
    lv2_f1_macro = f1_score(labels, predictions, average='macro')
    lv1_f1_micro = f1_score(lv1_labels, lv1_preds, average='micro')
    lv1_f1_macro = f1_score(lv1_labels, lv1_preds, average='macro')
    mean_f1 = (lv2_f1_macro + lv1_f1_macro + lv1_f1_micro + lv2_f1_micro) / 4.0

    eval_results = {'lv1_acc': accuracy_score(lv1_labels, lv1_preds),
                    'lv2_acc': accuracy_score(labels, predictions),
                    'lv1_f1_micro': lv1_f1_micro,
                    'lv1_f1_macro': lv1_f1_macro,
                    'lv2_f1_micro': lv2_f1_micro,
                    'lv2_f1_macro': lv2_f1_macro,
                    'mean_f1': mean_f1}

    return eval_results


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # increment file or directory path
    # i.e. save/v1 --> save/v{sep}2, save/v{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        suffix = path.suffix
        path = path.with_suffix('')
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # update path
    dir = path if path.suffix == '' else path.parent  # directory
    if not dir.exists() and mkdir:
        dir.mkdir(parents=True, exist_ok=True)  # make directory
    return path


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(list)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].append(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        meter_str = []
        for name, meter in self.meters.items():
            meter_str.append(
                "{}: {}".format(name, str(meter[-1]))
            )
        return self.delimiter.join(meter_str)

    def output_csv(self, path):
        df = pd.DataFrame(self.meters)
        df.to_csv(path)

    def plot(self, path, nums_crow=1):
        nums_params = len(self.meters)
        nums_line = nums_params // nums_crow + int(nums_params % nums_crow != 0)
        plt.figure(figsize=(16, 8))
        index = 1
        for key, value in self.meters.items():
            plt.subplot(nums_line, nums_crow, index)
            plt.plot(value)
            plt.title(key)
            index += 1
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
        plt.savefig(path)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score


class EMA:
    def __init__(self, model, decay, device=None):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.model.to(device=self.device)

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                if self.device is not None:
                    self.shadow[name].to(device=self.device)

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class FGM:
    def __init__(self, args, model):
        self.model = model
        self.backup = {}
        self.emb_name = args.emb_name
        self.epsilon = args.epsilon

    def attack(self):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


# def swa(model, model_dir, swa_start=1):
#     """
#     swa 滑动平均模型，一般在训练平稳阶段再使用 SWA
#     """
#     model_path_list = get_model_path_list(model_dir)
#
#     assert 1 <= swa_start < len(model_path_list) - 1, \
#         f'Using swa, swa start should smaller than {len(model_path_list) - 1} and bigger than 0'
#
#     swa_model = copy.deepcopy(model)
#     swa_n = 0.
#
#     with torch.no_grad():
#         for _ckpt in model_path_list[swa_start:]:
#             logger.info(f'Load model from {_ckpt}')
#             model.load_state_dict(torch.load(_ckpt, map_location=torch.device('cpu')))
#             tmp_para_dict = dict(model.named_parameters())
#
#             alpha = 1. / (swa_n + 1.)
#
#             for name, para in swa_model.named_parameters():
#                 para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))
#
#             swa_n += 1
#
#     # use 100000 to represent swa to avoid clash
#     swa_model_dir = os.path.join(model_dir, f'checkpoint-100000')
#     if not os.path.exists(swa_model_dir):
#         os.mkdir(swa_model_dir)
#
#     logger.info(f'Save swa model in: {swa_model_dir}')
#
#     swa_model_path = os.path.join(swa_model_dir, 'model.pt')
#
#     torch.save(swa_model.state_dict(), swa_model_path)
#
#     return swa_model


# if __name__ == '__main__':
#     save_path = increment_path('save/v')
#     print(save_path / 'a.txt')
