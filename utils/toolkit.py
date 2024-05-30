import math
import os
import numpy as np
import torch
import json
from enum import Enum


class ConfigEncoder(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum':
                o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {'$function': o.__module__ + "." + o.__name__}
        return json.JSONEncoder.default(self, o)


def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def tensor2numpy(x):
    return x.cpu().data.numpy() if x.is_cuda else x.data.numpy()


def target2onehot(targets, n_classes):
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def accuracy(y_pred, y_true, nb_old, increment=10):
    assert len(y_pred) == len(y_true), "Data length error."
    all_acc = {}
    all_acc["total"] = np.around((y_pred == y_true).sum() * 100 / len(y_true),
                                 decimals=2)

    # Grouped accuracy
    for class_id in range(0, np.max(y_true), increment):
        idxes = np.where(
            np.logical_and(y_true >= class_id, y_true
                           < class_id + increment))[0]
        label = "{}-{}".format(
            str(class_id).rjust(2, "0"),
            str(class_id + increment - 1).rjust(2, "0"))
        all_acc[label] = np.around(
            (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes),
            decimals=2)

    # Old accuracy
    idxes = np.where(y_true < nb_old)[0]
    all_acc["old"] = (0 if len(idxes) == 0 else np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2))

    # New accuracy
    idxes = np.where(y_true >= nb_old)[0]
    all_acc["new"] = np.around(
        (y_pred[idxes] == y_true[idxes]).sum() * 100 / len(idxes), decimals=2)

    return all_acc


def split_images_labels(imgs):
    # split trainset.imgs in ImageFolder
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)


def save_fc(args, model):
    _path = os.path.join(args['logfilename'], "fc.pt")
    if len(args['device']) > 1:
        fc_weight = model._network.fc.weight.data
    else:
        fc_weight = model._network.fc.weight.data.cpu()
    torch.save(fc_weight, _path)

    _save_dir = os.path.join(f"./results/fc_weights/{args['prefix']}")
    os.makedirs(_save_dir, exist_ok=True)
    _save_path = os.path.join(_save_dir, f"{args['csv_name']}.csv")
    with open(_save_path, "a+") as f:
        f.write(f"{args['time_str']},{args['model_name']},{_path} \n")


def save_model(args, model):
    #used in PODNet
    _path = os.path.join(args['logfilename'], "model.pt")
    if len(args['device']) > 1:
        weight = model._network
    else:
        weight = model._network.cpu()
    torch.save(weight, _path)


def _check_loss(loss, name=''):
    if loss is None:
        return
    loss = loss.item() if isinstance(loss, torch.Tensor) else loss
    if not math.isfinite(loss):
        if name:
            r = 'Loss {} is {}, stopping training'.format(name, loss)
        else:
            r = 'Loss is {}, stopping training'.format(loss)
        raise Exception(r)


def check_loss(loss):
    if isinstance(loss, dict):
        for k, v in loss.items():
            _check_loss(v, k)
    elif isinstance(loss, (list, tuple)):
        for idx, v in loss:
            _check_loss(v, idx)
    else:
        _check_loss(v)


class loss_counter(object):

    def __init__(self):
        self.counter = None
        self.count = 0

    def update(self, input):
        self.count += 1
        if isinstance(input, dict):
            if self.counter is None:
                self.counter = {}
            for k, v in input.items():
                if k not in self.counter:
                    self.counter[k] = v.item()
                else:
                    self.counter[k] += v.item()
        elif isinstance(input, (list, tuple)):
            if self.counter is None:
                self.counter = [0 for _ in range(len(input))]
            for idx, v in enumerate(input):
                self.counter[idx] += v.item()

    def get_loss(self):
        if isinstance(self.counter, dict):
            return {k: v / self.count for k, v in self.counter.items()}
        elif isinstance(self.counter, list):
            return [v / self.count for v in self.counter]

    def __str__(self) -> str:
        ret = []
        if isinstance(self.counter, dict):
            for k, v in self.counter.items():
                ret.append('{} {:.2e}'.format(k, v))
        elif isinstance(self.counter, list):
            for v in self.counter:
                ret.append('{:.2f}'.format(v))
        return ', '.join(ret)

    def __len__(self):
        return len(self.counter) if self.counter is not None else 0

    def reset(self):
        self.__init__()
