import copy
import logging
from tqdm import tqdm

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

from models.base import BaseLearner
from models.icarl import iCaRL
from utils.inc_net import IncrementalNet, SimpleCosineIncrementalNet
from utils.toolkit import tensor2numpy, check_loss, loss_counter

EPSILON = 1e-8


def KD_loss(student, teacher, T=4):
    return F.kl_div(F.log_softmax(student / T, dim=1),
                    F.softmax(teacher / T, dim=1),
                    reduction='batchmean') * T * T


class UFP(nn.Module):

    def __init__(
        self,
        in_dim,
        nb_classes,
        global_class_match=False,
    ):
        super().__init__()

        self.in_dim = 0
        self.nb_old_classes = 0
        self.nb_classes = 0
        self.old_in_dim = 0

        self.register_buffer('weight', torch.zeros((0)))
        self.register_buffer('ort_vectors', torch.zeros((0)))
        self.register_buffer('class_centroids', torch.zeros((0)))
        self.global_class_match = global_class_match
        self.centroids_init_flag = torch.zeros((0))

        self.update(nb_classes, in_dim)

    @staticmethod
    def gram_schmidt(n_vectors, ort_vectors):
        '''Gram-Schmidt Orthogonization'''

        n_dim = ort_vectors.shape[1]
        device = ort_vectors.device

        if n_vectors > n_dim:
            logging.warning(
                'Number of vectors should be smaller or equal to the dimension'
            )

        if ort_vectors.shape[0] == 0:
            ort_vectors = F.normalize(torch.randn(1, n_dim, device=device),
                                      p=2,
                                      dim=1)

        for _ in range(ort_vectors.shape[0], n_vectors):
            vector = F.normalize(torch.randn(n_dim, device=device), p=2, dim=0)
            PV_vector = torch.mv(ort_vectors.T, torch.mv(ort_vectors, vector))
            vector = F.normalize((vector - PV_vector).unsqueeze(0), p=2, dim=1)
            ort_vectors = torch.cat((ort_vectors, vector))

        return ort_vectors

    def channel_expand(self, new_dim):
        self.old_in_dim = self.in_dim
        if new_dim > self.in_dim:
            self.ort_vectors = torch.cat([
                self.ort_vectors,
                torch.zeros((self.nb_classes, new_dim - self.in_dim)).to(
                    self.ort_vectors.device)
            ], 1)
            self.class_centroids = torch.cat([
                self.class_centroids,
                torch.zeros((self.nb_classes, new_dim - self.in_dim)).to(
                    self.class_centroids.device)
            ], 1)
            self.in_dim = new_dim
        elif new_dim < self.in_dim:
            raise NotImplemented

    def classes_expand(self, nb_classes):
        if nb_classes > self.nb_classes:
            self.nb_old_classes = self.nb_classes
            self.nb_classes = nb_classes
            nb_new_class = self.nb_classes - self.nb_old_classes

            self.ort_vectors = self.gram_schmidt(self.nb_classes,
                                                 self.ort_vectors)
            self.class_centroids = torch.cat([
                self.class_centroids, self.ort_vectors[-nb_new_class:].clone()
            ], 0)
            self.centroids_init_flag = torch.cat([
                self.centroids_init_flag,
                torch.zeros((nb_new_class)).to(self.centroids_init_flag.device)
            ], 0)
        elif nb_classes < self.nb_classes:
            raise NotImplemented

    def update(self, nb_classes, in_dim=0):
        # ort_vectors and class_centroids
        self.channel_expand(in_dim)
        self.classes_expand(nb_classes)
        self.class_centroids = self.class_centroids.detach()

        self.weight = self.ort_vectors.detach()

        logging.info(f'Update UFP:{self.weight.shape}')

    def reset_parameters(self):
        self.ort_vectors = self.gram_schmidt(
            self.nb_classes,
            torch.zeros((0, self.in_dim),
                        device=self.ort_vectors.device)).detach()
        self.weight = self.ort_vectors.detach()
        self.class_centroids = self.ort_vectors.clone().detach()

    @torch.no_grad()
    def update_class_centroids(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        nb_per_labels = labels.bincount(minlength=self.nb_classes).reshape(
            -1, 1)
        centroids = torch.zeros_like(self.class_centroids, requires_grad=False)
        for idx in range(self.nb_classes):
            centroids[idx] = features[labels == idx].mean(0)
        centroids = F.normalize(centroids, p=2, dim=1)

        for idx in range(self.nb_classes):
            if nb_per_labels[idx] != 0:
                if self.centroids_init_flag[idx] < 0.5:
                    self.centroids_init_flag[idx] = 1
                    self.class_centroids[idx] = centroids[idx].clone().detach()
                else:
                    self.class_centroids[idx] = 0.9 * self.class_centroids[
                        idx] + 0.1 * centroids[idx].clone().detach()
        self.class_centroids = F.normalize(self.class_centroids, p=2, dim=1)
        return self.class_centroids

    @torch.no_grad()
    def new_class_match(self):
        start = 0 if self.global_class_match else self.nb_old_classes
        if self.nb_old_classes < self.nb_classes:
            new_weight = self.weight[start:].clone()
            centroid_target_dist = torch.mm(
                self.class_centroids[start:],
                new_weight.T).detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(-centroid_target_dist)
            for l, idx in zip(row_ind, col_ind):
                self.weight[start + l] = new_weight[idx]
        return self.weight

    def forward(self, input, labels=None):
        if labels is not None:
            self.update_class_centroids(input.clone().detach(), labels)
            self.weight = self.new_class_match()

        weight = self.weight

        out = F.linear(input, weight)

        return {
            'logits': out,
            'features': input,
            'prototypes': weight,
            'centroids': self.class_centroids
        }


class UPCL(BaseLearner):

    def __init__(self, args):
        self.t = args.get('temperature', 0.07)
        self.kd_t = args.get('kd_temperature', 4)
        self.kd_loss_type = args.get('kd_loss_type', 'rkd')
        self.kd_lambda = args.get('kd_lambda', 0.5)
        self.use_norm_feature = args.get('use_norm_feature', True)
        self.dyn_margin_tro = args.get('dyn_margin_tro', 1.0)
        self.online_match = args.get('online_match', True)

    def setup_network(self):
        self._network.train()

    def _get_prior(self, train_loader):
        prior = torch.zeros(self._total_classes, device=self._device)
        if self.dyn_margin_tro > 0:
            tro = self.dyn_margin_tro
            prior = torch.zeros(self._total_classes)
            for _, _, targets in train_loader:
                for idx in targets:
                    prior[idx] += 1
            prior = prior / prior.sum()
            prior = torch.log(prior**(tro * self.t) + 1e-8).to(self._device)
        elif self.dyn_margin_tro < 0:
            prior = torch.ones(self._total_classes, device=self._device)
            prior[:self._known_classes] = self.dyn_margin_tro
            prior[self._known_classes:] = -0.1
        return prior

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)
            self._old_network.eval()

        # hyper-parameters
        if self._cur_task == 0:
            lr = self.args['init_lr']
            weight_decay = self.args['init_weight_decay']
            milestones = self.args['init_milestones']
            gamma = self.args['init_lr_decay']
            epochs_num = self.args['init_epoch']
        else:
            lr = self.args['lrate']
            weight_decay = self.args['weight_decay']
            milestones = self.args['milestones']
            gamma = self.args['lrate_decay']
            epochs_num = self.args['epochs']

        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self._network.parameters()),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=milestones,
            gamma=gamma,
        )

        log_prior = self._get_prior(train_loader)
        logging.info(log_prior.cpu())

        prog_bar = tqdm(range(epochs_num))
        best_test_acc, best_epoch, best_network = -100, -1, None
        for _, epoch in enumerate(prog_bar):
            self.setup_network()
            epoch_loss = loss_counter()
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                loss_dict = {}
                inputs, targets = inputs.to(self._device), targets.to(
                    self._device)

                output = self._network(inputs, targets)

                logits = output["logits"]
                features = output['features']
                backbones_features = output.get('backbones_features', [])

                # PCL Loss
                cls_log_prob = F.normalize(
                    ((logits + log_prior) / self.t).exp(), dim=1, p=1).log()
                cls_mask = F.one_hot(targets, num_classes=self._total_classes)
                fea_logit = (features @ features.T) / self.t
                fea_log_prob = F.normalize(fea_logit.exp(), dim=1, p=1).log()
                fea_mask = torch.eq(targets.unsqueeze(0), targets.unsqueeze(1))

                proto_loss = -torch.sum((cls_mask * cls_log_prob).sum(1) /
                                        cls_mask.sum(1)) / cls_mask.shape[0]
                feat_loss = -torch.sum((fea_mask * fea_log_prob).sum(1) /
                                       fea_mask.sum(1)) / fea_mask.shape[0]

                if epoch > 50 or not self.online_match:
                    loss_dict['proto_loss'] = proto_loss
                loss_dict['feat_loss'] = 0.5**self._cur_task * feat_loss

                # KD Loss
                if self._cur_task > 0 and self.kd_lambda != 0 and self._old_network is not None:
                    with torch.no_grad():
                        old_output = self._old_network(inputs)
                    old_features = old_output['features']
                    old_logits = old_output['logits']
                    if self.kd_loss_type == 'fkd':
                        kd_loss = torch.mean(1 - torch.sum(F.normalize(
                            features[:, :old_features.shape[1]], p=2, dim=1) *
                                                           old_features,
                                                           dim=1))
                    else:
                        kd_loss = KD_loss(
                            logits[:, :self._known_classes],
                            old_logits,
                            self.kd_t,
                        )

                    kd_lambda = (self._known_classes / self._total_classes)
                    for k, v in loss_dict.items():
                        loss_dict[k] = (1 - kd_lambda) * v
                    loss_dict['kd_loss'] = self.kd_lambda * kd_lambda * kd_loss

                # Total Loss
                check_loss(loss_dict)
                loss = sum(filter(lambda x: x is not None, loss_dict.values()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Metrics
                loss_dict['total_loss'] = loss
                epoch_loss.update(loss_dict)
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total,
                                  decimals=2)
            info = "Task {}, Epoch {}/{} => {}, Train_accy {:.2f}".format(
                self._cur_task, epoch + 1, epochs_num, str(epoch_loss),
                train_acc)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += ', Test_accy {:.2f}'.format(test_acc)
                logging.info(info)

                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_epoch = epoch
                    best_network = self._network.module.copy() if isinstance(
                        self._network,
                        nn.DataParallel) else self._network.copy()
            prog_bar.set_description(info)
        logging.info(info)
        if isinstance(self._network, nn.DataParallel):
            self._network.module = best_network
        else:
            self._network = best_network
        logging.info(f'Best test acc {best_test_acc}% at epoch {best_epoch}')


class UPCL_IncrementalNet(IncrementalNet):

    def __init__(self, args, pretrained, gradcam=False):
        super(UPCL_IncrementalNet, self).__init__(args,
                                                  pretrained,
                                                  gradcam=gradcam)
        self.task_sizes = []
        self.use_norm_feature = args.get('use_norm_feature', True)
        self.global_class_match = args.get('global_class_match', False)

    def forward(self, x, label=None):
        x = self.convnet(x)
        if self.use_norm_feature:
            features = F.normalize(x['features'], p=2, dim=1)
        else:
            features = x['features']
        out = self.fc(features, label)

        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations

        out['fmaps'] = x['fmaps']
        out['backbones_features'] = [x['features']]
        return out

    def update_fc(self, nb_classes):
        self.fc = self.generate_fc(self.feature_dim, nb_classes)

        new_task_size = nb_classes - sum(self.task_sizes)
        self.task_sizes.append(new_task_size)

    def generate_fc(self, in_dim, nb_classes):
        if self.fc is None:
            self.fc = UFP(
                in_dim,
                nb_classes,
                self.global_class_match,
            )
        else:
            self.fc.update(nb_classes, in_dim)
        return self.fc

    def weight_align(self, increment):
        pass


class UPCL_iCaRL(UPCL, iCaRL):

    def __init__(self, args):
        iCaRL.__init__(self, args)
        UPCL.__init__(self, args)
        self._network = UPCL_IncrementalNet(args, False)

    def after_task(self):
        if self._cur_task > 0:
            self._network.weight_align(self._total_classes -
                                       self._known_classes)
        super().after_task()


# ###########################################################################
# CosineBaseline
# ###########################################################################


class CosineBaseline(BaseLearner):

    def __init__(self, args):
        super().__init__(args)
        self._network = SimpleCosineIncrementalNet(args, False)
        self.adj_margin = args.get('adj_margin', False)
        self.adj_margin_tro = args.get('adj_margin_tro', 1.0)

    def after_task(self):
        self._old_network = self._network.copy().freeze()
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes,
                                                self._total_classes))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.args.get(
                                           'batch_size', 32),
                                       shuffle=True,
                                       num_workers=4)
        test_dataset = data_manager.get_dataset(np.arange(
            0, self._total_classes),
                                                source="test",
                                                mode="test")
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=self.args.get(
                                          'batch_size', 32),
                                      shuffle=False,
                                      num_workers=4)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def setup_network(self):
        self._network.train()

    def _get_prior(self, train_loader):
        prior = torch.zeros(self._total_classes, device=self._device)
        if self.adj_margin_tro > 0:
            tro = self.adj_margin_tro
            prior = torch.zeros(self._total_classes)
            for _, _, targets in train_loader:
                for idx in targets:
                    prior[idx] += 1
            prior = prior / prior.sum()
            prior = torch.log(prior**tro + 1e-8).to(self._device)
        elif self.adj_margin_tro < 0:
            prior = torch.ones(self._total_classes, device=self._device)
            prior[:self._known_classes] = self.adj_margin_tro
            prior[self._known_classes:] = -2.0
        return prior

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)
            self._old_network.eval()

        # hyper-parameters
        if self._cur_task == 0:
            lr = self.args['init_lr']
            weight_decay = self.args['init_weight_decay']
            milestones = self.args['init_milestones']
            gamma = self.args['init_lr_decay']
            epochs_num = self.args['init_epoch']
        else:
            lr = self.args['lrate']
            weight_decay = self.args['weight_decay']
            milestones = self.args['milestones']
            gamma = self.args['lrate_decay']
            epochs_num = self.args['epochs']

        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self._network.parameters()),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=milestones,
            gamma=gamma,
        )

        if self.adj_margin:
            log_prior = self._get_prior(train_loader)
            logging.info(log_prior.cpu())

        prog_bar = tqdm(range(epochs_num))
        best_test_acc, best_epoch, best_network = -100, -1, None
        for _, epoch in enumerate(prog_bar):
            self.setup_network()
            epoch_loss = loss_counter()
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                loss_dict = {}
                inputs, targets = inputs.to(self._device), targets.to(
                    self._device)

                output = self._network(inputs)

                logits = output["logits"]
                features = output['features']

                if self.adj_margin:
                    logits = logits + log_prior
                loss_dict['ce_loss'] = F.cross_entropy(logits, targets)

                # KD Loss
                if self._cur_task > 0 and self._old_network is not None:
                    with torch.no_grad():
                        old_output = self._old_network(inputs)
                    old_features = old_output['features']
                    kd_loss = torch.mean(1 - torch.sum(F.normalize(
                        features[:, :old_features.shape[1]], p=2, dim=1) *
                                                       old_features,
                                                       dim=1))

                    kd_lambda = (self._known_classes / self._total_classes)
                    for k, v in loss_dict.items():
                        loss_dict[k] = (1 - kd_lambda) * v
                    loss_dict['kd_loss'] = kd_lambda * kd_loss

                # Total Loss
                check_loss(loss_dict)
                loss = sum(filter(lambda x: x is not None, loss_dict.values()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Metrics
                loss_dict['total_loss'] = loss
                epoch_loss.update(loss_dict)
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total,
                                  decimals=2)
            info = "Task {}, Epoch {}/{} => {}, Train_accy {:.2f}".format(
                self._cur_task, epoch + 1, epochs_num, str(epoch_loss),
                train_acc)
            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info += ', Test_accy {:.2f}'.format(test_acc)
                logging.info(info)

                if test_acc > best_test_acc:
                    best_test_acc = test_acc
                    best_epoch = epoch
                    best_network = self._network.module.copy() if isinstance(
                        self._network,
                        nn.DataParallel) else self._network.copy()
            prog_bar.set_description(info)
        logging.info(info)
        if isinstance(self._network, nn.DataParallel):
            self._network.module = best_network
        else:
            self._network = best_network
        logging.info(f'Best test acc {best_test_acc}% at epoch {best_epoch}')
