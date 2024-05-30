import logging
from tqdm import tqdm

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from models.icarl import iCaRL
from utils.toolkit import tensor2numpy, check_loss, loss_counter
from utils.inc_net import IncrementalNet
from torch.utils.data import DataLoader


class NCP(nn.Module):
    def __init__(self,
                 in_dim,
                 nb_classes,
                 max_nb_classes=100,
                 with_etfhead=True):
        super().__init__()
        self.in_dim = in_dim
        self.nb_old_classes = 0
        self.nb_classes = 0
        self.max_nb_classes = max_nb_classes

        self.register_buffer('weight', torch.zeros((0, in_dim)))
        self.register_buffer('class_centroids', torch.zeros((0, in_dim)))

        # NC
        orth_vec = self.gram_schmidt(self.in_dim, self.max_nb_classes)
        if with_etfhead:
            i_nc_nc = torch.eye(self.max_nb_classes)
            one_nc_nc = torch.mul(
                torch.ones(self.max_nb_classes, self.max_nb_classes),
                (1 / self.max_nb_classes))
            etf_vec = torch.mul(
                torch.matmul(i_nc_nc - one_nc_nc, orth_vec),
                torch.sqrt(
                    torch.tensor(self.max_nb_classes /
                                 (self.max_nb_classes - 1))))
        else:
            etf_vec = orth_vec
        self.register_buffer('etf_vec', etf_vec)

        self.update(nb_classes)

    def update(self, nb_classes):
        if nb_classes > self.nb_classes:
            self.nb_old_classes = self.nb_classes
            self.nb_classes = nb_classes
            self.weight = self.etf_vec[:self.nb_classes]
            self.class_centroids = torch.cat([
                self.class_centroids,
                self.etf_vec[self.nb_old_classes:self.nb_classes]
            ], 0)
        elif nb_classes < self.nb_classes:
            raise NotImplemented

    @staticmethod
    def gram_schmidt(n_dim, n_vectors):
        '''Gram-Schmidt Orthogonization'''

        if n_vectors > n_dim:
            logging.warning(
                'Number of vectors should be smaller or equal to the dimension'
            )

        ort_vectors = F.normalize(torch.randn(1, n_dim), p=2, dim=1)

        for _ in range(ort_vectors.shape[0], n_vectors):
            vector = F.normalize(torch.randn(n_dim), p=2, dim=0)
            PV_vector = torch.mv(ort_vectors.T, torch.mv(ort_vectors, vector))
            vector = F.normalize((vector - PV_vector).unsqueeze(0), p=2, dim=1)
            ort_vectors = torch.cat((ort_vectors, vector))

        return ort_vectors

    def forward(self, input, labels=None, epoch_rate=1.0):
        weight = self.weight.clone().detach()
        if labels is not None:
            weight[self.nb_old_classes:] = epoch_rate * self.weight[
                self.nb_old_classes:] + (
                    1 - epoch_rate) * self.class_centroids[
                        self.nb_old_classes:].clone().detach()
            weight = F.normalize(weight, p=2, dim=1)
        out = F.linear(input, weight)
        return {
            'logits': out,
            'weight': weight,
        }


class NC_IncrementalNet(IncrementalNet):
    def __init__(self, args, pretrained, gradcam=False):
        super(NC_IncrementalNet, self).__init__(args,
                                                pretrained,
                                                gradcam=gradcam)
        self.max_nb_classes = args.get('max_nb_classes', 100)
        self.with_etfhead = args.get('with_etfhead', True)

    def forward(self, x, label=None, epoch_rate=1.0):
        x = self.convnet(x)
        features = F.normalize(x['features'], p=2, dim=1)
        out = self.fc(features, label, epoch_rate)

        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations

        out['features'] = features
        out['prototypes'] = self.fc.weight.detach()
        out['fmaps'] = x['fmaps']
        return out

    def update_fc(self, nb_classes):
        self.fc = self.generate_fc(self.feature_dim, nb_classes)

    def generate_fc(self, in_dim, nb_classes):
        if self.fc is None:
            self.fc = NCP(in_dim, nb_classes, self.max_nb_classes,
                          self.with_etfhead)
        else:
            self.fc.update(nb_classes)
        return self.fc


class NCCIL(iCaRL):
    def __init__(self, args):
        super().__init__(args)
        self._network = NC_IncrementalNet(args, False)
        self.kd_labmda = args.get('kd_labmda', 5)
        self.with_ftc = args.get('with_ftc', True)
        self.with_fhtc = args.get('with_fhtc', False)

    def _update_centroid(self, train_loader):
        logging.info('update class centroids')

        if isinstance(self._network, nn.DataParallel):
            feature_dim = self._network.module.feature_dim
        else:
            feature_dim = self._network.feature_dim

        centroids = torch.zeros((self._total_classes - self._known_classes),
                                feature_dim).to(self._device)
        vectors, targets = self._extract_vectors(train_loader)

        vectors = torch.from_numpy(vectors[targets > self._known_classes]).to(
            self._device)
        targets = torch.from_numpy(targets[targets > self._known_classes] -
                                   self._known_classes).to(self._device)
        for l in torch.unique(targets):
            centroids[l] = vectors[targets == l].mean(0)
        centroids = F.normalize(centroids, p=2, dim=1)
        if isinstance(self._network, nn.DataParallel):
            self._network.module.fc.class_centroids[
                self._known_classes:] = centroids
        else:
            self._network.fc.class_centroids[self._known_classes:] = centroids
        return centroids

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._old_network is not None:
            self._old_network.to(self._device)

        self._update_centroid(train_loader)

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

        prog_bar = tqdm(range(epochs_num))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            epoch_loss = loss_counter()
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                loss_dict = {}
                inputs, targets = inputs.to(self._device), targets.to(
                    self._device)

                if self.with_fhtc:
                    r = min(epoch / epochs_num * 2, 1.0)
                elif self.with_ftc:
                    r = epoch / epochs_num
                else:
                    r = 1.0

                output = self._network(inputs, targets, r)
                logits = output['logits']
                features = output['features']

                # align_loss
                dot = torch.sum(features * output['weight'][targets], dim=1)
                loss_dict['align_loss'] = 0.5 * torch.mean(
                    ((dot -
                      (torch.ones_like(dot) * torch.ones_like(dot)))**2) /
                    torch.ones_like(dot))

                # KD Loss
                if self._cur_task > 0 and self.kd_labmda > 0:
                    old_features = self._old_network(inputs)['features']
                    kd_dot = torch.sum(features * old_features, dim=1)
                    loss_dict[
                        'distill_loss'] = self.kd_labmda * 0.5 * torch.mean(
                            ((kd_dot - (torch.ones_like(kd_dot) *
                                        torch.ones_like(kd_dot)))**2) /
                            torch.ones_like(kd_dot))

                # Total Loss
                check_loss(loss_dict)
                loss = sum(filter(lambda x: x is not None, loss_dict.values()))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Metrics
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
            prog_bar.set_description(info)
        logging.info(info)
