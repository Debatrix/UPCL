import os
import sys
import json
import logging
import numpy as np
import copy
import torch
from utils import factory
# from utils.confusion_matrix import plot_confusion_matrix
from utils.data_manager import DataManager
from utils.toolkit import count_parameters


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    if args['debug']:
        logs_dir = 'logs/_debug'
        log_file_path = 'logs/_debug/debug'
    else:
        logs_dir = "logs/{}/{}/{}_{}".format(args["model_name"],
                                             args["dataset"], init_cls,
                                             args['increment'])
        log_file_path = os.path.join(
            logs_dir, "{}_{}_{}".format(
                args["prefix"],
                args["seed"],
                args["convnet_type"],
            ))

    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=log_file_path + '.log'),
            logging.StreamHandler(sys.stdout),
        ],
    )

    try:
        with open(log_file_path + '.json', 'a') as f:
            f.write(json.dumps(args) + '\n')

        _set_random()
        _set_device(args)
        print_args(args)
        data_manager = DataManager(args["dataset"],
                                   args["shuffle"],
                                   args["seed"],
                                   args["init_cls"],
                                   args["increment"],
                                   imb_factor=args.get('imb_factor', 0.01),
                                   lt=args.get('lt', False),
                                   debug=args["debug"])
        model = factory.get_model(args["model_name"], args)

        cnn_curve, nme_curve = {
            "top1": [],
            "top5": []
        }, {
            "top1": [],
            "top5": []
        }
        for task in range(data_manager.nb_tasks):
            all_params = count_parameters(model._network)
            trainable_params = count_parameters(model._network, True)
            logging.info("All params: {}".format(all_params))
            logging.info("Trainable params: {}".format(trainable_params))

            model.incremental_train(data_manager)
            cnn_accy, nme_accy = model.eval_task(True)
            model.after_task()

            result = {
                'task': task,
                'all_params': all_params,
                'trainable_params': trainable_params
            }

            if nme_accy is not None:
                logging.info("NME: {}".format(nme_accy["grouped"]))
                result['grouped_nme'] = nme_accy["grouped"]

                nme_curve["top1"].append(nme_accy["top1"])
                nme_curve["top5"].append(nme_accy["top5"])

                logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
                logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))
                result['nme_top1_curve'] = nme_curve["top1"]
                result['nme_top5_curve'] = nme_curve["top5"]

                nme_avg_acc = sum(nme_curve["top1"]) / len(nme_curve["top1"])
                logging.info("Average Accuracy (NME): {}".format(nme_avg_acc))
                result['nme_avg_acc'] = nme_avg_acc
            else:
                logging.info("No NME accuracy.")

            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            result['grouped_cnn'] = cnn_accy["grouped"]

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))
            result['cnn_top1_curve'] = cnn_curve["top1"]
            result['cnn_top5_curve'] = cnn_curve["top5"]

            cnn_avg_acc = sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
            logging.info("Average Accuracy (CNN): {}".format(cnn_avg_acc))
            result['cnn_avg_acc'] = cnn_avg_acc

            with open(log_file_path + '.json', 'a') as f:
                f.write(json.dumps(result) + '\n')

            # if args.get('save_confusion_matrix', True):
            #     plot_confusion_matrix(cnn_accy["cm"], log_file_path + '.pdf',
            #                           args["increment"])
            if args.get('save_test', False):
                torch.save(cnn_accy, f'{log_file_path}_{task}.pth')

    except Exception as e:
        logging.exception(e)


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
