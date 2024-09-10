import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
devicess = [0]
from collections import defaultdict
import time
import argparse
import numpy as np
from PIL import Image
import torch
from sklearn import metrics
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
import torch.distributed as dist
import math
import torchio
# from torchio.transforms import (
#     ZNormalization,
# )
# from anfis_pytorch.membership import make_gauss_mfs
from tqdm import tqdm
from torchvision import utils
from config import hparams as hp
from utils.metric import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import xlrd
from pytsk.gradient_descent.antecedent import AntecedentGMF, antecedent_init_center
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import torch.nn.functional as F
source_train_images_0_dir=hp.source_train_images_0_dir
source_train_images_1_dir=hp.source_train_images_1_dir
source_train_labels_0_dir=hp.source_train_labels_0_dir
source_train_labels_1_dir=hp.source_train_labels_1_dir
source_test_images_0_dir =hp.source_test_images_0_dir
source_test_images_1_dir =hp.source_test_images_1_dir
source_test_labels_0_dir =hp.source_test_labels_0_dir
source_test_labels_1_dir =hp.source_test_labels_1_dir
source_val_images_0_dir =hp.source_val_images_0_dir
source_val_images_1_dir =hp.source_val_images_1_dir
source_val_labels_0_dir =hp.source_val_labels_0_dir
source_val_labels_1_dir =hp.source_val_labels_1_dir
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
import pandas as pd
path =  '' #Expert knowledge
def specificity_metric(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)
def parse_training_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument('-o', '--output_dir', type=str, default=hp.output_dir, required=False,
                        help='Directory to save checkpoints')
    parser.add_argument('--latest-checkpoint-file', type=str, default=hp.latest_checkpoint_file,
                        help='Store the latest checkpoint in each epoch')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, default=hp.total_epochs, help='Number of total epochs to run')
    training.add_argument('--epochs-per-checkpoint', type=int, default=hp.epochs_per_checkpoint,
                          help='Number of epochs per checkpoint')
    training.add_argument('--batch', type=int, default=hp.batch_size, help='batch-size')
    parser.add_argument(
        '-k',
        "--ckpt",
        type=str,
        default=hp.ckpt,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--init-lr", type=float, default=hp.init_lr, help="learning rate")
    # TODO
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )

    training.add_argument('--amp-run', action='store_true', help='Enable AMP')
    training.add_argument('--cudnn-enabled', default=True, help='Enable cudnn')
    training.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true',
                          help='disable uniform initialization of batchnorm layer weight')

    return parser

from sklearn.utils import resample
def calculate_ci(metric_func, y_true, y_pred, y_prob=None, n_bootstrap=1000, alpha=0.05):
    metrics = []
    for _ in range(n_bootstrap):
        # 进行有放回的抽样
        indices = resample(range(len(y_true)))
        if y_prob is not None:
            metric = metric_func(y_true[indices], y_prob[indices])
        else:
            metric = metric_func(y_true[indices], y_pred[indices])
        metrics.append(metric)

    # 计算置信区间
    lower = np.percentile(metrics, 100 * alpha / 2)
    upper = np.percentile(metrics, 100 * (1 - alpha / 2))
    return lower, upper
def data_excel(path):
    book = xlrd.open_workbook(path)
    #找到sheet页
    table = book.sheet_by_name("Sheet1")
    #获取总行数总列数
    row_Num = table.nrows
    col_Num = table.ncols
    key =table.col_values(0)[1:]# 这是第一行数据，作为字典的key值
    d = {}
    j = 1
    for i in range(row_Num-1):
        values = table.row_values(j)
        d[key[i]]=values[1:22]
        j += 1
    return d

def train():
    parser = argparse.ArgumentParser(description='Deep Fuzzy Network Training')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    from dataset import MedData_train
    os.makedirs(args.output_dir, exist_ok=True)

    if hp.mode == '2d':
        train_list = ''
        ddate = np.loadtxt(train_list)
        n_class = 21 # The number of expert knowledge
        x_train = ddate[0:, 0:n_class]
        n_rule = 5  # The number of fuzzy rules
        init_center = antecedent_init_center(x_train, n_rule=n_rule, n_init=50)
        gmf = nn.Sequential(
            AntecedentGMF(in_dim=n_class, n_rule=n_rule, high_dim=True, init_center=init_center),
            nn.LayerNorm(n_rule),
            nn.Dropout(p=0.3),
            nn.ReLU()
        )
        from models import MODEL
        model = MODEL(in_channels=1, n_class=n_class ,n_rule=n_rule, gmf=gmf)



    model = torch.nn.DataParallel(model, device_ids=devicess)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr, weight_decay=1e-2)

    # scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=20, verbose=True)
    scheduler = StepLR(optimizer, step_size=hp.scheduer_step_size, gamma=hp.scheduer_gamma)
    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-6)

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        print(os.path.join(args.output_dir, args.latest_checkpoint_file))
        ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file),
                          map_location=lambda storage, loc: storage)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])

        for state in optimizer.state.values():

            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        elapsed_epochs = ckpt["epoch"]
    else:
        elapsed_epochs = 0

    model.cuda()

    from loss_function import Classification_Loss, DiceLoss
    bce_loss = torch.nn.BCELoss()
    mse_loss = torch.nn.MSELoss().cuda()

    read_data = data_excel(path)
    train_dataset = MedData_train(source_train_images_0_dir, source_train_labels_0_dir, source_train_images_1_dir, source_train_labels_1_dir,read_data)
    train_loader = DataLoader(train_dataset.training_set,
                              batch_size=args.batch,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)

    model.train()

    epochs = args.epochs - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)
    for epoch in range(1, epochs + 1):
        print("epoch:" + str(epoch))
        epoch += elapsed_epochs
        num_iters = 0

        for i, batch in enumerate(train_loader):

            print(f"Batch: {i}/{len(train_loader)} epoch {epoch}")
            optimizer.zero_grad()

            x = batch['source']['data']
            seg = batch['label']['data']
            cls = batch['cls']
            reg = batch['reg']
            x = x.type(torch.FloatTensor).cuda()
            cls = cls.type(torch.FloatTensor).cuda()
            reg = reg.type(torch.FloatTensor).cuda()
            seg = seg.type(torch.FloatTensor).cuda()

            x = x.squeeze(-1)
            seg = seg.squeeze(-1)

            seg[seg != 0] = 1
            outputs, p_seg, p_reg = model(x)
            p_seg = torch.softmax(p_seg,dim=1)
            cls = cls.unsqueeze(1)

            print(outputs,cls)
            loss = bce_loss(outputs, cls)
            seg_loss = dice_loss(p_seg[:,1,:,:], seg)

            # x_log = F.log_softmax(p_reg, dim=1)
            # y = F.softmax(reg, dim=1)
            #
            # reg_loss_kl = kl_loss(x_log, y)
            reg_loss_mse = mse_loss(p_reg, reg)

            total_loss = loss + seg_loss +reg_loss_mse
            num_iters += 1
            total_loss.backward()

            optimizer.step()
            iteration += 1



            print("分割loss:" + str(seg_loss.item()), "MSE-loss:" + str(reg_loss_mse.item()),"Dice:" + str(dice),"bce-loss:" + str(loss.item()) )



        scheduler.step()

        # Store latest checkpoint in each epoch
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,

            },
            os.path.join(args.output_dir, args.latest_checkpoint_file),
        )

        # Save checkpoint
        if epoch % args.epochs_per_checkpoint == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(args.output_dir, f"checkpoint_{epoch:04d}.pt"),
            )


def test():
    parser = argparse.ArgumentParser(description='Deep Fuzzy Network Testing')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    from dataset import MedData_test

    if hp.mode == '2d':

        test_list = ''
        ddate = np.loadtxt(test_list)
        n_class = 21

        x_train = ddate[0:, 0:n_class]
        n_rule = 5
        init_center = antecedent_init_center(x_train, n_rule=n_rule, n_init=50)
        gmf = nn.Sequential(
            AntecedentGMF(in_dim=n_class, n_rule=n_rule, high_dim=True, init_center=init_center),
            nn.LayerNorm(n_rule),
            nn.Dropout(p=0.3),
            nn.ReLU()
        )

        from models import MODEL
        model = MODEL(in_channels=1, n_class=n_class ,n_rule=n_rule, gmf=gmf)


    model = torch.nn.DataParallel(model, device_ids=devicess, output_device=[1])

    print("load model:", args.ckpt)
    print(os.path.join(args.output_dir, args.latest_checkpoint_file))
    ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file),
                      map_location=lambda storage, loc: storage)

    model.load_state_dict(ckpt["model"])

    model.cuda()
    data_read = data_excel(path)
    test_dataset = MedData_test(source_test_images_0_dir, source_test_labels_0_dir, source_test_images_1_dir, source_test_labels_1_dir, data_read)
    test_loader = DataLoader(test_dataset.test_set,
                             batch_size=args.batch,
                             shuffle=True,
                             pin_memory=True,
                             drop_last=True)

    model.eval()
    predicts = []
    gts = []
    prob = []
    for i, batch in enumerate(test_loader):

        x = batch['source']['data']
        seg = batch['label']['data']
        cls = batch['cls']

        x = x.type(torch.FloatTensor).cuda()
        seg = seg.type(torch.FloatTensor).cuda()
        cls = cls.type(torch.LongTensor).cuda()
        x = x.squeeze(-1)
        seg = seg.squeeze(-1)
        with torch.no_grad():
            outputs, p_seg, p_reg = model(x)

            outputs_logit = np.where(outputs.cpu().detach().numpy() > 0.5, 1, 0)

            predicts.append(outputs_logit)
            gts.append(cls.cpu().detach().numpy())
            prob.append(outputs.cpu().detach().numpy())

    predicts = np.concatenate(predicts).flatten().astype(np.int16)
    gts = np.concatenate(gts).flatten().astype(np.int16)
    prob = np.concatenate(prob).flatten()
    pre_cancer = np.asarray(prob)
    true_label_list = np.asarray(gts)
    fpr, tpr, thre = metrics.roc_curve(true_label_list, pre_cancer)
    auc = metrics.auc(fpr, tpr)
    acc = metrics.accuracy_score(gts, predicts)
    precision = metrics.precision_score(gts,predicts)
    confusion_matrix = metrics.confusion_matrix(gts,predicts)
    sensitivity = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0])

    specificity = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])

    auc_ci = calculate_ci(roc_auc_score, gts, y_pred=None, y_prob=prob)
    acc_ci = calculate_ci(accuracy_score, gts, predicts)
    precision_ci = calculate_ci(precision_score, gts, predicts)
    sensitivity_ci = calculate_ci(recall_score, gts, predicts)
    specificity_ci = calculate_ci(specificity_metric, gts, predicts)

    ## log
    print(f"AUC: {auc:.3f}, 95% CI: ({auc_ci[0]:.3f}, {auc_ci[1]:.3f})")
    print(f"Accuracy: {acc:.3f}, 95% CI: ({acc_ci[0]:.3f}, {acc_ci[1]:.3f})")
    print(f"Precision: {precision:.3f}, 95% CI: ({precision_ci[0]:.3f}, {precision_ci[1]:.3f})")
    print(f"Sensitivity: {sensitivity:.3f}, 95% CI: ({sensitivity_ci[0]:.3f}, {sensitivity_ci[1]:.3f})")
    print(f"Specificity: {specificity:.3f}, 95% CI: ({specificity_ci[0]:.3f}, {specificity_ci[1]:.3f})")
    print('TN,FP,FN,TP',confusion_matrix[0, 0],confusion_matrix[0, 1],confusion_matrix[1, 0],confusion_matrix[1, 1])

if __name__ == '__main__':

    if hp.train_or_test == 'train':
        train()
    elif hp.train_or_test == 'test':
        test()
