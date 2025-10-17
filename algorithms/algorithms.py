import copy

import torch
import torch.nn as nn
import numpy as np

from models.models import classifier, ReverseLayerF, Discriminator, RandomLayer, Discriminator_CDAN, \
    codats_classifier, Discriminator_fea, Adapter, Discriminator_t, classifier_T, classifier2, TransformerModel, ItranModel, CNNT
from models.loss import MMD_loss, CORAL, ConditionalEntropyLoss, VAT, LMMD_loss, HoMM_loss
from utils import EMA

from torch.autograd import Variable
import torch.distributions as dist


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()
        self.MSELoss = nn.MSELoss(reduction='sum')

    def update(self, *args, **kwargs):
        raise NotImplementedError


class Lower_Upper_bounds(Algorithm):
    """
    Lower bound: train on source and test on target.
    Upper bound: train on target and test on target.
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(Lower_Upper_bounds, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, src_x, src_y):
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        src_cls_loss = self.cross_entropy(src_pred, src_y)

        loss = src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Src_cls_loss': src_cls_loss.item()}


class MMDA(Algorithm):
    """
    MMDA: https://arxiv.org/abs/1901.00282
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(MMDA, self).__init__(configs)

        self.mmd = MMD_loss()
        self.coral = CORAL()
        self.cond_ent = ConditionalEntropyLoss()

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams

    def update(self, src_x, src_y, trg_x):
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        src_cls_loss = self.cross_entropy(src_pred, src_y)

        trg_feat = self.feature_extractor(trg_x)

        coral_loss = self.coral(src_feat, trg_feat)
        mmd_loss = self.mmd(src_feat, trg_feat)
        cond_ent_loss = self.cond_ent(trg_feat)

        loss = self.hparams["coral_wt"] * coral_loss + \
               self.hparams["mmd_wt"] * mmd_loss + \
               self.hparams["cond_ent_wt"] * cond_ent_loss + \
               self.hparams["src_cls_loss_wt"] * src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Coral_loss': coral_loss.item(), 'MMD_loss': mmd_loss.item(),
                'cond_ent_wt': cond_ent_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class DANN(Algorithm):
    """
    DANN: https://arxiv.org/abs/1505.07818
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(DANN, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier_T(configs)  #训练教师网络，修改为T
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.domain_classifier = Discriminator_t(configs) #训练教师网络，修改为T

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )
        self.hparams = hparams
        self.device = device

    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # zero grad
        self.optimizer.zero_grad()
        self.optimizer_disc.zero_grad()

        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)

        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        trg_feat = self.feature_extractor(trg_x)

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

        # Domain classification loss
        # source
        src_feat_reversed = ReverseLayerF.apply(src_feat, alpha)
        src_domain_pred = self.domain_classifier(src_feat_reversed)
        src_domain_loss = self.cross_entropy(src_domain_pred, domain_label_src.long())

        # target
        trg_feat_reversed = ReverseLayerF.apply(trg_feat, alpha)
        trg_domain_pred = self.domain_classifier(trg_feat_reversed)
        trg_domain_loss = self.cross_entropy(trg_domain_pred, domain_label_trg.long())

        # Total domain loss
        domain_loss = src_domain_loss + trg_domain_loss

        loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + \
               self.hparams["domain_loss_wt"] * domain_loss

        loss.backward()
        self.optimizer.step()
        self.optimizer_disc.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class CDAN(Algorithm):
    """
    CDAN: https://arxiv.org/abs/1705.10667
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(CDAN, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.domain_classifier = Discriminator_CDAN(configs)
        self.random_layer = RandomLayer([configs.features_len * configs.final_out_channels, configs.num_classes],
                                        configs.features_len * configs.final_out_channels)

        # optimizers
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # hparams
        self.hparams = hparams
        self.criterion_cond = ConditionalEntropyLoss().to(device)
        self.device = device

    def update(self, src_x, src_y, trg_x):
        # prepare true domain labels
        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

        # source features and predictions
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        # target features and predictions
        trg_feat = self.feature_extractor(trg_x)
        trg_pred = self.classifier(trg_feat)

        # concatenate features and predictions
        feat_concat = torch.cat((src_feat, trg_feat), dim=0)
        pred_concat = torch.cat((src_pred, trg_pred), dim=0)

        # Domain classification loss
        feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1)).detach()
        disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
        disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)

        # update Domain classification
        self.optimizer_disc.zero_grad()
        disc_loss.backward()
        self.optimizer_disc.step()

        # prepare fake domain labels for training the feature extractor
        domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
        domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

        # Repeat predictions after updating discriminator
        feat_x_pred = torch.bmm(pred_concat.unsqueeze(2), feat_concat.unsqueeze(1))
        disc_prediction = self.domain_classifier(feat_x_pred.view(-1, pred_concat.size(1) * feat_concat.size(1)))
        # loss of domain discriminator according to fake labels

        domain_loss = self.cross_entropy(disc_prediction, domain_label_concat)

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

        # conditional entropy loss.
        loss_trg_cent = self.criterion_cond(trg_pred)

        # total loss
        loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] * domain_loss + \
               self.hparams["cond_ent_wt"] * loss_trg_cent

        # update feature extractor
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                'cond_ent_loss': loss_trg_cent.item()}


class DIRT(Algorithm):
    """
    DIRT-T: https://arxiv.org/abs/1802.08735
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(DIRT, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.domain_classifier = Discriminator(configs)

        # optimizers
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.optimizer_disc = torch.optim.Adam(
            self.domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # hparams
        self.hparams = hparams

        # criterion
        self.criterion_cond = ConditionalEntropyLoss().to(device)
        self.vat_loss = VAT(self.network, device).to(device)

        # device for further usage
        self.device = device

        self.ema = EMA(0.998)
        self.ema.register(self.network)

    def update(self, src_x, src_y, trg_x):
        # prepare true domain labels
        domain_label_src = torch.ones(len(src_x)).to(self.device)
        domain_label_trg = torch.zeros(len(trg_x)).to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0).long()

        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        # target features and predictions
        trg_feat = self.feature_extractor(trg_x)
        trg_pred = self.classifier(trg_feat)

        # concatenate features and predictions
        feat_concat = torch.cat((src_feat, trg_feat), dim=0)

        # Domain classification loss
        disc_prediction = self.domain_classifier(feat_concat.detach())
        disc_loss = self.cross_entropy(disc_prediction, domain_label_concat)

        # update Domain classification
        self.optimizer_disc.zero_grad()
        disc_loss.backward()
        self.optimizer_disc.step()

        # prepare fake domain labels for training the feature extractor
        domain_label_src = torch.zeros(len(src_x)).long().to(self.device)
        domain_label_trg = torch.ones(len(trg_x)).long().to(self.device)
        domain_label_concat = torch.cat((domain_label_src, domain_label_trg), 0)

        # Repeat predictions after updating discriminator
        disc_prediction = self.domain_classifier(feat_concat)

        # loss of domain discriminator according to fake labels
        domain_loss = self.cross_entropy(disc_prediction, domain_label_concat)

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)

        # conditional entropy loss.
        loss_trg_cent = self.criterion_cond(trg_pred)

        # Virual advariarial training loss
        loss_src_vat = self.vat_loss(src_x, src_pred)
        loss_trg_vat = self.vat_loss(trg_x, trg_pred)
        total_vat = loss_src_vat + loss_trg_vat
        # total loss
        loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["domain_loss_wt"] * domain_loss + \
               self.hparams["cond_ent_wt"] * loss_trg_cent + self.hparams["vat_loss_wt"] * total_vat

        # update exponential moving average
        self.ema(self.network)

        # update feature extractor
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'Domain_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item(),
                'cond_ent_loss': loss_trg_cent.item()}


class HoMM(Algorithm):
    """
    HoMM: https://arxiv.org/pdf/1912.11976.pdf
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(HoMM, self).__init__(configs)

        self.coral = CORAL()

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams
        self.device = device
        self.HoMM_loss = HoMM_loss()

    def update(self, src_x, src_y, trg_x):
        # extract source features
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        # extract target features
        trg_feat = self.feature_extractor(trg_x)
        trg_pred = self.classifier(trg_feat)

        # calculate source classification loss
        src_cls_loss = self.cross_entropy(src_pred, src_y)

        # calculate lmmd loss
        domain_loss = self.HoMM_loss(src_feat, trg_feat)

        # calculate the total loss
        loss = self.hparams["domain_loss_wt"] * domain_loss + \
               self.hparams["src_cls_loss_wt"] * src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'HoMM_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class DDC(Algorithm):
    """
    DDC: https://arxiv.org/abs/1412.3474
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(DDC, self).__init__(configs)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams
        self.device = device
        self.mmd_loss = MMD_loss()

    def update(self, src_x, src_y, trg_x):
        # extract source features
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        # extract target features
        trg_feat = self.feature_extractor(trg_x)

        # calculate source classification loss
        src_cls_loss = self.cross_entropy(src_pred, src_y)

        # calculate mmd loss
        domain_loss = self.mmd_loss(src_feat, trg_feat)

        # calculate the total loss
        loss = self.hparams["domain_loss_wt"] * domain_loss + \
               self.hparams["src_cls_loss_wt"] * src_cls_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'MMD_loss': domain_loss.item(), 'Src_cls_loss': src_cls_loss.item()}


class UDA_KDS(Algorithm):
    """
    AdvCDKD
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(UDA_KDS, self).__init__(configs)
        # 以下各个部件都按PPT里面那张图来命名 可以逐个对照
        self.task_learner = CNNT(configs)
        # self.task_learner = ItranModel(configs)
        # self.task_learner = TransformerModel(configs)
        self.task_classifier = classifier(configs) #注意和joint_classifier的尺寸不一样
        self.num_tasks = len(configs.scenarios)
        self.task_network = nn.Sequential(self.task_learner, self.task_classifier)
        # 每个任务的特有网络依旧用列表组织

        self.task_networks = nn.ModuleList([copy.deepcopy(self.task_network) for _ in range(self.num_tasks)])

        self.task_optimizers = [torch.optim.Adam(
            list(network.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
            betas=(0.5, 0.99))
            for network in self.task_networks]

        self.hparams = hparams
        self.device = device


    def update(self, datas, labels, step, epoch, len_dataloader, is_data_available):
        # 梯度清零
        for task_optimizer in self.task_optimizers:
            task_optimizer.zero_grad()

        LCET = []
        # LfT = []
        LCET_total = 0
        # LfT_total = 0
        for task_idx, task_data in enumerate(datas): # task_data torch.Size([bsize, 9, 128])
            # 获取当前任务的特有网络
            # if task_data.size(0) == 0:  # 如果 task_data 为空则跳过
            #     continue
            task_network = self.task_networks[task_idx]
            task_learner, task_classifier = task_network
            task_feat = task_learner(task_data)  # forward torch.Size([bsize, 32])
            task_pred = task_classifier(task_feat)  # forward torch.Size([bsize, 6])
            LCE_task = self.MSELoss(task_pred.squeeze(), labels[task_idx]) #一个值

            LCET.append(LCE_task)
            LCET_total += LCE_task
            LCE_task.backward()
            self.task_optimizers[task_idx].step()
        # # .item()用于把张量转换为数值
        return {'LCET': LCET_total.item() }


# class UDA_KD(Algorithm):
#     """
#     AdvCDKD
#     """
#
#     def __init__(self, backbone_fe, configs, hparams, device):
#         super(UDA_KD, self).__init__(configs)
#         from models import models
#         # 教师网络
#         # self.t_feature_extractor = models.CNN_T(configs)
#         # self.t_classifier = models.classifier_T(configs)
#         # self.network_t = nn.Sequential(self.t_feature_extractor, self.t_classifier)
#
#         # 以下各个部件都按PPT里面那张图来命名 可以逐个对照
#         self.joint_learner = CNNT(configs)
#         # self.joint_learner = ItranModel(configs)
#         # self.joint_learner = TransformerModel(configs)
#         self.task_learner = backbone_fe(configs)
#         self.joint_classifier = classifier(configs)
#         self.task_classifier = classifier(configs) #注意和joint_classifier的尺寸不一样
#         self.joint_network = nn.Sequential(self.joint_learner, self.joint_classifier) # 联合网络（是不是应该更大？）
#         self.num_tasks = len(configs.scenarios)
#         self.task_network = nn.Sequential(self.task_learner, self.task_classifier)
#         # 每个任务的特有网络依旧用列表组织
#
#         self.task_networks = nn.ModuleList([copy.deepcopy(self.task_network) for _ in range(self.num_tasks)])
#
#         self.joint_discriminator = Discriminator(configs)
#
#         self.joint_optimizer = torch.optim.Adam(
#             list(self.joint_network.parameters()), #联合模型需要优化的参数
#             lr=hparams["learning_rate"],
#             weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
#         )
#
#         self.task_optimizers = [torch.optim.Adam(
#             list(network.parameters()),
#             lr=hparams["learning_rate"],
#             weight_decay=hparams["weight_decay"],
#             betas=(0.5, 0.99))
#             for network in self.task_networks]
#
#
#         self.disc_optimizer = torch.optim.Adam(
#             self.joint_discriminator.parameters(),
#             lr=hparams["learning_rate"],
#             weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
#         )
#
#         self.hparams = hparams
#         self.device = device
#         self.temperature = hparams["temperature"]
#
#
#     def update(self, datas, labels, step, epoch, len_dataloader, is_data_available):
#         p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] #一个介于0到1之间的比例值，表示训练进度的百分比
#         alpha = 2. / (1. + np.exp(-10 * p)) - 1  #ReservedLayer的参数 在训练初期（p较小），alpha的值接近0；随着训练的进行（p增大），alpha 的值逐渐增加，最终接近1
#
#         stacked_datas = torch.cat(datas, dim=0).view(-1, datas[0].size(-2), datas[0].size(-1)) #把所有任务的样本数据堆叠起来 torch.Size([bsize*num_tasks, 9, 128])
#         stacked_feats = self.joint_learner(stacked_datas) #torch.Size([bsize*num_tasks, 32(final_out_channels)])
#         joint_feat = stacked_feats.view(self.num_tasks, -1, stacked_feats.size(-1)) #每个任务的特征分开 torch.Size([num_tasks, bsize, 32])
#
#         self.joint_optimizer.zero_grad()  #梯度清零
#         self.disc_optimizer.zero_grad()
#
#         stacked_pred = self.joint_classifier(stacked_feats) #算法第5行 torch.Size([num_tasks*bsize, 6(num_classes)])
#         joint_pred = stacked_pred.view(self.num_tasks, -1, stacked_pred.size(-1))  # 每个任务的pred特征分开 torch.Size([num_tasks, bsize, 6])
#         joint_pred_prob = torch.nn.functional.softmax(joint_pred, dim=2) #torch.Size([num_tasks, bsize, 6])
#         # contribution = torch.sum(-joint_pred_prob * torch.log(joint_pred_prob), dim=2)  # torch.Size([num_tasks, bsize]) #意义不明！！
#         stacked_labels = torch.cat(labels, dim=0).view(-1) #标签按照数据的方式堆叠  torch.Size([num_tasks*bsize])
#         LCE = self.MSELoss(stacked_pred.squeeze(), stacked_labels)  # 计算LCE
#         # print(stacked_feats)
#         # print(stacked_labels)
#         # print(LCE)
#
#         LDJ = 0
#         weights = []
#         for task, features in enumerate(joint_feat): #循环次数为任务数  features形状为torch.Size([bsize, 32])
#             task_labels = torch.full((len(features),), task).to(self.device) #标签代表任务 分别是0,1,2.。。 torch.Size([bszie])
#             reserved_feats = ReverseLayerF.apply(features, alpha) #经过梯度反转层  torch.Size([bsize, 32])
#             task_preds = self.joint_discriminator(reserved_feats)  # torch.Size([bsize, 3(num_tasks)])
#             task_loss = self.cross_entropy(task_preds, task_labels.long())  # 计算交叉熵损失 两个参数不需要形状相同
#             # predicted_classes = torch.argmax(task_preds, dim=1)
#             # correct_predictions = (predicted_classes == task_labels)
#             # # 计算预测正确的个数
#             # num_correct = correct_predictions.sum().item()
#             # accuracy = num_correct / len(task_labels)
#             # print(f"准确率: {accuracy:.2f}")
#             LDJ += task_loss
#
#             task_preds_prob = torch.nn.functional.softmax(task_preds, dim=1)  # 转换成0-1的概率分布(n(任务数）列加起来为1）torch.Size([bszie, 3(num_tasks)])
#             weight = torch.sum(-task_preds_prob * torch.log(task_preds_prob), dim=1) #算一个熵，用熵决定权重(不知道对不对) torch.Size([bsize])
#             weights.append(weight)
#
#         loss1 = self.hparams["cls_loss_wt"] * LCE + self.hparams["domain_loss_wt"] * LDJ
#         # print("LCE", LCE.item())
#         # print("LDJ", LDJ)
#         loss1.backward()
#         self.joint_optimizer.step()
#         self.disc_optimizer.step()
#
#         # 梯度清零
#         for task_optimizer in self.task_optimizers:
#             task_optimizer.zero_grad()
#
#         LCET = []
#         # LfT = []
#         LCET_total = 0
#         # LfT_total = 0
#         for task_idx, task_data in enumerate(datas): # task_data torch.Size([bsize, 9, 128])
#             # 获取当前任务的特有网络
#             task_network = self.task_networks[task_idx]
#             task_learner, task_classifier = task_network
#             task_feat = task_learner(task_data)  # forward torch.Size([bsize, 32])
#             shared_feat = weights[task_idx].detach().unsqueeze(1) * joint_feat[task_idx].detach() #执行按元素乘 （bsize*1）*(bsize*32) = torch.Size([bszie, 32])
#             # shared_feat = joint_feat[task_idx].detach()
#             # task_feat_plus = contribution[task_idx].detach().unsqueeze(1) * torch.cat((task_feat,shared_feat),dim=1) #执行按元素乘 （bsize*1）*(bsize*64) = torch.Size([bszie, 64])
#             # print(task_feat.shape)
#             # task_feat_plus = torch.cat((task_feat, shared_feat),dim=1)  # 执行按元素乘 （bsize*1）*(bsize*64) = torch.Size([bszie, 64])
#             task_feat_plus = (task_feat + shared_feat) / 2
#             # print(task_feat_plus.shape)
#             # Lf = torch.nn.functional.softmax(task_feat, dim=1) * \
#             #         (torch.log(torch.nn.functional.softmax(task_feat, dim=1))
#             #         - torch.nn.functional.log_softmax(joint_feat[task_idx].detach(), dim=1))  # KL散度??? detach很关键
#             # Lf = (Lf.sum(dim=1)*weights[task_idx].detach()).sum(dim=0) / task_feat.size(0)
#             # LfT.append(Lf)
#             # LfT_total += Lf
#             task_pred = task_classifier(task_feat_plus)  # forward torch.Size([bsize, 6])
#             LCE_task = self.MSELoss(task_pred.squeeze(), labels[task_idx]) #一个值
#
#             LCET.append(LCE_task)
#             LCET_total += LCE_task
#             # 13: 对每一个task 进行更新
#             LCE_task.backward()
#             self.task_optimizers[task_idx].step()
#         # loss2 = LfT_total + LJK_total
#         # loss2.backward()
#         # for task_optimizer in self.task_optimizers:
#         #     task_optimizer.step()
#         total_loss = loss1 + LCET_total
#         # # .item()用于把张量转换为数值
#         return {'Total_loss': total_loss.item(), 'LCE': LCE.item(), 'LDJ': LDJ.item(), 'LCET': LCET_total.item() }

class UDA_KD(Algorithm):
    """
    AdvCDKD
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(UDA_KD, self).__init__(configs)
        from models import models

        # 以下各个部件都按PPT里面那张图来命名 可以逐个对照
        self.joint_learner = backbone_fe(configs)
        # self.joint_learner = ItranModel(configs)
        # self.joint_learner = TransformerModel(configs)
        self.task_learner = CNNT(configs)
        self.joint_classifier = classifier(configs)
        self.task_classifier = classifier(configs) #注意和joint_classifier的尺寸不一样
        self.joint_network = nn.Sequential(self.joint_learner, self.joint_classifier) # 联合网络（是不是应该更大？）
        self.num_tasks = len(configs.scenarios)
        self.task_network = nn.Sequential(self.task_learner, self.task_classifier)
        # 每个任务的特有网络依旧用列表组织

        self.task_networks = nn.ModuleList([copy.deepcopy(self.task_network) for _ in range(self.num_tasks)])

        self.joint_discriminator = Discriminator(configs)

        self.joint_optimizer = torch.optim.Adam(
            list(self.joint_network.parameters()), #联合模型需要优化的参数
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.task_optimizers = [torch.optim.Adam(
            list(network.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
            betas=(0.5, 0.99))
            for network in self.task_networks]


        self.disc_optimizer = torch.optim.Adam(
            self.joint_discriminator.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.hparams = hparams
        self.device = device
        self.temperature = hparams["temperature"]


    def update(self, datas, labels, step, epoch, len_dataloader, is_data_available):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] #一个介于0到1之间的比例值，表示训练进度的百分比
        alpha = (p - 2.2) / (24.1 - 2.2)  #ReservedLayer的参数 在训练初期（p较小），alpha的值接近0；随着训练的进行（p增大），alpha 的值逐渐增加，最终接近1

        stacked_datas = torch.cat(datas, dim=0).view(-1, datas[0].size(-2), datas[0].size(-1)) #把所有任务的样本数据堆叠起来 torch.Size([bsize*num_tasks, 9, 128])
        stacked_feats = self.joint_learner(stacked_datas) #torch.Size([bsize*num_tasks, 32(final_out_channels)])
        joint_feat = stacked_feats.view(self.num_tasks, -1, stacked_feats.size(-1)) #每个任务的特征分开 torch.Size([num_tasks, bsize, 32])

        stacked_pred = self.joint_classifier(stacked_feats) #算法第5行 torch.Size([num_tasks*bsize, 6(num_classes)])
        joint_pred = stacked_pred.view(self.num_tasks, -1, stacked_pred.size(-1))  # 每个任务的pred特征分开 torch.Size([num_tasks, bsize, 6])
        joint_pred_prob = torch.nn.functional.softmax(joint_pred, dim=2) #torch.Size([num_tasks, bsize, 6])
        # contribution = torch.sum(-joint_pred_prob * torch.log(joint_pred_prob), dim=2)  # torch.Size([num_tasks, bsize]) #意义不明！！
        stacked_labels = torch.cat(labels, dim=0).view(-1) #标签按照数据的方式堆叠  torch.Size([num_tasks*bsize])
        LCE = self.MSELoss(stacked_pred.squeeze(), stacked_labels)  # 计算LCE

        absolute_error = torch.abs(stacked_pred.squeeze() - stacked_labels)
        beta = 10.0
        weight = torch.exp(absolute_error / beta)
        # print(weight)
        weights = weight.view(self.num_tasks, -1)

        task_labels = torch.arange(self.num_tasks).repeat_interleave(self.hparams["batch_size"]).to(self.device)
        task_preds = self.joint_discriminator(stacked_feats.detach())
        loss_disc = self.cross_entropy(task_preds, task_labels.long())

        self.disc_optimizer.zero_grad()
        loss_disc.backward()
        self.disc_optimizer.step()

        reserved_feats = ReverseLayerF.apply(stacked_feats, alpha)
        task_preds = self.joint_discriminator(reserved_feats)
        loss_feat = -self.cross_entropy(task_preds, task_labels.long())

        self.joint_optimizer.zero_grad()
        loss1 = self.hparams["cls_loss_wt"] * LCE + self.hparams["domain_loss_wt"] * loss_feat
        loss1.backward()
        self.joint_optimizer.step()

        # 梯度清零
        for task_optimizer in self.task_optimizers:
            task_optimizer.zero_grad()

        LCET = []
        LCET_total = 0
        for task_idx, task_data in enumerate(datas): # task_data torch.Size([bsize, 9, 128])
            # 获取当前任务的特有网络
            task_network = self.task_networks[task_idx]
            task_learner, task_classifier = task_network
            task_feat = task_learner(task_data)  # forward torch.Size([bsize, 32])
            shared_feat = weights[task_idx].detach().unsqueeze(1) * joint_feat[task_idx].detach() #执行按元素乘 （bsize*1）*(bsize*32) = torch.Size([bszie, 32])
            shared_feat = joint_feat[task_idx].detach()
            # task_feat_plus = contribution[task_idx].detach().unsqueeze(1) * torch.cat((task_feat,shared_feat),dim=1) #执行按元素乘 （bsize*1）*(bsize*64) = torch.Size([bszie, 64])
            # print(task_feat.shape)
            # task_feat_plus = torch.cat((task_feat, shared_feat),dim=1)  # 执行按元素乘 （bsize*1）*(bsize*64) = torch.Size([bszie, 64])
            task_feat_plus = (task_feat + shared_feat) / 2
            # print(task_feat_plus.shape)
            task_pred = task_classifier(task_feat_plus)  # forward torch.Size([bsize, 6])
            criterion = nn.MSELoss(reduction='none')
            loss = criterion(task_pred.squeeze(), labels[task_idx])  # 一个值
            weighted_loss = loss * weights[task_idx].detach()
            LCE_task = weighted_loss.sum()
            # LCE_task = self.MSELoss(task_pred.squeeze(), labels[task_idx]) #一个值

            LCET.append(LCE_task)
            LCET_total += LCE_task
            LCE_task.backward()
            self.task_optimizers[task_idx].step()
        total_loss = loss1 + LCET_total
        return {'Total_loss': total_loss.item(), 'LCE': LCE.item(), 'LCET': LCET_total.item() }


class JointUKD(Algorithm):
    """
    JointUKD
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(JointUKD, self).__init__(configs)
        from models import models
        self.t_feature_extractor = models.CNN_T(configs)
        self.t_classifier = models.classifier_T(configs)
        self.network_t = nn.Sequential(self.t_feature_extractor, self.t_classifier)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.domain_classifier = Discriminator(configs)

        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.network_t.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.hparams = hparams
        self.device = device
        self.temperature = hparams["temperature"]


    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # zero grad
        self.optimizer.zero_grad()
        self.network_t.train()

        # Teacher inference on Source and Target
        src_feat_t = self.t_feature_extractor(src_x)
        src_pred_t = self.t_classifier(src_feat_t)
        src_pred_t_soften = torch.nn.functional.log_softmax(src_pred_t/self.temperature,dim=1)

        trg_feat_t = self.t_feature_extractor(trg_x)
        trg_pred_t = self.t_classifier(trg_feat_t)
        trg_pred_t_soften = torch.nn.functional.log_softmax(trg_pred_t / self.temperature, dim=1)

        # Student inference on Source and Target
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)
        src_pred_s_soften = torch.nn.functional.log_softmax(src_pred / self.temperature, dim=1)

        trg_feat = self.feature_extractor(trg_x)
        trg_pred = self.classifier(trg_feat)
        trg_pred_s_soft = torch.nn.functional.log_softmax(trg_pred / self.temperature, dim=1)

        from mmd import MMD_loss
        mmd_loss = MMD_loss()(src_feat_t,trg_feat_t)
        loss_ce_t = self.cross_entropy(src_pred_t, src_y)
        loss_tda = mmd_loss + 0.8 * loss_ce_t

        loss_tkd = torch.nn.functional.kl_div(trg_pred_s_soft, trg_pred_t_soften, reduction='batchmean', log_target=True)

        loss_kd_src = torch.nn.functional.kl_div(src_pred_s_soften, src_pred_t_soften, reduction='batchmean', log_target=True)
        loss_ce_s = self.cross_entropy(src_pred, src_y)

        loss_skd = loss_kd_src + 0.8 * loss_ce_s
        import math
        g = math.log10(0.9/0.1) / self.hparams["num_epochs"]
        beta = 0.1 * math.exp(g*epoch)
        loss = (1-beta) * loss_tda + beta*(loss_skd+loss_tkd)
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'loss_tda': loss_tda.item(), 'loss_skd': loss_skd.item(), 'loss_tkd':loss_tkd.item()}


class AAD(Algorithm):
    """
    AAD
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(AAD, self).__init__(configs)
        from models import models
        self.t_feature_extractor = models.CNN_T(configs)
        self.t_classifier = models.classifier_T(configs)
        self.network_t = nn.Sequential(self.t_feature_extractor, self.t_classifier)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.data_domain_classifier = Discriminator(configs)
        self.feature_domain_classifier = Discriminator_fea(configs)
        self.adapter = Adapter(configs)

        self.optimizer = torch.optim.Adam(
            list(self.network.parameters()) + list(self.adapter.parameters()),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.optimizer_disc = torch.optim.Adam(
            self.data_domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.optimizer_feat = torch.optim.Adam(
            self.feature_domain_classifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.hparams = hparams
        self.device = device
        self.temperature = hparams["temperature"]


    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        self.network_t.eval()

        real_label = 1
        fake_label = 0

        ########################################################
        # (1) update D network: maximize log(D(fea_t)) + log(1-D(G(x))
        # fea_t: feature from teacher network
        # x: input data
        # G(x): feature from student network
        ########################################################
        self.optimizer_feat.zero_grad()

        # Format Batch
        src_feat_t = self.t_feature_extractor(src_x)
        src_feat_t = Variable(src_feat_t, requires_grad=False)

        f_domain_label = torch.full((src_x.shape[0],), real_label, dtype=torch.float, device=self.device)

        # Forward pass real batch through D
        output = self.feature_domain_classifier(src_feat_t).view(-1)
        # Calculate loss on all-real batch
        errD_real = nn.BCELoss()(output, f_domain_label)
        # Calculate gradients for D in backward pass
        errD_real.backward()

        # Train with all-fake batch, Generate fake features with G
        # Student Forward
        src_feat = self.feature_extractor(src_x)
        src_feat_hint = self.adapter(src_feat)

        f_domain_label.fill_(fake_label)
        # Classify all fake batch with D

        output = self.feature_domain_classifier(src_feat_hint.detach()).view(-1)
        # output = self.feature_domain_classifier(src_feat.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = nn.BCELoss()(output, f_domain_label)
        # Calculate the gradients for this batch
        errD_fake.backward()

        # Add the gradients from all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        self.optimizer_feat.step()

        ########################################################
        # (2) update G network: maximize log(D(G(x))
        ########################################################

        # zero grad
        self.optimizer.zero_grad()
        self.optimizer_disc.zero_grad()

        src_pred_t = self.t_classifier(src_feat_t)
        src_pred_t_soften = torch.nn.functional.log_softmax(src_pred_t / self.temperature, dim=1)


        src_pred = self.classifier(src_feat)
        src_pred_s_soften = torch.nn.functional.log_softmax(src_pred / self.temperature, dim=1)

        # fake labels are real for generator cost
        f_domain_label.fill_(real_label)
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = self.feature_domain_classifier(src_feat_hint).view(-1)

        # Calculate G's loss based on this output
        errG = nn.BCELoss()(output, f_domain_label)

        # Add KD loss
        soft_loss = torch.nn.functional.kl_div(src_pred_s_soften, src_pred_t_soften, reduction='batchmean', log_target=True)
        kd_loss = soft_loss * self.temperature ** 2

        # Task classification  Loss
        src_cls_loss = self.cross_entropy(src_pred.squeeze(), src_y)


        loss = self.hparams["src_cls_loss_wt"] * src_cls_loss + self.hparams["soft_loss_wt"] * kd_loss + self.hparams ['errG'] * errG

        loss.backward()
        self.optimizer.step()
        self.optimizer_disc.step()

        return {'Total_loss': loss.item(), 'Src_cls_loss': src_cls_loss.item(), 'KD_loss':kd_loss.item(), 'errD': errD.item(), 'errG':errG.item() }


class MobileDA(Algorithm):
    """
    MobileDA
    """

    def __init__(self, backbone_fe, configs, hparams, device):
        super(MobileDA, self).__init__(configs)
        from models import models
        self.t_feature_extractor = models.CNN_T(configs)
        self.t_classifier = models.classifier_T(configs)
        self.network_t = nn.Sequential(self.t_feature_extractor, self.t_classifier)

        self.feature_extractor = backbone_fe(configs)
        self.classifier = classifier(configs)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.domain_classifier = Discriminator(configs)

        self.coral = CORAL()

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"], betas=(0.5, 0.99)
        )

        self.hparams = hparams
        self.device = device
        self.temperature = hparams["temperature"]


    def update(self, src_x, src_y, trg_x, step, epoch, len_dataloader):
        p = float(step + epoch * len_dataloader) / self.hparams["num_epochs"] + 1 / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # zero grad
        self.optimizer.zero_grad()
        self.network_t.eval()

        trg_feat_t = self.t_feature_extractor(trg_x)
        trg_pred_t = self.t_classifier(trg_feat_t)
        trg_pred_t_soften = torch.nn.functional.log_softmax(trg_pred_t / self.temperature, dim=1)

        # Student inference on Source and Target
        src_feat = self.feature_extractor(src_x)
        src_pred = self.classifier(src_feat)

        trg_feat = self.feature_extractor(trg_x)
        trg_pred = self.classifier(trg_feat)
        trg_pred_s_soft = torch.nn.functional.log_softmax(trg_pred / self.temperature, dim=1)

        loss_ce_s = self.cross_entropy(src_pred, src_y)
        loss_soft = torch.nn.functional.kl_div(trg_pred_s_soft, trg_pred_t_soften, reduction='batchmean',
                                              log_target=True)

        loss_dc = self.coral(src_feat, trg_feat)

        loss = loss_ce_s + 0.7* loss_soft + 0.3 * loss_dc
        loss.backward()
        self.optimizer.step()

        return {'Total_loss': loss.item(), 'loss_ce': loss_ce_s.item(), 'loss_soft': loss_soft.item(), 'loss_dc':loss_dc.item()}


