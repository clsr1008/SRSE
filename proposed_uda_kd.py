import os
import argparse
import warnings
from itertools import zip_longest

import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

import torch
import torch.nn.functional as F
import torch.nn as nn

import os
import wandb
import pandas as pd
import numpy as np
from dataloader.dataloader import data_generator, few_shot_data_generator, generator_percentage_of_data
from configs.data_model_configs import get_dataset_class
from configs.hparams import get_hparams_class

from configs.sweep_params import sweep_alg_hparams
from utils import fix_randomness, copy_Files, starting_logs, save_checkpoint, _calc_metrics
from utils import calc_dev_risk, calculate_risk
import warnings

import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

import collections
from algorithms.algorithms import get_algorithm_class
from models.models import get_backbone_class
from utils import AverageMeter


torch.backends.cudnn.benchmark = True  # to fasten TCN

class adv_cross_domain_kd_trainer(object):
    """
   This class contain the main training functions for our AdAtime
    """
    def __init__(self, args):
        self.da_method = args.da_method  # Selected  DA Method
        self.dataset = args.dataset  # Selected  Dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)  # device
        self.num_sweeps = args.num_sweeps

        # Exp Description
        self.run_description = args.run_description
        self.experiment_description = args.experiment_description
        # sweep parameters
        self.is_sweep = args.is_sweep
        self.sweep_project_wandb = args.sweep_project_wandb
        self.wandb_entity = args.wandb_entity
        self.hp_search_strategy = args.hp_search_strategy
        self.metric_to_minimize = args.metric_to_minimize

        # paths
        self.home_path = os.getcwd() #文件目录
        self.save_dir = args.save_dir #experiments_logs
        self.data_path = os.path.join(args.data_path, self.dataset) #数据集存放位置
        self.create_save_dir() #创建experiments_logs目录

        # Specify runs
        self.num_runs = args.num_runs

        # get dataset and base model configs 返回相应的类
        self.dataset_configs, self.hparams_class = self.get_configs()

        # to fix dimension of features in classifier and discriminator networks.
        self.dataset_configs.final_out_channels = self.dataset_configs.tcn_final_out_channles if args.backbone == "TCN" else self.dataset_configs.final_out_channels

        # Specify number of hparams
        self.default_hparams = {**self.hparams_class.alg_hparams[self.da_method],
                                **self.hparams_class.train_params}

    def sweep(self):
        # sweep configurations
        sweep_runs_count = self.num_sweeps
        sweep_config = {
            'method': self.hp_search_strategy,
            'metric': {'name': self.metric_to_minimize, 'goal': 'minimize'},
            'name': self.da_method,
            'parameters': {**sweep_alg_hparams[self.da_method]}
        }
        sweep_id = wandb.sweep(sweep_config, project=self.sweep_project_wandb, entity=self.wandb_entity)

        wandb.agent(sweep_id, self.train, count=sweep_runs_count)  # Training with sweep

        # resuming sweep
        # wandb.agent('8wkaibgr', self.train, count=25,project='HHAR_SA_Resnet', entity= 'iclr_rebuttal' )

    def train(self):
        if self.is_sweep:
            wandb.init(config=self.default_hparams)
            run_name = f"sweep_{self.dataset}"
        else:
            run_name = f"{self.run_description}"
            wandb.init(config=self.default_hparams, mode="online", name=run_name)

        self.hparams = wandb.config
        # Logging(exp日志目录路径)
        self.exp_log_dir = os.path.join(self.save_dir, self.experiment_description, run_name)
        os.makedirs(self.exp_log_dir, exist_ok=True)
        copy_Files(self.exp_log_dir)  # save a copy of training files:

        scenarios = self.dataset_configs.scenarios  # return the scenarios given a specific dataset.
        # # 定义的实验效果评价指标（记录每一次run的指标）
        # self.metrics = {'accuracy': [], 'f1_score': [], 'src_risk': [], 'few_shot_trg_risk': [],
        #                 'trg_risk': [], 'dev_risk': []}

        # for i in scenarios:
        #     src_id = i[0]
        #     trg_id = i[1]

        for run_id in range(self.num_runs):  # specify number of consecutive runs
            # fixing random seed (固定随机性，同样的随机种子下多次运行实验时，实验的结果保持一致）
            fix_randomness(run_id)

            # Logging（返回日志对象和scenario日志目录路径）
            self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                               scenarios, run_id)

            # Load data（按照batch加载数据）
            self.load_data(scenarios)

            # get student algorithm（学生模型 UDA_KD和CNN）
            algorithm_class = get_algorithm_class(self.da_method)
            backbone_fe = get_backbone_class(self.backbone)

            algorithm = algorithm_class(backbone_fe, self.dataset_configs, self.hparams, self.device)

            # Load Pre-trained Teacher model（加载教师模型的参数）
            # best_teacher = src_id + '_to_' + trg_id + '_checkpoint.pt'
            # model_t_name = os.path.join(self.save_dir, self.dataset, 'Teacher_CNN', best_teacher)
            # checkpoint = torch.load(model_t_name)
            # algorithm.network_t.load_state_dict(checkpoint["network_dict"])
            # print(torch.__version__)
            # print(torch.cuda.is_available())
            # print("CUDA version:", torch.version.cuda)  # 检查 PyTorch 使用的 CUDA 版本
            algorithm.to(self.device)

            # Average meters 跟踪并计算每个epoch或批次中的平均损失值、准确率或其他指标
            loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
            # training..
            for epoch in range(1, self.hparams["num_epochs"] + 1):
                joint_loaders = enumerate(zip(*self.train_dls)) #把所有任务的数据迭代器用列表组织起来，做成联合迭代器,截短做法
                # empty_tensor = torch.empty(0)
                # joint_loaders = enumerate(zip_longest(*self.train_dls, fillvalue=(empty_tensor, empty_tensor)))
                len_dataloader = min(len(list(loader)) for loader in self.train_dls) #所有任务的loader中批次数量的最小值 6
                # print(len_dataloader)
                algorithm.train()  # 进入训练模式

                for step, all_batches in joint_loaders:  # step代表第几批次 all_batches是迭代器返回的某个批次（包含所有任务）
                    is_data_available = [batch[0].size(0) != 0 for batch in all_batches]
                    # print(is_data_available)
                    datas, labels = zip(*all_batches) #将数据和标签分离
                    datas = [data.float().to(self.device) for data in datas] #格式转换 datas是列表 每个data代表一个任务 data形状是32*9*128
                    labels = [label.to(self.device) for label in labels] #labels是列表 每个label代表一个任务 长度是32
                    losses = algorithm.update(datas, labels, step, epoch, len_dataloader, is_data_available)  # 核心的训练过程！！！
                    for key, val in losses.items():  # 不断更新各项损失的平均指标（每个loader更新一次）
                        loss_avg_meters[key].update(val, all_batches[0][0].size(0))

                # logging
                self.logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
                for key, val in loss_avg_meters.items():
                    self.logger.debug(f'{key}\t: {val.avg:2.4f}')
                self.logger.debug(f'-------------------------------------')

            self.algorithm = algorithm
            # return #测试点
            # 保存算法模型参数
            save_checkpoint(self.home_path, self.algorithm, scenarios, self.dataset_configs,
                            self.scenario_log_dir, self.hparams)
            # 计算在测试集中target域的损失（顺便记下pred_labels和true_labels）
            if(self.da_method == "UDA_KDS"):
                self.evaluate_s()
            else:
                self.evaluate()
            # 计算和保存acc和F1 score
            self.calc_results_per_run()


        # logging metrics（计算所有场景的得分平均值和标准差，返回一个dataframe形状的exel表）
        self.calc_overall_results()
        average_metrics = {metric: np.mean(value) for (metric, value) in self.metrics.items()} #返回每项指标的均值字典
        wandb.log(average_metrics)
        wandb.log({'hparams': wandb.Table(
            dataframe=pd.DataFrame(dict(self.hparams).items(), columns=['parameter', 'value']),
            allow_mixed_types=True)})
        wandb.log({'avg_results': wandb.Table(dataframe=self.averages_results_df, allow_mixed_types=True)})
        wandb.log({'std_results': wandb.Table(dataframe=self.std_results_df, allow_mixed_types=True)})

    def evaluate(self):
        joint_learner = self.algorithm.joint_learner.to(self.device)
        joint_classifier = self.algorithm.joint_classifier.to(self.device)
        joint_learner.eval()  # 进入评估模式
        joint_classifier.eval()

        task_networks = self.algorithm.task_networks
        for task_learner, task_classifier in task_networks:
            task_learner.eval()
            task_classifier.eval()

        total_loss_ = []

        # self.test_pred_labels = np.array([])
        # self.test_true_labels = np.array([])
        self.test_pred_labels = []  # 存储每个任务的预测标签列表
        self.test_true_labels = []  # 存储每个任务的真实标签列表

        test_loaders = zip(*self.test_dls)  # 把所有任务的数据迭代器用列表组织起来，做成联合迭代器

        for all_batches in test_loaders:  # step代表第几批次 all_batches是迭代器返回的某个批次（包含所有任务）
            datas, labels = zip(*all_batches)  # 将数据和标签分离
            data_list = [data.float().to(self.device) for data in datas]  # 格式转换 datas是列表 每个data代表一个任务 data形状是32*9*128
            labels_list = [label.to(self.device) for label in labels]  # labels是列表 每个label代表一个任务 长度是32

            with torch.no_grad():  # 不计算梯度
                task_loss = 0
                for taskidx, (data, labels) in enumerate(zip(data_list, labels_list)):  # 遍历每一个任务
                    data = data.float().to(self.device)
                    labels = labels.view((-1)).to(self.device)
                    task_learner, task_classifier = task_networks[taskidx]
                    task_fea = task_learner(data)
                    joint_fea = joint_learner(data)
                    # task_pred = task_classifier(torch.cat((task_fea,joint_fea),dim=1)) #重新适应task_classifier的输入尺寸
                    task_pred = task_classifier((task_fea + joint_fea) / 2)
                    # print(task_pred.shape)
                    # task_pred = joint_classifier(joint_fea)
                    # 确保 task_pred 和 labels 形状相同
                    task_pred = torch.squeeze(task_pred)  # 移除 task_pred 中的单维度
                    # print(task_pred)  # 打印 task_pred 的形状
                    # print(labels)  # 打印 labels 的形状
                    fn_loss = nn.MSELoss(reduction='sum')
                    loss = fn_loss(task_pred, labels)
                    task_loss += loss.item()
                    pred = task_pred.detach()

                    # self.test_pred_labels = np.append(self.test_pred_labels, pred.cpu().numpy())
                    # self.test_true_labels = np.append(self.test_true_labels, labels.data.cpu().numpy())
                    # 如果是第一个批次，初始化列表
                    if len(self.test_pred_labels) <= taskidx:
                        self.test_pred_labels.append(pred.cpu().numpy())
                        self.test_true_labels.append(labels.data.cpu().numpy())
                    else:
                        # 将当前批次的标签附加到相应的任务列表中
                        self.test_pred_labels[taskidx] = np.append(self.test_pred_labels[taskidx], pred.cpu().numpy())
                        self.test_true_labels[taskidx] = np.append(self.test_true_labels[taskidx],
                                                                   labels.data.cpu().numpy())

            total_loss_.append(task_loss)

        # # 定义要关注的层名称
        # target_layer_name = "0.block1.conv.0.weight"
        #
        # # 查看 joint_network 的特定层参数
        # print("Joint Network Specific Layer Parameters:")
        # for name, param in joint_learner.named_parameters():
        #     if name == target_layer_name:
        #         print(f"{name}: shape = {param.shape}")
        #         print(f"参数值示例: {param[:2]}")  # 仅打印部分参数值（如前两行）
        #
        # # 查看 task_networks 中每个子网络的特定层参数
        # print("\nTask Networks Specific Layer Parameters:")
        # for i, task_network in enumerate(task_networks):
        #     print(f"Task Network {i + 1} Specific Layer:")
        #     for name, param in task_network.named_parameters():
        #         if name == target_layer_name:
        #             print(f"{name}: shape = {param.shape}")
        #             print(f"参数值示例: {param[:2]}")  # 仅打印部分参数值

        self.test_loss = torch.tensor(total_loss_).mean()  # average loss

    def evaluate_s(self):

        task_networks = self.algorithm.task_networks
        for task_learner, task_classifier in task_networks:
            task_learner.eval()
            task_classifier.eval()

        total_loss_ = []
        # self.test_pred_labels = np.array([])
        # self.test_true_labels = np.array([])
        self.test_pred_labels = []  # 存储每个任务的预测标签列表
        self.test_true_labels = []  # 存储每个任务的真实标签列表

        test_loaders = zip(*self.test_dls)  # 把所有任务的数据迭代器用列表组织起来，做成联合迭代器

        for all_batches in test_loaders:  # step代表第几批次 all_batches是迭代器返回的某个批次（包含所有任务）
            datas, labels = zip(*all_batches)  # 将数据和标签分离
            data_list = [data.float().to(self.device) for data in datas]  # 格式转换 datas是列表 每个data代表一个任务 data形状是32*9*128
            labels_list = [label.to(self.device) for label in labels]  # labels是列表 每个label代表一个任务 长度是32

            with torch.no_grad():  # 不计算梯度
                task_loss = 0
                for taskidx, (data, labels) in enumerate(zip(data_list, labels_list)):  # 遍历每一个任务
                    data = data.float().to(self.device)
                    labels = labels.view((-1)).to(self.device)
                    task_learner, task_classifier = task_networks[taskidx]
                    task_fea = task_learner(data)
                    task_pred = task_classifier(task_fea) #重新适应task_classifier的输入尺寸
                    # 确保 task_pred 和 labels 形状相同
                    task_pred = torch.squeeze(task_pred)  # 移除 task_pred 中的单维度
                    fn_loss = nn.MSELoss(reduction='sum')
                    loss = fn_loss(task_pred, labels)
                    task_loss += loss.item()
                    pred = task_pred.detach()

                    # 如果是第一个批次，初始化列表
                    if len(self.test_pred_labels) <= taskidx:
                        self.test_pred_labels.append(pred.cpu().numpy())
                        self.test_true_labels.append(labels.data.cpu().numpy())
                    else:
                        # 将当前批次的标签附加到相应的任务列表中
                        self.test_pred_labels[taskidx] = np.append(self.test_pred_labels[taskidx], pred.cpu().numpy())
                        self.test_true_labels[taskidx] = np.append(self.test_true_labels[taskidx],
                                                                   labels.data.cpu().numpy())

            total_loss_.append(task_loss)

        self.test_loss = torch.tensor(total_loss_).mean()  # average loss

    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        hparams_class = get_hparams_class(self.dataset)
        return dataset_class(), hparams_class()

    def load_data(self, scenarios):
        train_dls, test_dls = [], []
        for scenario in scenarios: #加载每个任务的训练集迭代器和测试集迭代器 用列表存储不同任务的
            train_dl, test_dl = data_generator(self.data_path, scenario, self.dataset_configs, self.hparams)
            print(len(train_dl))
            train_dls.append(train_dl)
            test_dls.append(test_dl)

        self.train_dls = train_dls
        self.test_dls = test_dls
        # self.few_shot_dl = few_shot_data_generator(self.test_dl)

    def create_save_dir(self):
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

    def calc_results_per_run(self):
        '''
        Calculates the acc, f1 and risk values for each cross-domain scenario
        '''
        # print(self.test_pred_labels[1][:20])
        # print(self.test_true_labels[1][:20])
        self.overall_rmse, self.overall_mape, self.overall_r2, task_RMSEs, task_MAPEs, task_R2s = _calc_metrics(
            self.test_pred_labels, self.test_true_labels, self.scenario_log_dir, self.home_path)

        # 定义的实验效果评价指标（记录每一次run的指标）
        self.metrics = {
            'Overall_rmse': [],
            'Overall_mape': [],
            'Overall_r2': [],
            'src_risk': [],
            'few_shot_trg_risk': [],
            'trg_risk': [],
            'dev_risk': []
        }
        # 添加每个任务的 accuracy 和 f1_score 存储结构
        num_tasks = len(self.test_pred_labels)  # 假设 self.test_pred_labels 的长度与任务数量一致
        for i in range(num_tasks):
            self.metrics[f'Task_{i}_rmse'] = []
            self.metrics[f'Task_{i}_mape'] = []
            self.metrics[f'Task_{i}_r2'] = []

        # if self.is_sweep:
        #     self.src_risk = calculate_risk(self.algorithm, self.src_test_dl, self.device)
        #     self.trg_risk = calculate_risk(self.algorithm, self.trg_test_dl, self.device)
        #     self.few_shot_trg_risk = calculate_risk(self.algorithm, self.few_shot_dl, self.device)
        #     self.dev_risk = calc_dev_risk(self.algorithm, self.src_train_dl, self.trg_train_dl, self.src_test_dl,
        #                                   self.dataset_configs, self.device)
        #
        #     run_metrics = {'accuracy': self.acc,
        #                    'f1_score': self.f1,
        #                    'src_risk': self.src_risk,
        #                    'few_shot_trg_risk': self.few_shot_trg_risk,
        #                    'trg_risk': self.trg_risk,
        #                    'dev_risk': self.dev_risk}
        #
        #     df = pd.DataFrame(columns=["acc", "f1", "src_risk", "few_shot_trg_risk", "trg_risk", "dev_risk"])
        #     df.loc[0] = [self.acc, self.f1, self.src_risk, self.few_shot_trg_risk, self.trg_risk,
        #                  self.dev_risk]
        # else:
        #     run_metrics = {'accuracy': self.acc, 'f1_score': self.f1}
        #     df = pd.DataFrame(columns=["acc", "f1"])
        #     df.loc[0] = [self.acc, self.f1]
        #
        # for (key, val) in run_metrics.items(): self.metrics[key].append(val) #把每一次run的结果存入self.metrics中
        #
        # scores_save_path = os.path.join(self.home_path, self.scenario_log_dir, "scores.xlsx")
        # df.to_excel(scores_save_path, index=False)
        # self.results_df = df

        # 创建一个 DataFrame 来存储每个任务的rmse mape r2
        df = pd.DataFrame(columns=["task", "acc", "f1"])
        df = pd.DataFrame(columns=["task", "rmse", "mape", "r2"])
        for i, (rmse, mape, r2) in enumerate(zip(task_RMSEs, task_MAPEs, task_R2s)):
            df.loc[i] = [f"Task_{i}", rmse, mape, r2]

        # 添加总体的到 DataFrame
        df.loc[len(task_MAPEs)] = ["Overall", self.overall_rmse, self.overall_mape, self.overall_r2]

        # 将每个任务及总体的rmse mape r2存入 self.metrics
        for i, (rmse, mape, r2) in enumerate(zip(task_RMSEs, task_MAPEs, task_R2s)):
            task_metrics = {'rmse': rmse, 'mape': mape, 'r2': r2}
            for key, val in task_metrics.items():
                self.metrics[f"Task_{i}_{key}"].append(val)

        # 也存储总体的指标
        overall_metrics = {'rmse': self.overall_rmse, 'mape': self.overall_mape, 'r2': self.overall_r2}
        for key, val in overall_metrics.items():
            self.metrics[f"Overall_{key}"].append(val)

        # 保存结果到 Excel 文件
        scores_save_path = os.path.join(self.home_path, self.scenario_log_dir, "scores.xlsx")
        df.to_excel(scores_save_path, index=False)
        self.results_df = df

    def calc_overall_results(self):
        exp = self.exp_log_dir

        # for exp in experiments:
        if self.is_sweep:
            results = pd.DataFrame(
                columns=["scenario", "rmse", "mape", "r2", "src_risk", "few_shot_trg_risk", "trg_risk", "dev_risk"])
        else:
            results = pd.DataFrame(columns=["run", "task", "rmse", "mape", "r2"])
        # 返回目录所有条目名称的列表
        runs_list = os.listdir(exp)
        runs_list = [i for i in runs_list if "_run_" in i]
        runs_list.sort()

        # unique_scenarios_names = [f'{i}_to_{j}' for i, j in self.dataset_configs.scenarios] #("2", "11"), ("7", "13"), ("12", "16"), ("9", "18"), ("6", "23")
        # # 将每个scenario的score dataframe合并到一个大的dataframe中
        # for run in runs_list:
        #     run_dir = os.path.join(exp, run)
        #     scores = pd.read_excel(os.path.join(run_dir, 'scores.xlsx'))
        #     scores.insert(0, 'run', '_'.join(run.split('_')[-2:])) #在第一列插入一个叫scenario的新列 值为类似2_to_11
        #     results = pd.concat([results, scores])
        for run in runs_list:
            run_dir = os.path.join(exp, run)
            scores = pd.read_excel(os.path.join(run_dir, 'scores.xlsx'))

            # 将 run 的标识插入到 DataFrame 中，并处理多任务结果
            for i, row in scores.iterrows():
                task_name = row["task"]  # 假设 'task' 列包含任务名称或 'Overall' 标识
                row_data = {"run": '_'.join(run.split('_')[-2:]), "task": task_name, "rmse": row["rmse"], "mape": row["mape"], "r2": row["r2"]}
                if self.is_sweep:
                    row_data.update({"src_risk": None, "few_shot_trg_risk": None, "trg_risk": None, "dev_risk": None})
                results = pd.concat([results, pd.DataFrame([row_data])])

        # print('results:')
        # print(results)  # results 类型object
        # print(results['task'])  # 'task' 列数据类型 object
        # 'rmse', 'mape', 'r2' 转换为数值类型的列
        numeric_columns = ['rmse', 'mape', 'r2']
        for col in numeric_columns:
            results[col] = pd.to_numeric(results[col], errors='coerce')
        # 计算平均值和标准差
        # avg_results = results.groupby("task").mean(numeric_only=True).reset_index()  # 按任务进行分组，计算均值
        # avg_results = results.groupby("task")[['rmse', 'mape', 'r2']].mean(numeric_only=True).reset_index()
        avg_results = results.groupby("task").mean(numeric_only=True).reset_index()
        # print('avg_results=', avg_results)
        # std_results = results.groupby("task").std(numeric_only=True).reset_index()  # 按任务进行分组，计算标准差
        std_results = results.groupby("task").std(numeric_only=True).reset_index()  # 按任务进行分组，计算标准差

        # 保存平均结果和标准差结果到 Excel 文件
        report_save_path_avg = os.path.join(exp, f"Average_results.xlsx")
        report_save_path_std = os.path.join(exp, f"std_results.xlsx")

        self.averages_results_df = pd.DataFrame(avg_results)
        self.std_results_df = pd.DataFrame(std_results)
        avg_results.to_excel(report_save_path_avg)
        std_results.to_excel(report_save_path_std)


parser = argparse.ArgumentParser()

# ========  Experiments Name ================
parser.add_argument('--save_dir',               default='experiments_logs',         type=str, help='Directory containing all experiments')
parser.add_argument('--experiment_description', default='HAR',               type=str, help='Name of your experiment (HAR, FD, HHAR_SA,EEG ')
parser.add_argument('--run_description',        default='UDA_KD_CNN',                     type=str, help='name of your runs, ')

# ========= Select the DA methods ============
parser.add_argument('--da_method',              default='UDA_KD',               type=str, help='UDA_KD')

# ========= Select the DATASET ==============
parser.add_argument('--data_path',              default=r'./data',                  type=str, help='Path containing dataset')
parser.add_argument('--dataset',                default='HAR',                      type=str, help='Dataset of choice: (HAR - WISDM - HHAR_SA - EEG)')

# ========= Select the BACKBONE ==============
parser.add_argument('--backbone',               default='CNN',                      type=str, help='Backbone of choice: (CNN - RESNET18 - TCN - RESNET34 -RESNET1D_WANG)')

# ========= Experiment settings ===============
parser.add_argument('--num_runs',               default=3,                          type=int, help='Number of consecutive run with different seeds')
parser.add_argument('--device',                 default='cuda:0',                   type=str, help='cpu or cuda')

# ======== sweep settings =====================
parser.add_argument('--is_sweep',               default=False,                      type=bool, help='singe run or sweep')
parser.add_argument('--num_sweeps',             default=20,                         type=str, help='Number of sweep runs')

# We run sweeps using wandb plateform, so next parameters are for wandb.
parser.add_argument('--sweep_project_wandb',    default='TEST_SOMETHING',       type=str, help='Project name in Wandb')
parser.add_argument('--wandb_entity',           type=str, help='Entity name in Wandb (can be left blank if there is a default entity)')
parser.add_argument('--hp_search_strategy',     default="random",               type=str, help='The way of selecting hyper-parameters (random-grid-bayes). in wandb see:https://docs.wandb.ai/guides/sweeps/configuration')
parser.add_argument('--metric_to_minimize',     default="src_risk",             type=str, help='select one of: (src_risk - trg_risk - few_shot_trg_risk - dev_risk)')

args = parser.parse_args()


if __name__ == "__main__":
    trainer = adv_cross_domain_kd_trainer(args)

    if args.is_sweep:
        trainer.sweep()
    else:
        trainer.train()
