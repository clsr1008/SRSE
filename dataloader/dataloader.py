import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.utils.data import Dataset
from torchvision import transforms

from sklearn.model_selection import train_test_split

import os
import numpy as np
import random


class Load_Dataset(Dataset): #对数据集进行基本转换和归一化
    def __init__(self, dataset, normalize):
        super(Load_Dataset, self).__init__()

        # 提取循环数据
        cycle_keys = sorted(dataset.keys())
        num_cycles = len(cycle_keys)

        # 初始化 X 和 Y
        X_train = []
        y_train = []

        # 遍历每个循环，提取数据
        for cycle in cycle_keys:
            cccv_data = dataset[cycle]["CCCV"]
            X_train.append([
                cccv_data["time"],
                cccv_data["voltage"],
                cccv_data["current"],
                cccv_data["charge"]
            ])
            y_train.append(dataset[cycle]["SOH"])

        # 将 X 和 Y 转换为 NumPy 数组并调整形状
        X_train = np.array(X_train)  # 形状为 (num_cycles, 4, 1000)
        y_train = np.array(y_train)  # 形状为 (num_cycles,)

        X_train = torch.from_numpy(X_train) #转换成张量
        y_train = torch.from_numpy(y_train)

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2) #升维

        if X_train.shape.index(min(X_train.shape[1], X_train.shape[2])) != 1:  # make sure the Channels in second dim（一般情况下不会执行）
            X_train = X_train.permute(0, 2, 1)

        self.x_data = X_train
        self.y_data = y_train

        self.num_channels = X_train.shape[1]

        if normalize:
            # Assume datashape: num_samples, num_channels, seq_length
            data_mean = torch.FloatTensor(self.num_channels).fill_(0).tolist()  # assume min= number of channels
            data_std = torch.FloatTensor(self.num_channels).fill_(1).tolist()  # assume min= number of channels
            data_transform = transforms.Normalize(mean=data_mean, std=data_std)
            self.transform = data_transform
        else:
            self.transform = None

        self.len = X_train.shape[0]

    def __getitem__(self, index):
        if self.transform is not None:
            output = self.transform(self.x_data[index].view(self.num_channels, -1, 1))
            self.x_data[index] = output.view(self.x_data[index].shape)

        return self.x_data[index].float(), self.y_data[index]

    def __len__(self):
        return self.len


def data_generator(data_path, scenario, dataset_configs, hparams):
    batch = scenario
    # print(data_path)
    pkl_files = []
    # batch_path = os.path.join(data_path, batch)
    # 遍历 batch 路径下的所有 .pkl 文件
    # for filename in os.listdir(batch_path):
    #     if filename.endswith(".pkl"):
    #         file_path = os.path.join(batch_path, filename)
    #         print(file_path)
    #         # 加载 .pkl 文件并将内容添加到 train_dataset 列表
    #         pkl_file = joblib.load(file_path)
    #         pkl_files.append(pkl_file)

    # data_path = os.path.join(data_path, "NCM_NCA")
    # print(data_path)
    # for filename in os.listdir(data_path):
    #     if filename.endswith(".pkl"):
    #         filename_no_ext = filename[:-4]  # 去掉 ".pkl"
    #         parts = filename_no_ext.split("_")  # 假设文件名是 CY25-05_1-#1.pkl 这种格式
    #
    #         if len(parts) < 2:
    #             continue  # 确保文件名格式正确
    #
    #         num_parts = parts[-1].split("-")  # 处理 "1-#1"
    #
    #         if len(num_parts) < 2:
    #             continue  # 确保格式正确
    #
    #         discharge_rate = num_parts[0]  # 取倒数第二个数字
    #
    #         if discharge_rate in batch:
    #             file_path = os.path.join(data_path, filename)
    #             print(f"Loading: {file_path}")
    #             pkl_file = joblib.load(file_path)
    #             pkl_files.append(pkl_file)

    # data_path = os.path.join(data_path, "NCM")
    # print(data_path)
    # for filename in os.listdir(data_path):
    #     if filename.endswith(".pkl"):
    #         filename_no_ext = filename[:-4]  # 去掉 ".pkl"
    #         parts = filename_no_ext.split("_")  # 解析 "CY25-05_1-#1.pkl"
    #
    #         if len(parts) < 2:
    #             continue  # 确保文件名格式正确
    #
    #         # 解析温度
    #         prefix_parts = parts[0].split("-")  # 处理 "CY25-05"
    #         if len(prefix_parts) < 2:
    #             continue  # 确保格式正确
    #
    #         temperature = prefix_parts[0][2:]  # 提取 "CY25-05" 中的 "25"
    #
    #         if temperature in batch:
    #             file_path = os.path.join(data_path, filename)
    #             print(f"Loading: {file_path}")
    #             pkl_file = joblib.load(file_path)
    #             pkl_files.append(pkl_file)

    data_path = os.path.join(data_path, "NCA")
    print(data_path)
    for filename in os.listdir(data_path):
        if filename.endswith(".pkl") and filename.startswith("CY25"):  # 只处理 CY25 开头的文件
            filename_no_ext = filename[:-4]  # 去掉 ".pkl"
            parts = filename_no_ext.split("_")  # 解析 "CY25-05_1-#1.pkl"

            if len(parts) < 2:
                continue  # 确保文件名格式正确

            # 解析充电速率
            prefix_parts = parts[0].split("-")  # 处理 "CY25-05"
            if len(prefix_parts) < 2:
                continue  # 确保格式正确

            charge_rate = prefix_parts[1]  # 提取 "CY25-05" 中的 "05"

            if charge_rate in batch:
                file_path = os.path.join(data_path, filename)
                print(f"Loading: {file_path}")
                pkl_file = joblib.load(file_path)
                pkl_files.append(pkl_file)

    # Loading datasets（对数据集进行归一化操作）
    origin_datasets = [Load_Dataset(file, dataset_configs.normalize) for file in pkl_files]
    # 提取并拼接所有 origin_dataset 的 x_data 和 y_data
    x_data = torch.cat([dataset.x_data for dataset in origin_datasets], dim=0)
    y_data = torch.cat([dataset.y_data for dataset in origin_datasets], dim=0)

    if dataset_configs.normalize:
        for c in range(x_data.size(1)):  # 对每个通道独立归一化
            x_data[:, c, :] = (x_data[:, c, :] - x_data[:, c, :].mean()) / (x_data[:, c, :].max() - x_data[:, c, :].mean())

    # 将拼接后的数据构建为 total_dataset
    total_dataset = TensorDataset(x_data, y_data)

    # 计算训练集和测试集的大小
    train_size = int(0.8 * len(total_dataset))
    test_size = len(total_dataset) - train_size

    # 使用 random_split 将 total_dataset 划分为训练集和测试集
    train_dataset, test_dataset = random_split(total_dataset, [train_size, test_size])

    # Dataloaders（把数据集划分为batch，返回迭代器）
    batch_size = hparams["batch_size"]
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=True, num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=dataset_configs.drop_last, num_workers=0)
    return train_loader, test_loader


def few_shot_data_generator(data_loader):
    x_data = data_loader.dataset.x_data
    y_data = data_loader.dataset.y_data
    if not isinstance(y_data, (np.ndarray)):
        y_data = y_data.numpy()

    NUM_SAMPLES_PER_CLASS = 5
    NUM_CLASSES = len(np.unique(y_data))

    samples_count_dict = {id: 0 for id in range(NUM_CLASSES)}

    # if the min number of samples in one class is less than NUM_SAMPLES_PER_CLASS
    y_list = y_data.tolist()
    counts = [y_list.count(i) for i in range(NUM_CLASSES)]

    for i in samples_count_dict:
        if counts[i] < NUM_SAMPLES_PER_CLASS:
            samples_count_dict[i] = counts[i]
        else:
            samples_count_dict[i] = NUM_SAMPLES_PER_CLASS

    # if min(counts) < NUM_SAMPLES_PER_CLASS:
    #     NUM_SAMPLES_PER_CLASS = min(counts)

    samples_ids = {}
    for i in range(NUM_CLASSES):
        samples_ids[i] = [np.where(y_data == i)[0]][0]

    selected_ids = {}
    for i in range(NUM_CLASSES):
        selected_ids[i] = random.sample(list(samples_ids[i]), samples_count_dict[i])

    # select the samples according to the selected random ids
    y = torch.from_numpy(y_data)
    selected_x = x_data[list(selected_ids[0])]
    selected_y = y[list(selected_ids[0])]

    for i in range(1, NUM_CLASSES):
        selected_x = torch.cat((selected_x, x_data[list(selected_ids[i])]), dim=0)
        selected_y = torch.cat((selected_y, y[list(selected_ids[i])]), dim=0)

    few_shot_dataset = {"samples": selected_x, "labels": selected_y}
    # Loading datasets
    few_shot_dataset = Load_Dataset(few_shot_dataset, None)

    # Dataloaders
    few_shot_loader = torch.utils.data.DataLoader(dataset=few_shot_dataset, batch_size=len(few_shot_dataset),
                                                  shuffle=False, drop_last=False, num_workers=0)
    return few_shot_loader


def generator_percentage_of_data(data_loader):
    x_data = data_loader.dataset.x_data
    y_data = data_loader.dataset.y_data

    X_train, X_val, y_train, y_val = train_test_split(x_data, y_data, test_size=0.1, random_state=0)

    few_shot_dataset = {"samples": X_val, "labels": y_val}
    # Loading datasets
    few_shot_dataset = Load_Dataset(few_shot_dataset, None)

    few_shot_loader = torch.utils.data.DataLoader(dataset=few_shot_dataset, batch_size=32,
                                                  shuffle=True, drop_last=True, num_workers=0)
    return few_shot_loader
