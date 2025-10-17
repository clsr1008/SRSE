def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

feature_dim = 4 #[16, 32,64]

class HAR():
    def __init__(self):
        super(HAR, self)
        # self.scenarios = [("2", "11"), ("7", "13"), ("12", "16"), ("9", "18"), ("6", "23")]
        self.scenarios = ["1", "3", "5"]
        # self.scenarios = [("18", "27"), ("20", "5"), ("24", "8"), ("28", "27"), ("30", "20")] # additional
        self.class_names = ['walk', 'upstairs', 'downstairs', 'sit', 'stand', 'lie']
        self.sequence_len = 128
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 9
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5
        self.num_classes = 6

        # CNN and RESNET features
        self.mid_channels = feature_dim #16 #32 #64
        self.final_out_channels =feature_dim * 2 #32 # 64 #128
        self.features_len = 1

        self.feature_dim = 9
        self.seq_length = 128
        self.num_layers = 3
        # self.dropout = 0.1
        self.interval = 8

        # Teacher model features
        self.mid_channels_t = 64
        self.final_out_channels_t =128
        self.features_len_t = 1

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128

class TJU():
    def __init__(self):
        super(TJU, self)
        # self.batch = ["NCA", "NCM", "NCM_NCA"]
        # self.batch = ["1", "2", "4"]  #discharge rate
        # self.batch = ["25", "35", "45"]  #temperature
        self.batch = ["1", "05", "025"]  #charge rate
        self.scenarios = list(self.batch)

        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 4
        self.kernel_size = 5
        self.stride = 1
        self.padding = 2

        # CNN and RESNET features
        self.mid_channels = 8
        self.mid_channels2 = 16
        self.mid_channels3 = 24
        self.mid_channels4 = 32
        self.final_out_channels = 4
        self.features_len = 1

        self.feature_dim = 4
        self.seq_length = 1000
        self.num_layers = 1
        self.dropout = 0.01
        self.interval = 1

        self.n_heads = 8
        self.d_model = 256
        self.d_ff = 512
        self.use_norm = True
        self.activation = 'relu'
        self.factor = 1
        self.output_attention = False

        # discriminator
        self.disc_hid_dim = 64

        # mlp
        self.mlp_num_layers = 3
        # DNN
        self.DNN_num_layers = 4
        # LSTM
        self.lstm_num_layers = 2

class XJTU():
    def __init__(self):
        super(XJTU, self)
        self.batch = ["Processed_Batch-1", "Processed_Batch-2", "Processed_Batch-3", "Processed_Batch-4"]
        self.scenarios = list(self.batch)

        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 4
        self.kernel_size = 5
        self.stride = 1
        self.padding = 2
        self.dropout = 0.5

        # CNN and RESNET features
        self.mid_channels = 8
        self.mid_channels2 = 16
        self.mid_channels3 = 24
        self.mid_channels4 = 16
        self.final_out_channels = 4
        self.features_len = 1

        self.feature_dim = 4
        self.seq_length = 1000
        self.num_layers = 3  # conv层只写了3，大于的话会报错none
        # self.dropout = 0.1
        self.interval = 1

        # discriminator
        self.disc_hid_dim = 64

class SANDIA():
    def __init__(self):
        super(SANDIA, self)
        self.batch = ["LFP-15C", "LFP-25C", "LFP-35C"]
        self.scenarios = list(self.batch)

        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 4
        self.kernel_size = 5
        self.stride = 1
        self.padding = 2
        self.dropout = 0.5

        # CNN and RESNET features
        self.mid_channels = 8
        self.mid_channels2 = 16
        self.mid_channels3 = 24
        self.mid_channels4 = 16
        self.final_out_channels = 4
        self.features_len = 1

        self.feature_dim = 4
        self.seq_length = 1000
        self.num_layers = 3  # conv层只写了3，大于的话会报错none
        # self.dropout = 0.1
        self.interval = 1

        # discriminator
        self.disc_hid_dim = 64


class EEG():
    def __init__(self):
        super(EEG, self).__init__()
        # data parameters
        self.num_classes = 5
        self.class_names = ['W', 'N1', 'N2', 'N3', 'REM']
        self.sequence_len = 3000
        self.scenarios = [("0", "11"), ("12", "5"), ("7", "18"), ("16", "1"), ("9", "14")]
        # self.scenarios = [("3", "19"), ("18", "12"), ("13", "17"), ("5", "15"), ("6", "2")] #additional
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 1
        self.kernel_size = 25
        self.stride = 6
        self.dropout = 0.2

        # features
        self.mid_channels = feature_dim
        self.final_out_channels = feature_dim*2
        self.features_len = 1

        # Teacher model features
        self.mid_channels_t = 64
        self.final_out_channels_t =128
        self.features_len_t = 1

        # TCN features
        self.tcn_layers = [32,64]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 15# 25
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500
        self.disc_hid_dim = 100


class HHAR_SA(object):  ## HHAR dataset, SAMSUNG device.
    def __init__(self):
        super(HHAR_SA, self).__init__()
        self.sequence_len = 128
        self.scenarios = [("2", "7"), ("0", "6"), ("1", "6"), ("3", "8"), ("4", "5")]
        # self.scenarios = [("5", "0"), ("6", "1"), ("7", "4"), ("8", "3"), ("0", "2")] # additional
        self.class_names = ['bike', 'sit', 'stand', 'walk', 'stairs_up', 'stairs_down']
        self.num_classes = 6
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 3
        self.kernel_size = 5
        self.stride = 1
        self.dropout = 0.5

        # features
        self.mid_channels =feature_dim
        self.final_out_channels =feature_dim *2
        self.features_len = 1

        # Teacher model features
        self.mid_channels_t = 64
        self.final_out_channels_t =128
        self.features_len_t = 1

        # TCN features
        self.tcn_layers = [75,150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500


class FD(object):
    def __init__(self):
        super(FD, self).__init__()
        self.sequence_len = 5120
        self.scenarios = [ ("0", "3"), ("0", "1"),("2", "1"), ("1", "2"),("2", "3")]
        # self.scenarios = [ ("1", "0"), ("1", "3"), ("3", "0"), ("3", "1"), ("3", "2")] #additional
        self.class_names = ['Healthy', 'D1', 'D2']
        self.num_classes = 3
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # Model configs
        self.input_channels = 1
        self.kernel_size = 32
        self.stride = 6
        self.dropout = 0.5

        # CNN and RESNET features
        self.mid_channels = feature_dim
        self.final_out_channels = feature_dim * 2
        self.features_len = 1

        # Teacher model features
        self.mid_channels_t = 64
        self.final_out_channels_t = 128
        self.features_len_t = 1

        # TCN features
        self.tcn_layers = [75, 150]
        self.tcn_final_out_channles = self.tcn_layers[-1]
        self.tcn_kernel_size = 17
        self.tcn_dropout = 0.0

        # lstm features
        self.lstm_hid = 128
        self.lstm_n_layers = 1
        self.lstm_bid = False

        # discriminator
        self.disc_hid_dim = 64
        self.DSKN_disc_hid = 128
        self.hidden_dim = 500