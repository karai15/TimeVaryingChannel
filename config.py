import numpy as np

class Param:
    def __init__(self):

        # 時間(シンボル)サンプリング
        self.Ns = 100  # シンボル数
        self.Ns_obs = 25  # 観測数
        self.fdTsamp_max = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # 1観測サンプル区間での最大位相回転量 (1/2未満であればエイリアシング発生しない)
        # self.fdTsamp_max = np.array([0.4])  # 1観測サンプル区間での最大位相回転量 (1/2未満であればエイリアシング発生しない)
        self.dt_smp = int(np.ceil(self.Ns / self.Ns_obs))  # 観測間隔
        self.fdTs_max = self.fdTsamp_max / self.dt_smp  # 正規化ドップラー ((真の)1サンプル区間での最大位相回転量)

        # 周波数サンプリング
        self.Nc = 100  # サブキャリア数
        self.Nc_obs = 25  # 観測数
        self.tauFsamp_max = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # 1観測サンプル区間での最大位相回転量 (1/2未満であればエイリアシング発生しない)
        # self.tauFsamp_max = np.array([0.4])  # 1観測サンプル区間での最大位相回転量 (1/2未満であればエイリアシング発生しない)
        self.df_smp = int(np.ceil(self.Nc / self.Nc_obs))
        self.tauFs_max = self.tauFsamp_max / self.df_smp  # ((真の)1サンプル区間での最大位相回転量) (最大遅延/シンボル長 (GI長より 8%未満が妥当))

        # 物理パラメータ
        self.Nl = 16  # Number of delay path
        self.SNR = 20  # 1/(noise variance) [dB]
        self.decay_db = 0.4  # decay factor of delay path [dB]

        self.N_rand = 10  # ランダムサンプル数 # 最終的に (ドップラ, 遅延, ランダム)だけデータが作成される
