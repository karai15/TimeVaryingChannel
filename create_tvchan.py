import numpy as np
import sys
import pickle
from src.TimeVaryingChannel.myplot import *  # plot
from src.TimeVaryingChannel.myfunc import *  # function
from src.TimeVaryingChannel.timeVaryChannel import *  # function
from src.TimeVaryingChannel.config import Param  # input param

# 正規化パラメータの出力
def print_param(param, n_fdTs, n_tauFs):
    # print
    print("\n #### 時間サンプリング ####")
    print(" シンボル間での最大位相回転量(正規化ドップラー): fd*Ts = ", param.fdTs_max[n_fdTs])  # サンプル間で最大周波数が何回転したか
    print(" 時間観測サンプル間での最大位相回転量: fd*Ts_obs = ", param.fdTsamp_max[n_fdTs])  # サンプル間で最大周波数が何回転したか
    print(" 観測シンボル全体での最大位相回転量: fd*Ts*Ns = ", param.fdTsamp_max[n_fdTs] * param.Ns_obs)  # 観測区間内で最大周波数が何回転したか

    print("\n #### 周波数サンプリング ####")
    print(" サブキャリア間での最大位相回転量(正規化遅延時間): tau*Fs = ", param.tauFs_max[n_tauFs])
    print(" 周波数観測サンプル間での最大位相回転量: tau*Fs_obs = ", param.tauFsamp_max[n_tauFs])
    print(" 観測帯域幅全体での最大位相回転量: tau*Bandwidth = ", param.tauFsamp_max[n_tauFs] * param.Nc_obs)
    print("\n")

# 正規化パラメータをシステムパラメータに変換して出力
def print_phy_param(fdTsamp_max, tauFsamp_max):

    # System param に変換
    fc = 28e9  # carrier frequency
    bandwidth = 400e6
    dt = 1 / bandwidth  # sampling interval
    df = 120e3  # scs
    Tg = (1 / df) * 0.072  # GI
    Ts = 1 / df + Tg  # Length of OFDM symbol
    wavelength = 3e8 / fc
    p_dt_smp = 50  # パイロット時間間隔
    p_df_smp = 10  # パイロット周波数間隔

    fd_max = fdTsamp_max / (Ts * p_dt_smp)  # 最大ドップラー
    tau_max = tauFsamp_max / (df * p_df_smp)  # 最大遅延時間
    velocity = 3.6 * fd_max * wavelength  # 移動速度 [km/h]

    print("\n #### パラメータ ####")
    print("サンプリング周期 1/B [us] :", dt * 1e6)
    print("シンボル長 Ts [us] :", Ts * 1e6)
    print("GI長 Tg [us] :", Tg * 1e6)
    print(" 最大遅延時間 [us] :", tau_max * 1e6)
    print(" 最大ドップラー [Hz] :", fd_max)
    print(" 移動速度 [km/h] :", velocity)

    test = 1


if __name__ == '__main__':

    """
    #### メモ ####
    # 時間変動は　観測サンプリング間の位相回転量fdTsamp_maxで調整可能
        # 観測区間を大きくしたい場合は, 観測数 Ns_obsを増やす.
        # 観測区間を固定して, 時間サンプリング周波数をk倍したいのであれば, 
          fdTsamp_maxを1/k倍, N_obsをk倍する (単にfdTsamp_maxを1/k倍だけではサンプル間隔が短くなる分,観測区間が短くなってしまう)
       
    # 周波数変動は 正規化遅延時間「tauFs_max」でを調整可能
        # 観測区間を大きくしたい場合はNcを大きくする (Nc_obsを大きくしても観測区間は固定でサンプリングが細かくなるだけ)
        # 周波数サンプリングを細かくしたい場合はNc_obsを大きくする
    """

    # np.random.seed(seed=32)  # seed

    ###################################
    # param
    filepath = "./save_data/output_tvchan.pickle"  # 保存先
    param = Param()  # 入力パラメータ取得
    N_fdTs = len(param.fdTs_max)
    N_tauFs = len(param.tauFs_max)

    print_phy_param(param.fdTsamp_max, param.tauFsamp_max)
    ###################################

    # 保存データ
    H_true_data = np.zeros((param.Nc, param.Ns, N_fdTs, N_tauFs, param.N_rand), dtype=complex)   # (周波数, 時間, ドップラー, 遅延時間)
    H_obs_data = np.zeros((param.Nc_obs, param.Ns_obs, N_fdTs, N_tauFs, param.N_rand), dtype=complex)   # (周波数, 時間, ドップラー, 遅延時間)

    for n_fdTs, fdTs_max in enumerate(param.fdTs_max):
        for n_tauFs, tauFs_max in enumerate(param.tauFs_max):
            print_param(param, n_fdTs, n_tauFs)
            for n_rand in range(param.N_rand):
                ###################################
                # Calc time varying channel
                H_true = tv_chan(param.Nc, param.Ns, param.Nl, param.decay_db, tauFs_max, fdTs_max)  # True channel
                noise = np.sqrt(db2real(-param.SNR)) * np.sqrt(1/2) * (np.random.normal(0, 1, (param.Nc_obs, param.Ns_obs)) + 1j * np.random.normal(0, 1, (param.Nc_obs, param.Ns_obs)) )
                H_obs = H_true[0:param.Nc:param.df_smp, 0:param.Ns:param.dt_smp] + noise  # sampling
                H_itpl = DftInterpolateCR(H_obs, param.Nc, param.Ns)  # interpolate
                ###################################

                ###################################
                # データ保存
                H_true_data[:, :, n_fdTs, n_tauFs, n_rand] = H_true[:, :]  # (周波数, 時間, ドップラー, 遅延時間)
                H_obs_data[:, :, n_fdTs, n_tauFs, n_rand] = H_obs[:, :]  # (周波数, 時間, ドップラー, 遅延時間)
                ###################################

            ###################################
            # # # plot
            # plot_channel(H_true, H_obs, H_itpl)  # plot
            ###################################

    ###################################
    # Save
    class SaveClass: pass  # データ保存用クラス作成
    saveInst = SaveClass()  # インスタンスの作成
    saveInst.param = param
    saveInst.H_true_data = H_true_data  # (周波数, 時間, ドップラー, 遅延時間)
    saveInst.H_obs_data = H_obs_data  # (周波数, 時間, ドップラー, 遅延時間)

    with open(filepath, "wb") as f:
        pickle.dump(saveInst, f)
    ###################################


    # ###################################
    # # System param に変換
    # fc = 28e9  # carrier frequency
    # bandwidth = 400e6
    # dt = 1 / bandwidth  # sampling interval
    # df = 120e3  # scs
    # Tg = (1 / df) * 0.072  # GI
    # Ts = 1 / df + Tg  # Length of OFDM symbol
    # wavelength = 3e8 / fc
    # p_dt_smp = 10  # パイロット時間間隔
    # p_df_smp = 100  # パイロット周波数間隔
    #
    # tau_max = tauFsamp_max / (df * p_dt_smp)  # 最大遅延時間
    # fd_max = fdTsamp_max / (Ts * p_df_smp)  # 最大ドップラー
    # velocity = 3.6 * fd_max * wavelength  # 移動速度 [km/h]
    #
    # print("\n #### パラメータ ####")
    # print(" 最大遅延時間[s] :", tau_max)
    # print(" 最大ドップラー[Hz] :", fd_max)
    # print(" 移動速度[km/h] :", velocity)
    # ###################################

