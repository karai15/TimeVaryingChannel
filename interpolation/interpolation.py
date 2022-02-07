import os
import pickle
import numpy as np
from src.TimeVaryingChannel.interpolation.func_interpolation import *
from src.TimeVaryingChannel.interpolation.plot_config import plot_config


# データ読み込み
def load_data(option_abs, filepath):

    saveInst = SaveClass()  # インスタンス化
    with open(filepath, "rb") as f:
        saveInst = pickle.load(f)

    # 保存データ
    H_true_data = saveInst.H_true_data  # (周波数, 時間, ドップラー, 遅延時間)
    H_obs_data = saveInst.H_obs_data  # (周波数, 時間, ドップラー, 遅延時間)
    param = saveInst.param

    # 絶対値で出力する場合
    if option_abs:
        H_true_data = np.abs(H_true_data)
        H_obs_data = np.abs(H_obs_data)

    return H_true_data, H_obs_data, param

# H_obsの欠損部分をnanでパディング
def nan_padding(H_obs, Nc, Ns):
    # パラメータ取得
    Nc_obs = H_obs.shape[0]  # サブキャリア数
    Ns_obs = H_obs.shape[1]  # シンボル数
    r_f = int(Nc / Nc_obs)  # サンプリングレート(周波数)
    r_t = int(Ns / Ns_obs)  # サンプリングレート(時間)

    # H_obsの未観測点にnanを埋める
    H_obs_nan = np.zeros((Nc, Ns), dtype=H_obs.dtype)
    H_obs_nan[:, :] = np.nan
    H_obs_nan[0:Nc:r_f, 0:Ns:r_t] = H_obs

    return H_obs_nan

# チャネル情報をGPが扱える形に変形
def trans_data_gp(H_true, H_obs):
    Nx = H_true.shape[0]
    Ny = H_true.shape[1]
    H_obs_nan = nan_padding(H_obs, Nx, Ny)

    # 観測
    Xa_obs = np.array([])
    Ya_obs = np.array([])
    Za_obs = np.array([])
    # 予測
    Xa_pred = np.array([])
    Ya_pred = np.array([])
    for x in range(Nx):
        for y in range(Ny):
            Xa_pred = np.append(Xa_pred, x)
            Ya_pred = np.append(Ya_pred, y)

            if np.isnan(H_obs_nan[x, y]) == False:
                Xa_obs = np.append(Xa_obs, x)
                Ya_obs = np.append(Ya_obs, y)
                Za_obs = np.append(Za_obs, H_obs_nan[x, y])

    return Xa_obs, Ya_obs, Za_obs, Xa_pred, Ya_pred

# チャネルplot
def save_plt_channel(fdTs, tauFs, H_true, H_obs, H_itpl, method_list):

    # param
    Nc = H_true.shape[0]
    Ns = H_true.shape[1]
    Nc_obs = H_obs.shape[0]
    Ns_obs = H_obs.shape[1]
    N_method = len(method_list)  # N_method >= 2 を想定
    df_smp = int(np.ceil(Nc / Nc_obs))
    dt_smp = int(np.ceil(Ns / Ns_obs))

    n_time = 0
    n_freq = 0

    # plot
    pp = plot_config()
    linewidth = 2.5
    leg_size = 15
    markeredgewidth = 2
    markersize = 8
    label_ftz = 30
    fig, ax = plt.subplots(2, 1, squeeze=False, figsize=(15, 6), tight_layout=True)

    #############################################
    # h vs freq
    t = 1 / Nc * np.arange(Nc)
    t_obs = 1 / Nc * dt_smp * np.arange(Nc_obs)

    ax[0, 0].plot(t, np.abs(H_true[:, n_time]),
                  label="true", color="black", linestyle="solid", alpha=pp["alpha"], linewidth=linewidth)
    ax[0, 0].plot(t_obs, np.abs(H_obs[:, n_time]),
                  label="observed", color="black", linestyle="None", marker="o", markerfacecolor="None",
                  markeredgewidth=markeredgewidth, markersize=markersize)
    for n_method, method in enumerate(method_list):
        ax[0, 0].plot(t, np.abs(H_itpl[n_method, :, n_time]),
                      label=method, color=pp["color_list"][n_method],
                      linestyle=pp["line_list"][n_method+1], linewidth=linewidth)
    ax[0, 0].set_xlabel("frequency", fontsize=label_ftz)
    ax[0, 0].set_ylabel("channel", fontsize=label_ftz)
    ax[0, 0].grid()
    ax[0, 0].legend(fontsize=leg_size)
    #############################################

    #############################################
    # h vs time
    t = 1 / Ns * np.arange(Nc)
    t_obs = 1 / Ns * df_smp * np.arange(Nc_obs)

    ax[1, 0].plot(t, np.abs(H_true[n_freq, :]),
                  label="true", color="black", linestyle="solid", alpha=pp["alpha"], linewidth=2.5)
    ax[1, 0].plot(t_obs, np.abs(H_obs[n_freq, :]),
                  label="observed", color="black", linestyle="None", marker="o", markerfacecolor="None",
                  markeredgewidth=2, markersize=8)
    for n_method, method in enumerate(method_list):
        ax[1, 0].plot(t, np.abs(H_itpl[n_method, n_freq, :]),
                      label=method, color=pp["color_list"][n_method],
                      linestyle=pp["line_list"][n_method + 1], linewidth=2.5)
    ax[1, 0].set_xlabel("time", fontsize=label_ftz)
    ax[1, 0].set_ylabel("channel", fontsize=label_ftz)
    ax[1, 0].grid()
    ax[1, 0].legend(fontsize=leg_size)
    #############################################
    plt.suptitle("$f_dT_s$ = " + str(fdTs) + ", " + r"$\tau F_s$ = " + str(tauFs))  # title
    # plt.show()

    # save img
    FpSaveImg = "../save_data/img/"
    filename = "1d_tvchan_" + "fdTs_" + str(fdTs) + "_tauFs_" + str(tauFs) + ".png"
    fig.savefig(FpSaveImg + filename)

# 画像の保存
def save_img(fdTs, tauFs, H_true, H_obs, H_itpl, method_list):

    N_method = len(method_list)  # N_method >= 2 を想定
    pp = plot_config()
    title_ftz = 30

    fig, ax = plt.subplots(2, len(method_list), squeeze=False, figsize=(15, 8))
    # True, Obs画像
    ax[0, 0].imshow(np.abs(H_true), cmap="jet")
    ax[0, 0].set_title("true", fontsize=title_ftz)
    ax[0, 0].axes.xaxis.set_visible(False)
    ax[0, 0].axes.yaxis.set_visible(False)
    # 補間画像
    ax[0, 1].imshow(np.abs(H_obs), cmap="jet")
    ax[0, 1].set_title("observed", fontsize=title_ftz)
    ax[0, 1].axes.xaxis.set_visible(False)
    ax[0, 1].axes.yaxis.set_visible(False)

    for n in range(N_method-2):
        fig.delaxes(ax[0, 2+n])

    for n_method, method in enumerate(method_list):
        ax[1, n_method].imshow(np.abs(H_itpl[n_method, :, :]), cmap="jet")  # (手法, 周波数, 時間)
        ax[1, n_method].axes.xaxis.set_visible(False)
        ax[1, n_method].axes.yaxis.set_visible(False)
        ax[1, n_method].set_title(method, fontsize=title_ftz)

    plt.suptitle("$f_dT_s$ = " + str(fdTs) + ", " + r"$\tau F_s$ = " + str(tauFs))  # title
    # plt.show()

    # save img
    FpSaveImg = "../save_data/img/"
    filename = "2d_tvchan_" + "fdTs_" + str(fdTs) + "_tauFs_" + str(tauFs) + ".png"
    fig.savefig(FpSaveImg + filename)

# MSEの保存
def save_plot_mse(MSE_data, fdTsamp_max_list, tauFsamp_max_list, method_list):

    # パラメータの取得
    pp = plot_config()
    N_fdTs = len(fdTsamp_max_list)
    N_tauFs = len(tauFsamp_max_list)
    # MSE_data # (補間手法, ドップラ, 遅延時間)

    # MSE vs fdTs
    fig_fdTs, ax_fdTs = plt.subplots(1, N_tauFs, squeeze=False, figsize=(15, 3))
    for n_tauFs, tauFsamp_max in enumerate(tauFsamp_max_list):
        for n_method, method in enumerate(method_list):

            if method == "linear":  # 双線形補間をスキップ
                ax_fdTs[0, n_tauFs].grid()
                continue

            MSE_fdTs = MSE_data[n_method, :, n_tauFs]  # (補間手法, ドップラ, 遅延時間)
            ax_fdTs[0, n_tauFs].plot(fdTsamp_max_list, MSE_fdTs,
                          color=pp["color_list"][n_method], linestyle=pp["line_list"][n_method],
                          marker=pp["mark_list"][n_method], markersize=pp["marker_size"],
                          markerfacecolor=pp["mark_face_color"], markeredgewidth=pp["mark_edge_width"],
                          label="$"+method+"$")
            ax_fdTs[0, n_tauFs].set_xlabel("$f_dT_s$")
            ax_fdTs[0, n_tauFs].set_ylabel("MSE")
            ax_fdTs[0, n_tauFs].set_title(r"$\tau F_s$"+"="+str(tauFsamp_max))
            ax_fdTs[0, n_tauFs].grid()
            ax_fdTs[0, n_tauFs].legend(fontsize=pp["leg_ftz"])

    # MSE vs tauFs
    fig_tauFs, ax_tauFs = plt.subplots(1, N_fdTs, squeeze=False, figsize=(15, 3))
    for n_fdTs, fdTsamp_max in enumerate(fdTsamp_max_list):
        for n_method, method in enumerate(method_list):

            if method == "linear":  # 双線形補間をスキップ
                ax_tauFs[0, n_fdTs].grid()
                continue

            MSE_tauFs = MSE_data[n_method, n_fdTs, :]  # (補間手法, ドップラ, 遅延時間)
            ax_tauFs[0, n_fdTs].plot(fdTsamp_max_list, MSE_tauFs,
                          color=pp["color_list"][n_method], linestyle=pp["line_list"][n_method],
                          marker=pp["mark_list"][n_method], markersize=pp["marker_size"],
                          markerfacecolor=pp["mark_face_color"], markeredgewidth=pp["mark_edge_width"],
                          label="$"+method+"$")
            ax_tauFs[0, n_fdTs].set_xlabel(r"$\tau F_s$")
            ax_tauFs[0, n_fdTs].set_ylabel("MSE")
            ax_tauFs[0, n_fdTs].set_title("$f_dT_s$"+"="+str(fdTsamp_max))
            ax_tauFs[0, n_fdTs].grid()
            ax_tauFs[0, n_fdTs].legend(fontsize=pp["leg_ftz"])

    # save img
    FpSaveImg = "../save_data/img/"
    filename_fdTs = "MSE_vs_fdTs" + ".png"
    filename_tauFs = "MSE_vs_tauFs" + ".png"
    fig_fdTs.savefig(FpSaveImg + filename_fdTs)
    fig_tauFs.savefig(FpSaveImg + filename_tauFs)

# 補間
def interpolation(H_true, H_obs, method_list):

    # method_list = ["dft","linear", "cubic", "quintic", "gp"]
    H_itpl_list = []
    MSE_list = []
    for method in method_list:

        # DFT補間
        if method == "dft":
            H_itpl_dft, MSE_dft = dft_interpolation(H_true, H_obs)
            H_itpl_list.append(H_itpl_dft)
            MSE_list.append(MSE_dft)

        # 多項式補間
        if method == "linear" or method == "cubic" or method == "quintic":
            # kind_list = ["linear", "cubic", "quintic"]  # 補間手法 ("linear", "cubic", "quintic")
            kind_list = [method]  # 補間手法 ("linear", "cubic", "quintic")
            H_itpl_poly_list, MSE_poly_list = polynomial_interpolation(H_true, H_obs, kind_list)
            H_itpl_list.append(H_itpl_poly_list[0])
            MSE_list.append(MSE_poly_list[0])

        # Gaussian Process
        if method == "gp":
            max_iters = 1000
            Xa_obs, Ya_obs, Za_obs, Xa_pred, Ya_pred = trans_data_gp(H_true, H_obs)
            H_itpl_gp, MSE_gp = gp_interpolation(Xa_obs, Ya_obs, Za_obs, Xa_pred, Ya_pred, H_true, max_iters)

            #############################
            # H_itpl_gp = H_itpl_dft  # 後で消す
            # MSE_gp = MSE_dft
            #############################

            H_itpl_list.append(H_itpl_gp)
            MSE_list.append(MSE_gp)

    return H_itpl_list, MSE_list

# main
if __name__ == '__main__':

    # ファイルパス
    filepath_input = "../save_data/output_tvchan.pickle"
    filepath_out = "../save_data/output_mse_rbf_ARD_noiseFix_sn20.pickle"
    class SaveClass: pass  # 保存データクラス
    method_list = ["dft", "linear", "gp"]  # 補間手法 ["dft","linear", "cubic", "quintic", "gp"]

    # 既にデータが保存されている場合
    if os.path.exists(filepath_out):
        saveInst = SaveClass()  # インスタンス化
        with open(filepath_out, "rb") as f:
            saveInst = pickle.load(f)

        H_true_data = saveInst.H_true_data
        H_obs_data = saveInst.H_obs_data
        MSE_data = saveInst.MSE_data
        H_itpl_data = saveInst.H_itpl_data
        param = saveInst.param
        method_list = saveInst.method_list

        #####################
        # # test
        # MSE_data[2, 4, 4, 1] = 0.01
        test = MSE_data[:, 3, 1, :]
        test[1, 4] = 0.01
        test_avg = np.average(test, axis=1)
        aaa = 1
        #####################

    # データが保存されていない場合は補間手法を実行
    else:
        option_abs = True  # absをとるかどうか
        H_true_data, H_obs_data, param = load_data(option_abs, filepath_input)  # (周波数, 時間, ドップラー, 遅延時間, ランダム)
        MSE_data = np.zeros((len(method_list), len(param.fdTsamp_max), len(param.tauFsamp_max), param.N_rand))  # (補間手法, ドップラ, 遅延時間, ランダム)
        H_itpl_data = np.zeros((len(method_list), len(param.fdTsamp_max), len(param.tauFsamp_max), param.N_rand, H_true_data.shape[0], H_true_data.shape[1]))  # (補間手法, ドップラ, 遅延時間, ランダム, 周波数, 時間)

    # ドップラ, 遅延時間のループ
    for n_fdTs, fdTsamp_max in enumerate(param.fdTsamp_max):
        for n_tauFs, tauFsamp_max in enumerate(param.tauFsamp_max):
            for n_rand in range(param.N_rand):
                # データ取得
                H_true = H_true_data[:, :, n_fdTs, n_tauFs, n_rand]  # (周波数, 時間, ドップラー, 遅延時間)
                H_obs = H_obs_data[:, :, n_fdTs, n_tauFs, n_rand]

                # 補間
                if os.path.exists(filepath_out) == False:  # 既存データがある場合はスキップ
                    # 補間
                    H_itpl_list, MSE_list = interpolation(H_true, H_obs, method_list)
                    # データ回収
                    MSE_data[:, n_fdTs, n_tauFs, n_rand] = np.array([MSE_list])[:] # (補間手法, ドップラ, 遅延時間, ランダム)
                    for n_method, H_itpl in enumerate(H_itpl_list):
                        H_itpl_data[n_method, n_fdTs, n_tauFs, n_rand, :, :] = H_itpl[:, :]  # (補間手法, ドップラ, ランダム, 遅延時間, 周波数, 時間)


                ##############
                # if n_fdTs == 4 and n_tauFs
                ##############

            # 評価
            print("\n (正規化ドップラ, 正規化遅延時間) = ", fdTsamp_max, ",", tauFsamp_max)
            for n_method, method in enumerate(method_list):
                print(" MSE_" + method + ":", np.average(MSE_data[n_method, n_fdTs, n_tauFs, :]) )

            # 画像の保存
            save_img(fdTsamp_max, tauFsamp_max, H_true, H_obs, H_itpl_data[:, n_fdTs, n_tauFs, n_rand, :, :], method_list)
            save_plt_channel(fdTsamp_max, tauFsamp_max, H_true, H_obs, H_itpl_data[:, n_fdTs, n_tauFs, n_rand, :, :], method_list)


    #################################################
    # データ保存
    if os.path.exists(filepath_out) == False:  # 保存ファイルがない場合に保存
        # class SaveClass: pass  # データ保存用クラス作成
        saveInst = SaveClass()  # インスタンスの作成
        saveInst.param = param
        saveInst.method_list = method_list
        saveInst.H_true_data = H_true_data  # (周波数, 時間, ドップラー, 遅延時間, ランダム)
        saveInst.H_obs_data = H_obs_data  # (周波数, 時間, ドップラー, 遅延時間, ランダム)
        saveInst.H_itpl_data = H_itpl_data  # (補間手法, ドップラ, 遅延時間, ランダム, 周波数, 時間)
        saveInst.MSE_data = MSE_data  # (補間手法, ドップラ, 遅延時間, ランダム)

        with open(filepath_out, "wb") as f:
            pickle.dump(saveInst, f)
    #################################################

    # MSE plot の保存
    MSE_data = np.average(MSE_data, axis=3)  # ランダム方向で平均化 (補間手法, ドップラ, 遅延時間)
    save_plot_mse(MSE_data, param.fdTsamp_max, param.tauFsamp_max, method_list)
    plt.show()


    """
    ・パワポ作成
    ・2次元カーネル
    ・チャネル誤差からチャネル容量の計算
    ・パラメータ換算
    ・カーネルの式入れてもいいかも
    ・fdTsが0.5以上の時に性能が逆転しそう

    #############################################################
    ガウス過程について
    https://statmodeling.hatenablog.com/entry/how-to-use-GPy
    
    kernel = GPy.kern.RBF(input_dim=1, variance=1, lengthscale=0.2)  # lengthscaleでRBFカーネルの動径半径を決定？
    kernel = GPy.kern.Matern32(input_dim=2, ARD=True) # ARD=Trueは入力の次元1つに対し、1つのlengthscaleパラメータを割り振ること（すなわちGPは等方でないことを表す
    
    # カーネルの結合
    kernel = GPy.kern.RBF(input_dim, ARD=True) + GPy.kern.Bias(input_dim) + GPy.kern.Linear(input_dim) + GPy.kern.White(input_dim)
    
    # GPLVM    
    model = GPy.models.BayesianGPLVM(X, input_dim, kernel=kernel, num_inducing=30)  # num_inducing が中間変数?
    
    white(variance=1.0)

    #############################################################

    """

