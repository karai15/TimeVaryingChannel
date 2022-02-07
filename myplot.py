import numpy as np
import matplotlib.pyplot as plt
from src.TimeVaryingChannel.timeVaryChannel import *  # function

def plot_config():
    """
    グラフの設定
    参考サイト
    https://qiita.com/MENDY/items/fe9b0c50383d8b2fd919
    https://qiita.com/M_Kumagai/items/b11de7c9d06b3c43431d
    https://qiita.com/qsnsr123/items/325d21621cfe9e553c17
    https://minus9d.hatenablog.com/entry/2016/04/21/215532
    """

    fsz = 15  # フォントサイズ
    plt.rcParams['font.family'] = 'Times New Roman'  # font familyの設定 'sans-serif':ゴシック
    plt.rcParams['mathtext.fontset'] = 'stix'  # math fontの設定
    plt.rcParams["font.size"] = fsz  # 全体のフォントサイズが変更されます。
    plt.rcParams['axes.linewidth'] = 1.0  # グラフの枠線の太さ

    # 軸
    plt.rcParams['xtick.labelsize'] = fsz  # x軸のフォントサイズ
    plt.rcParams['ytick.labelsize'] = fsz  # y軸のフォントサイズ
    plt.rcParams['xtick.direction'] = 'in'  # x軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    plt.rcParams['ytick.direction'] = 'in'  # y軸の目盛線が内向き('in')か外向き('out')か双方向か('inout')
    plt.rcParams['xtick.major.width'] = 1.2  # x軸目盛の太さ
    plt.rcParams['ytick.major.width'] = 1.2  # y軸目盛の太さ
    # plt.rcParams['axes.grid']=True
    plt.rcParams['grid.linestyle'] = '--'
    plt.rcParams['grid.linewidth'] = 0.8  # gridの太さ

    # legend
    plt.rcParams["legend.fancybox"] = False  # 丸角
    plt.rcParams["legend.framealpha"] = 1  # 透明度の指定、0で塗りつぶしなし
    plt.rcParams["legend.edgecolor"] = 'black'  # edgeの色を変更

    # 図のサイズ
    fig_ftz = 12
    # plt.rcParams["figure.figsize"] = [1.4 * fig_ftz, 1 * fig_ftz]  # 図の縦横のサイズ([横(inch),縦(inch)])
    # # plt.rcParams["figure.dpi"] = 150            # dpi(dots per inch)
    # plt.rcParams["figure.autolayout"] = False  # レイアウトの自動調整を利用するかどうか
    # plt.rcParams["figure.subplot.left"] = 0.14  # 余白
    # plt.rcParams["figure.subplot.bottom"] = 0.1  # 余白
    # plt.rcParams["figure.subplot.right"] = 0.90  # 余白
    # plt.rcParams["figure.subplot.top"] = 0.95  # 余白
    # plt.rcParams["figure.subplot.wspace"] = 0.20  # 図が複数枚ある時の左右との余白
    # plt.rcParams["figure.subplot.hspace"] = 0.20  # 図が複数枚ある時の上下との余白

    plt.rcParams["figure.autolayout"] = True  # レイアウトの自動調整

    # plot param
    plot_param = {
        "color_list": ["g", "r", "b", "k", "y", "m", "c"],
        "mark_list": ['s', '^', 'o', 'D', '*', 'p', 'v', 'x'],
        "line_list": ["solid", "dashdot", "dotted", "dashed"],
        "leg_ftz": 10.5,  # 凡例フォントサイズ
        "alpha": 0.8,  # 透明度
        "mark_face_color": "None",  # marker内の色
        "marker_size": 8,
        "mark_edge_width": 1.0
    }

    return plot_param

# plot CFR(Channel Frequency Response) and CIR (Channel Impulse Response)
def plotTvCR(h_true, hi_true, h_obs, hi_obs, h_itpl, hi_itpl, title, TvFlag):
    """
    :param h_true: CFR (Nc, 1)
    :param hi_true: CIR (Nc, 1)
    :param h_obs: Observed CFR (Nc_obs, 1)
    :param hi_obs: Observed CIR (Nc_obs, 1)
    :param h_itpl: Interpolated CFR (Nc_obs, 1)
    :param hi_itpl: Interpolated CIR (Nc_obs, 1)
    :param title: Plot title
    :param TvFlag: if Time Varying TvFlag=1
    """
    Nc = h_true.shape[0]  # Num of samples
    Nc_obs = h_obs.shape[0]  # Num of sparse samples
    d_smp = np.ceil(Nc / Nc_obs).astype(int)

    T_true = Nc
    T_obs = d_smp * Nc_obs

    x = 1 / T_true * np.arange(Nc)
    x_itpl = 1 / T_true * T_obs / T_true * np.arange(Nc)
    x_obs = 1 / T_true * d_smp * np.arange(Nc_obs)

    xi = 1 / (Nc_obs) * np.arange(Nc)
    xi_itpl = 1 / (d_smp * Nc_obs) * np.arange(Nc)
    xi_obs = 1 / Nc_obs * np.arange(Nc_obs)

    if TvFlag == 1:
        x_label = "$t$ (normalized by $T_{obs}$)"
        xi_label = "$f_{doppler}$ (normalized by $1/ \Delta t$)"
        yi_label = "Impulse response (abs)"
    elif TvFlag == 0:
        x_label = "$f$ (normalized by bandwidth)"
        xi_label = "$\u03c4$ (normalized by $1/ \Delta f$)"
        yi_label = "Doppler spectrum (abs)"


    ##############################################################################
    # ppt用
    pp = plot_config()
    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(6, 2))
    # Frequency response
    ax.plot(x, np.abs(h_true), label="True", color="b", linestyle="solid", linewidth = 3.0)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency response")
    ax.grid()

    fig, ax = plt.subplots(1, 1, tight_layout=True, figsize=(6, 2))
    # Frequency response
    ax.plot(x, np.abs(h_true), label="True", color="red", linestyle="solid", linewidth = 3.0)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Frequency response")
    ax.grid()
    ##############################################################################

    fig, ax = plt.subplots(1, 2, tight_layout=True)
    # Frequency response
    ax[0].plot(x, np.abs(h_true), label="True", color="b", linestyle="solid")
    ax[0].plot(x_obs, np.abs(h_obs), label="Observed", color="r", marker='o', linestyle="None")
    ax[0].plot(x_itpl, np.abs(h_itpl), label="Interpolation", color="r", linestyle="solid")
    ax[0].set_xlabel(x_label)
    ax[0].set_ylabel("Frequency response (abs)")
    ax[0].grid()
    ax[0].legend()

    # Impulse response
    ax[1].plot(xi, np.abs(hi_true), label="True", color="b", linestyle="solid", marker='o')
    ax[1].plot(xi_obs, np.abs(hi_obs), label="Observed", color="r", marker='o', linestyle="None")
    ax[1].plot(xi_itpl, np.abs(hi_itpl), label="Interpolation", color="r", linestyle="solid")
    ax[1].set_xlabel(xi_label)
    ax[1].set_ylabel(yi_label)
    ax[1].grid()
    ax[1].legend()

    plt.suptitle(title)


def plotTvCR3D(H, H_obs, H_itpl):
    # param
    Nc, Ns = H.shape  # Num of SubCarriers
    x = np.arange(Ns)
    y = np.arange(Nc)
    X, Y = np.meshgrid(x, y)

    ##############################################
    # plot 3D channel
    fig = plt.figure()
    # True TvCFR
    ax = fig.add_subplot(121, projection="3d")
    ax.set_xlabel("symbols")
    ax.set_ylabel("frequency")
    ax.set_zlabel("True TvCFR|H|")
    ax.plot_surface(X, Y, np.abs(H), cmap='ocean')
    # Interpolated TvCFR
    ax = fig.add_subplot(122, projection="3d")
    ax.set_xlabel("symbols")
    ax.set_ylabel("frequency")
    ax.set_zlabel("Interpolated Tv CFR |H_itpl|")
    ax.plot_surface(X, Y, np.abs(H_itpl), cmap='ocean')
    ##############################################

    ##############################################
    # plot heatmap
    fig, ax = plt.subplots(1, 3)
    # True
    im_0 = ax[0].imshow(np.abs(H), cmap="jet")
    ax[0].set_xlabel("symbols")
    ax[0].set_ylabel("frequency")
    ax[0].set_title("True")
    # fig.colorbar(im_0, ax=ax[0])
    ax[0].axes.xaxis.set_visible(False)
    ax[0].axes.yaxis.set_visible(False)

    # Obs
    im_1 = ax[1].imshow(np.abs(H_obs), cmap="jet")
    ax[1].set_xlabel("symbols")
    ax[1].set_ylabel("frequency")
    ax[1].set_title("Obs")
    # fig.colorbar(im_0, ax=ax[0])
    ax[1].axes.xaxis.set_visible(False)
    ax[1].axes.yaxis.set_visible(False)

    # Interpolation
    im_2 = ax[2].imshow(np.abs(H_itpl), cmap="jet")
    ax[2].set_xlabel("symbols")
    ax[2].set_ylabel("frequency")
    ax[2].set_title("Interpolation")
    # fig.colorbar(im_1, ax=ax[1])
    ax[2].axes.xaxis.set_visible(False)
    ax[2].axes.yaxis.set_visible(False)
    ##############################################


# plot Time Varying Channel Response
def plot_channel(H, H_obs, H_itpl):
    """
    :param H: True TvCFR (Nc, Ns)
    :param H_obs: Observed TvCFR (NC_obs, Ns_obs)
    :param H_itpl: Interpolated TvCFR (NC_obs, Ns_obs)
    :return:
    """
    # param
    Nc, Ns = H.shape  # Num of samples in f and t
    Nc_obs, Ns_obs = H_obs.shape  # Num of observed samples in f and t
    df_smp = np.ceil(Nc / Nc_obs).astype(int)  # sampling interval in f
    dt_smp = np.ceil(Ns / Ns_obs).astype(int)  # sampling interval in t

    # plitしたい(f, tau, sym)を指定
    kf_obs = 0
    ktau_obs = 0
    ksym_obs = 0
    kf = df_smp * kf_obs
    ktau = df_smp * ktau_obs
    ksym = dt_smp * ksym_obs

    # CFR, CIR (H, H_obs, H_itpl)
    Hi = transHrCR(H, Nc, Ns, "CFRtoCIR")
    Hi_obs = transHrCR(H_obs, Nc_obs, Ns_obs, "CFRtoCIR")
    # Hi_itpl = transHrCR(H_itpl, 1, "CFRtoCIR")
    Hi_itpl = transHrCR(H_obs, Nc, Ns, "CFRtoCIR")

    # Hdprl (Freq, Dplr)
    Hdprl = transHrCR(H, Nc, Ns, "TVtoDPLR")
    Hdplr_obs = transHrCR(H_obs, Nc_obs, Ns_obs, "TVtoDPLR")
    # Hdplr_itpl = transHrCR(H_itpl, 1, "TVtoDPLR")
    Hdplr_itpl = transHrCR(H_obs, Nc, Ns, "TVtoDPLR")

    ##############################################
    # plot CFR and CIR (snapshot)
    title = "CFR and CIR " + "(symbol:" + str(ksym) + ")"
    plotTvCR(H[:, ksym], Hi[:, ksym], H_obs[:, ksym_obs], Hi_obs[:, ksym_obs], H_itpl[:, ksym], Hi_itpl[:, ksym], title,
             0)

    # plot Time Varying
    title = "Time Varying CFR and CIR " + "(f=" + str(kf) + ", tau=" + str(ktau) + ")"
    plotTvCR(H[kf, :], Hdprl[ktau, :], H_obs[kf_obs, :], Hdplr_obs[ktau_obs, :], H_itpl[kf, :], Hdplr_itpl[ktau, :],
             title, 1)

    # plot 3D channel
    plotTvCR3D(H, H_obs, H_itpl)

    ##############################################
    # # save fig
    # H_save, min_H, max_H = create_save_H(H)  # (real, imag, zeros) に変形 (3次元目がないと保存できないので0で埋める)
    # H_obs_save, min_H_obs, max_H_obs = create_save_H(H_obs)
    # FpSaveData = "./save_data/"
    # plt.imsave(FpSaveData + "channel_true.png", H_save)
    # plt.imsave(FpSaveData + "channel_obs.png", H_obs_save)
    #
    # # save data
    # np.savez(FpSaveData + 'channel_np', H, H_obs, H_itpl)
    # FpSaveData = "./save_data/"
    # plt.imsave(FpSaveData + "channel_true_abs.png", np.abs(H), cmap='Greys_r')  # Greys_r
    # plt.imsave(FpSaveData + "channel_obs_abs.png", np.abs(H_obs), cmap='Greys_r')  # Greys_r
    # np.savez(FpSaveData + 'channel_np_abs', np.abs(H), np.abs(H_obs))
    ##############################################
    plt.show()


# チャネルを画像保存できる形式に変換
def create_save_H(H):
    H, min_H, max_H = normalizaton_complex(H)  # [0 1] に正規化 (min_H, max_H は元のスケールに戻すときに必要)
    H_save = np.zeros((H.shape[0], H.shape[1], 3))
    H_save[:, :, 0] = np.real(H)  # (real, imag, zeros)
    H_save[:, :, 1] = np.imag(H)
    return H_save, min_H, max_H
