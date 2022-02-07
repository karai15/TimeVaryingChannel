import numpy as np
import random
import GPy
import matplotlib.pyplot as plt
import copy

# GP
def gp_interpolation(Xa_obs, Ya_obs, Za_obs, Xa_pred, Ya_pred, z_true_2d, max_iters=100):
    """
    参考サイト
        https://qiita.com/nmmg0031785/items/06c94b688ce9a29d346c
        https://statmodeling.hatenablog.com/entry/how-to-use-GPy
    """

    # データ読み込み
    Input = np.stack([Xa_obs, Ya_obs], axis=1)  # (x座標, y座標)
    Output = Za_obs[:, None]  # (z座標)

    # モデル構築
    kernel = GPy.kern.RBF(input_dim=2, ARD=True)  # カーネルの種類 https://qiita.com/goldengate94704/items/471a7fec5513e681a138
    # kernel = GPy.kern.Matern32(input_dim=2, ARD=True) # ARD=Trueは入力の次元1つに対し、1つのlengthscaleパラメータを割り振ること（すなわちGPは等方でないことを表す
    model = GPy.models.GPRegression(Input, Output, kernel)
    model.Gaussian_noise.variance = 1/31.6227766  # ノイズ分散
    model.Gaussian_noise.variance.fix()  # パラメータ固定
    model.optimize(messages=True, max_iters=max_iters)
    print(model)

    # 予測
    Nx = z_true_2d.shape[0]
    Ny = z_true_2d.shape[1]
    xy_pred = np.array([Xa_pred, Ya_pred]).T
    z_qua_pred = model.predict_quantiles(xy_pred, quantiles=(2.5, 50, 97.5))
    z_pred_mean = z_qua_pred[1][:, 0]  # 期待値
    z_pred_2d = np.zeros((Nx, Ny))  # (Nx, Ny) の予測画像
    for nx in range(Nx):
        z_pred_2d[nx, :] = z_pred_mean[nx * Ny:(nx + 1) * Ny]


    # MSE評価
    MSE_gp = 1 / z_true_2d.size * np.linalg.norm(z_true_2d - z_pred_2d, "fro") ** 2
    # print("MSE_GP:", MSE_gp)

    ##############################################
    # ## plot
    # fig, ax = plt.subplots(1, 2, squeeze=False)
    # ax[0, 0].imshow(z_true_2d, cmap="jet")
    # ax[0, 0].set_title("True")
    # ax[0, 1].imshow(z_pred_2d, cmap="jet")
    # ax[0, 1].set_title("Pred_GP")
    # plt.show()
    ##############################################

    return z_pred_2d, MSE_gp  # 予測


# 多項式補間
def polynomial_interpolation(Z_true, Z_obs, kind_list):
    from scipy import interpolate
    import copy

    # param
    Nx = Z_true.shape[0]
    Ny = Z_true.shape[0]
    Nx_obs = Z_obs.shape[0]
    Ny_obs = Z_obs.shape[1]
    rx = int(np.ceil(Nx / Nx_obs))  # サンプルレート
    ry = int(np.ceil(Ny / Ny_obs))  # サンプルレート

    # データ
    x_true = np.arange(Nx)
    y_true = np.arange(Ny)
    X_true, Y_true = np.meshgrid(x_true, y_true)
    X_obs = X_true[0:Nx:rx, 0:Ny:ry]
    Y_obs = Y_true[0:Nx:rx, 0:Ny:ry]

    # 補間
    # kind_list = ["linear", "cubic", "quintic"]
    Z_itpl_list = []
    for kind in kind_list:
        f_itpl = interpolate.interp2d(X_obs, Y_obs, Z_obs, kind=kind)  # kind= linear(1次), cubic(3次)  quintic(5次)
        Z_itpl = f_itpl(x_true, y_true)
        Z_itpl_list.append(Z_itpl)

    # MSE評価
    N_kind = len(kind_list)
    MSE_itpl_list = []
    for n_kind in range(N_kind):
        MSE_itpl = 1 / Z_true.size * np.linalg.norm(Z_true - Z_itpl_list[n_kind], "fro") ** 2
        MSE_itpl_list.append(MSE_itpl)
        # print("MSE_" + kind_list[n_kind] + ": ", MSE_itpl)

    ##############################################
    # ## plot
    # fig, ax = plt.subplots(1, N_kind+1, squeeze=False)
    # for n_kind in range(N_kind):
    #     ax[0, n_kind].imshow(Z_itpl_list[n_kind], cmap="jet")
    #     ax[0, n_kind].set_title(kind_list[n_kind])
    # ax[0, N_kind].imshow(Z_true, cmap="jet")
    # ax[0, N_kind].set_title("True")
    # plt.show()
    ##############################################

    return Z_itpl_list, MSE_itpl_list

# DFT補間
def dft_interpolation(H_true, H_obs):

    # param
    Nc, Ns = H_true.shape
    Nc_obs, Ns_obs = H_obs.shape

    # 周波数領域
    D_f = 1 / np.sqrt(Nc_obs) * np.fft.fft(np.eye(Nc_obs))
    Dh_f = 1 / np.sqrt(Nc_obs) * np.fft.fft(np.eye(Nc))
    Dh_f = Dh_f[:, 0:Nc_obs]  # 200, 50

    if Nc_obs % 2 == 0:  # 偶数の場合
        nc_inter = int(np.floor(Nc_obs / 2))  # 中央の値
        Dh_f_post = np.fliplr(np.conjugate(Dh_f[:, 1:nc_inter]))  # 前半部分の共役を後半に追加 (実質半分しかスペクトルを見ていない)
        Dh_f[:, nc_inter+1:Nc_obs] = Dh_f_post
    else:
        nc_inter = int(np.ceil(Nc_obs / 2))  # 中央の値
        Dh_f_post = np.fliplr(np.conjugate(Dh_f[:, 1:nc_inter]))  # 前半部分の共役を後半に追加 (実質半分しかスペクトルを見ていない)
        Dh_f[:, nc_inter:Nc_obs] = Dh_f_post

    F_f = Dh_f @ np.conjugate(D_f)

    # 時間領域
    D_t = 1 / np.sqrt(Ns_obs) * np.fft.fft(np.eye(Ns_obs))
    Dh_t = 1 / np.sqrt(Ns_obs) * np.fft.fft(np.eye(Ns))
    Dh_t = Dh_t[:, 0:Ns_obs]  # 200, 50

    # ns_inter = int(np.floor(Ns_obs/2))  # 中央の値
    # Dh_t_post = np.fliplr(np.conjugate(Dh_t[:, 1:ns_inter]))  # 前半部分の共役を後半に追加 (実質半分しかスペクトルを見ていない)
    # Dh_t[:, ns_inter+1:Ns_obs] = Dh_t_post

    if Ns_obs % 2 == 0:  # 偶数の場合
        ns_inter = int(np.floor(Ns_obs / 2))  # 中央の値
        Dh_t_post = np.fliplr(np.conjugate(Dh_t[:, 1:ns_inter]))  # 前半部分の共役を後半に追加 (実質半分しかスペクトルを見ていない)
        Dh_t[:, ns_inter+1:Ns_obs] = Dh_t_post
    else:
        ns_inter = int(np.ceil(Ns_obs / 2))  # 中央の値
        Dh_t_post = np.fliplr(np.conjugate(Dh_t[:, 1:ns_inter]))  # 前半部分の共役を後半に追加 (実質半分しかスペクトルを見ていない)
        Dh_t[:, ns_inter:Ns_obs] = Dh_t_post

    F_t = Dh_t @ np.conjugate(D_t)

    # 補間
    H_itpl = F_f @ H_obs @ F_t.T  # interpolation

    # # MSE評価
    MSE_itpl = 1 / H_true.size * np.linalg.norm(H_true - H_itpl, "fro") ** 2
    # print("MSE_DFT: ", MSE_itpl)

    ##############################################
    # ## plot
    # fig, ax = plt.subplots(1, 2, squeeze=False)
    # ax[0, 0].imshow(np.abs(H_true), cmap="jet")
    # ax[0, 0].set_title("True")
    # ax[0, 1].imshow(np.abs(H_itpl), cmap="jet")
    # ax[0, 1].set_title("True")
    # plt.show()
    ##############################################

    return H_itpl, MSE_itpl