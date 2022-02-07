import numpy as np
from src.TimeVaryingChannel.myfunc import *  # function

# Calc TimeVarying Channel
def tv_chan(Nc, Ns, Nl, decay_db, tauFs_max, fdTs_max):
    """
    :param Nc: サブキャリア数
    :param Ns: シンボル数
    :param Nl: 到来波数
    :param decay_db: decay factor of delay path [dB]
    :param tauFs_max: 正規化遅延時間の最大値 (サブキャリア間隔Fs で正規化)
    :param fdTs_max: 正規化ドップラー周波数の最大値 (シンボル長Ts で正規化)
    :return:
    """

    # param
    tauFs_set = np.linspace(0, tauFs_max, Nl)  # 正規化遅延時間
    doa_set = np.pi * np.random.rand(Nl) - np.pi/2  # DoA [-pi/2 pi/2] 一様分布
    # doa_set = 1/Nl * np.pi * np.arange(Nl)  # DoA  [-pi/2 pi/2] 等間隔
    # fdTs_set = 2 * fdTs_max * np.cos(doa_set)  # ドップラー周波数は正の値のみに限定している（その分周波数自体を2倍にして帯域は等価にしている）
    fdTs_set = fdTs_max * np.cos(doa_set)  # ドップラー周波数は正の値のみに限定している（その分周波数自体を2倍にして帯域は等価にしている）
    amp_set = np.array([db2real(-l * decay_db) for l in range(Nl)])  # Amplitude
    ini_phase_set = -1j * 2 * np.pi * np.random.rand(Nl)
    a_init = 1 / np.linalg.norm(amp_set, ord=2) * amp_set * np.exp(ini_phase_set)  # Complex Amplitude

    # calc channel
    F = 1 / np.sqrt(Nc) * np.fft.fft(np.eye(Nc))  # DFT matrix
    B = np.zeros((Nc, Nl), dtype=complex)  # Array Factor
    for l in range(Nl):
        phy = 2 * np.pi * tauFs_set[l]
        B[:, l] = np.exp(-1j * np.linspace(0, Nc * phy, Nc))

    A = np.zeros((Nl, Ns), dtype=complex)
    H = np.zeros((Nc, Ns), dtype=complex)
    # Hi = np.zeros((Nc, Ns), dtype=complex)

    for t in range(Ns):  # time varying
        # calc channel
        a = a_init * np.exp(1j * 2 * np.pi * fdTs_set * t)  # Complex Amplitude expの中は正の値しかとらないように調整(本来の負の分を正側に押し込めてる)
        h = B @ a  # Frequency response (Nc, 1)
        # hi = np.conjugate(F.T) @ h  # Impulse response (Nc, 1)

        # store data
        A[:, t] = a
        H[:, t] = h
        # Hi[:, t] = hi

    return H

# Transform CFR->CIR or CIR->CFR (High Resolution)
def transHrCR(H, Nc_hr, Ns_hr, TypeCR):
    """
    :param H: CFR (Channel Frequency Response) or CIR (Impulse) (Nc, Ns)
    :param k: resolution coefficient
    :param mCR: "CFRtoCIR" or "CIRtoCFR"
    :return: H_hr: High resolution TvCFR or TvCIR ot DplrCFR or DplrCIR
    """
    Nc, Ns = H.shape  # Num of samples

    if TypeCR == "CFRtoCIR":
        F = 1 / np.sqrt(Nc) * np.fft.fft(np.eye(Nc_hr))  # DFT matrix
        H_hr = np.conjugate(F[0:Nc, :].T) @ H  # HrCIR (Nc_hr, Ns) (IDFT)

    elif TypeCR == "CIRtoCFR":
        F = 1 / np.sqrt(Nc) * np.fft.fft(np.eye(Nc_hr))  # DFT matrix
        H_hr = F[0:Nc, :].T @ H  # HrCIR (Nc_hr, Ns) (IDFT)

    elif TypeCR == "CFRtoCFR":
        F_1 = 1 / np.sqrt(Nc) * np.fft.fft(np.eye(Nc))
        F_2 = 1 / np.sqrt(Nc) * np.fft.fft(np.eye(Nc_hr))
        H_hr = F_2[0:Nc, :].T @ np.conjugate(F_1.T) @ H  # HrCFR ot HrCIR (Nc_hr, 1)

    elif TypeCR == "CIRtoCIR":
        F_1 = 1 / np.sqrt(Nc) * np.fft.fft(np.eye(Nc))
        F_2 = 1 / np.sqrt(Nc) * np.fft.fft(np.eye(Nc_hr))
        H_hr = np.conjugate(F_2[0:Nc, :].T) @ F_1.T @ H  # HrCFR ot HrCIR (Nc_hr, 1)

    elif TypeCR == "TVtoDPLR":
        F = 1 / np.sqrt(Ns) * np.fft.fft(np.eye(Ns_hr))
        H_hr = H @ F[0:Ns, :]  # DFT

    elif TypeCR == "DPLRtoTV":
        F = 1 / np.sqrt(Ns) * np.fft.fft(np.eye(Ns_hr))
        H_hr = H @ np.conjugate(F[0:Ns, :])  # IDFT

    return H_hr

# DFT Interpolation
def DftInterpolateCR(H_obs, Nc, Ns):
    """
    :param H_obs: Observed tvCFR (Nc_obs, Ns_obs)
    :param Nc: Num of samples in Frequency (Nc > Nc_obs)
    :param Ns: Num of samples in Time (Ns > Ns_obs)
    :return H_itpl: Interpolated tvCFR (Nc, Ns)
    """

    # param
    Nc_obs, Ns_obs = H_obs.shape

    # Frequency interpolation matrix
    D_f = 1 / np.sqrt(Nc_obs) * np.fft.fft(np.eye(Nc_obs))
    Dh_f = 1 / np.sqrt(Nc_obs) * np.fft.fft(np.eye(Nc))
    F_f = Dh_f[0:Nc_obs, :].T @ np.conjugate(D_f)
    # F_f = np.conjugate(Dh_f[0:Nc_obs, :].T) @ D_f

    # Time interpolation matrix
    D_t = 1 / np.sqrt(Ns_obs) * np.fft.fft(np.eye(Ns_obs))
    Dh_t = 1 / np.sqrt(Ns_obs) * np.fft.fft(np.eye(Ns))
    F_t = np.conjugate(Dh_t[0:Ns_obs, :].T) @ D_t

    H_itpl = F_f @ H_obs @ F_t.T  # interpolation

    return H_itpl

