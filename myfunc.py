import numpy as np


# 配列を列ベクトルに変形 (N,1)
def c_vec(array):
    return array.reshape(-1, 1)


# 配列を行ベクトルに変形 (1,N)
def r_vec(array):
    return array.reshape(1, -1)


# [dB]から電力に変換
def db2real(db_array):
    real_array = 10 ** (db_array / 10)
    return real_array


# 電力から[dB]に変換
def real2db(real_array):
    db_array = 10 * np.log10(real_array)
    return db_array


# 複素行列を [0 1] に正規化
def normalizaton_complex(H):
    min_H = np.min(np.array([np.min(np.real(H)), np.min(np.imag(H))]))  # 実部と虚部の最小値の中での最小値
    max_H = np.max(np.array([np.max(np.real(H)), np.max(np.imag(H))]))  # 実部と虚部の最大値の中での最大値
    H = (H - min_H * (1 + 1j)) / (max_H - min_H)  # [0 1]に正規化
    return H, min_H, max_H

# [0 1] に正規化された複素行列を元のスケールに戻す
def denormalizaton_complex(H, min_H, max_H):
    return (max_H - min_H) * H + min_H * (1 + 1j)

