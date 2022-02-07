import matplotlib.pyplot as plt

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