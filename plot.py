import numpy as np
import matplotlib.pyplot as plt

def plot_gaussian_ab():

    # scale = 0.5
    # lr = 1e-3
    # epochs = 200

    # scale = 1.0
    # lr = 1e-4
    # epochs = 150

    scale = 2.0
    lr = 2e-6
    epochs = 201

    ab_list = [(0.1, 0.1), (0.5, 1.5), (2., 2.)]

    fig=plt.figure(figsize=(9,6))

    plt.rcParams['font.family']  = ['Times New Roman']
    plt.yticks(fontproperties = 'Times New Roman', size = 25)
    plt.xticks(fontproperties = 'Times New Roman', size = 25)


    for a,b in ab_list:
        ab_path = f'./log/ar/dim5_T100_iso1/lsa_a{a}_b{b}_scale{scale}_clipFalse_lr{lr}_seed1/ab.npy'
        ab = np.load(ab_path)
        plt.plot(ab[:epochs],label=f"$(a_0,b_0)$ = ({a}, {b})", linewidth=2.5)

    theory_item = (0.2 / scale ** 2)
    theory = theory_item * np.ones(shape=(200))
    plt.plot(theory[:epochs], '--',label=f"$1/5\sigma^2$ = {theory_item}", linewidth=2.5)

    plt.ylabel('Dynamics of ab',labelpad=8, fontsize = 30)
    plt.xlabel('Epoch',labelpad=8, fontsize = 30)
    plt.legend(fontsize = 20)

    plt.savefig(f'./figs/ab_gaussian_scale{scale}.jpg',bbox_inches='tight', dpi=400)

def plot_gaussian_gap():

    # scale = 0.5
    # lr = 1e-3
    # epochs = 200

    # scale = 1.0
    # lr = 1e-4
    # epochs = 150

    scale = 2.0
    lr = 2e-6
    epochs = 201

    ab_list = [(0.1, 0.1), (0.5, 1.5), (2., 2.)]

    fig=plt.figure(figsize=(9,6))

    plt.rcParams['font.family']  = ['Times New Roman']
    plt.yticks(fontproperties = 'Times New Roman', size = 25)
    plt.xticks(fontproperties = 'Times New Roman', size = 25)

    for a,b in ab_list:
        ab_path = f'./log/ar/dim5_T100_iso1/lsa_a{a}_b{b}_scale{scale}_clipFalse_lr{lr}_seed1/gap.npy'
        ab = np.load(ab_path)
        plt.plot(ab[1:epochs],label=f"$(a_0,b_0)$ = ({a}, {b})", linewidth=2.5)

    theory_item = 0.2
    theory = theory_item * np.ones(shape=(200))
    plt.plot(theory[:epochs], '--',label=f"Ratio = {theory_item}", linewidth=2.5)

    plt.ylabel('Ratio of pred/true at $T_{te}$',labelpad=8, fontsize = 30)
    plt.xlabel('Epoch',labelpad=8, fontsize = 30)
    plt.legend(fontsize = 20)

    plt.savefig(f'./figs/gap_gaussian_scale{scale}.jpg',bbox_inches='tight', dpi=400)
    plt.show()

def plot_sparse_ab():

    # scale = 0.5
    # lr = 3e-2
    # epochs = 200

    scale = 1.0
    lr = 1e-3
    epochs = 201

    # scale = 2.0
    # lr = 1e-4
    # epochs = 121

    ab_list = [(0.1, 0.1), (0.5, 1.5), (2., 2.)]

    fig=plt.figure(figsize=(9,6))

    plt.rcParams['font.family']  = ['Times New Roman']
    plt.yticks(fontproperties = 'Times New Roman', size = 25)
    plt.xticks(fontproperties = 'Times New Roman', size = 25)


    for a,b in ab_list:
        ab_path = f'./log/ar/dim5_T100_iso2/lsa_a{a}_b{b}_scale{scale}_clipFalse_lr{lr}_seed1/ab.npy'
        ab = np.load(ab_path)
        plt.plot(ab[1:epochs],label=f"$(a_0,b_0)$ = ({a}, {b})", linewidth=2.5)

    theory_item = (1 / scale ** 2)
    theory = theory_item * np.ones(shape=(200))
    plt.plot(theory[:epochs], '--',label=f"$1/c^2$ = {theory_item}", linewidth=2.5)

    plt.ylabel('Dynamics of ab',labelpad=8, fontsize = 30)
    plt.xlabel('Epoch',labelpad=8, fontsize = 30)
    plt.legend(fontsize = 20)

    plt.savefig(f'./figs/ab_sparse_scale{scale}.jpg',bbox_inches='tight', dpi=400)

def plot_sparse_gap():

    # scale = 0.5
    # lr = 3e-2
    # epochs = 200

    scale = 1.0
    lr = 1e-3
    epochs = 201

    # scale = 2.0
    # lr = 1e-4
    # epochs = 121

    ab_list = [(0.1, 0.1), (0.5, 1.5), (2., 2.)]

    fig=plt.figure(figsize=(9,6))

    plt.rcParams['font.family']  = ['Times New Roman']
    plt.yticks(fontproperties = 'Times New Roman', size = 25)
    plt.xticks(fontproperties = 'Times New Roman', size = 25)

    for a,b in ab_list:
        ab_path = f'./log/ar/dim5_T100_iso2/lsa_a{a}_b{b}_scale{scale}_clipFalse_lr{lr}_seed1/gap.npy'
        ab = np.load(ab_path)
        plt.plot(ab[5:epochs],label=f"$(a_0,b_0)$ = ({a}, {b})", linewidth=2.5)

    theory_item = 0
    theory = theory_item * np.ones(shape=(200))
    plt.plot(theory[:epochs], '--',label=f"Theoretical MSE = {theory_item}", linewidth=2.5)

    plt.ylabel('MSE loss at $T_{te}$',labelpad=8, fontsize = 30)
    plt.xlabel('Epoch',labelpad=8, fontsize = 30)
    plt.legend(fontsize = 20)

    plt.savefig(f'./figs/gap_sparse_scale{scale}.jpg',bbox_inches='tight', dpi=400)
    plt.show()

def plot_full_one_W():

    ab_list = [(0.1, 0.1), (0.5, 1.5), (2., 2.)]

    for a,b in ab_list:
        ab_path = f'./log/ar/dim5_T100_iso0/lsa_a{a}_b{b}_scale1.0_clipFalse_lr0.0005_seed1/WPV.npy'
        ab = np.load(ab_path)
        
        plt.imshow(ab, cmap='viridis', interpolation='nearest')
        plt.colorbar()

        plt.savefig(f'./figs/full_a{a}_b{b}_WPV.jpg',bbox_inches='tight', dpi=400)
        plt.close()

        ab_path = f'./log/ar/dim5_T100_iso0/lsa_a{a}_b{b}_scale1.0_clipFalse_lr0.0005_seed1/WKQ.npy'
        ab = np.load(ab_path)
        
        plt.imshow(ab, cmap='viridis', interpolation='nearest')
        plt.colorbar()

        plt.savefig(f'./figs/full_a{a}_b{b}_WKQ.jpg',bbox_inches='tight', dpi=400)
        plt.close()

if __name__ == '__main__':
    plot_full_one_W()