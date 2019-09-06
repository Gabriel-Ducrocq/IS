import healpy as hp
import numpy as np
from classy import Class
import config
import json
import matplotlib.pyplot as plt
import pylab
import healpy as hp
from scipy.stats import invgamma


cosmo = Class()

LENSING = 'yes'
OUTPUT_CLASS = 'tCl pCl lCl'

COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"]
COSMO_PARAMS_MEAN = np.array([0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561])
COSMO_PARAMS_SIGMA = np.array([0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071])
LiteBIRD_sensitivities = np.array([36.1, 19.6, 20.2, 11.3, 10.3, 8.4, 7.0, 5.8, 4.7, 7.0, 5.8, 8.0, 9.1, 11.4, 19.6])


def generate_theta():
    return np.random.normal(COSMO_PARAMS_MEAN, COSMO_PARAMS_SIGMA)

def generate_cls(theta):
    params = {'output': OUTPUT_CLASS,
              'l_max_scalars': config.L_MAX_SCALARS,
              'lensing': LENSING}
    d = {name:val for name, val in zip(COSMO_PARAMS_NAMES, theta)}
    params.update(d)
    cosmo.set(params)
    cosmo.compute()
    cls = cosmo.lensed_cl(config.L_MAX_SCALARS)
    #10^12 parce que les cls sont exprimés en kelvin carré, du coup ça donne une stdd en 10^6
    cls["tt"] *= 1e12
    cosmo.struct_cleanup()
    cosmo.empty()
    return cls["tt"]


def generate_sky_map(cls_, noise=False):
    s = hp.synalm(cls_, lmax=config.L_MAX_SCALARS)
    if noise:
        n = np.sqrt(config.noise_covar[0]/2)*np.random.normal(size=config.dimension_sph) \
            + 1j * np.sqrt(config.noise_covar[0]/2)* np.random.normal(size=config.dimension_sph)

        n[:config.L_MAX_SCALARS+1] = np.sqrt(config.noise_covar[0])*np.random.normal(size=config.L_MAX_SCALARS+1)
        n[0] = 0
        n[1] = 0
        return s + n

    return s



def plot_results(path, index, check_alm = False):
    results = np.load(path)[()]
    path_cls = np.array(results["path_cls"])
    path_alms = np.array(results["path_alms"])
    obs = results["obs_map"]

    obs_alms = hp.map2alm(obs, lmax=results["config"]["L_MAX_SCALARS"])
    realized_cls = hp.anafast(obs, lmax=results["config"]["L_MAX_SCALARS"])
    true_cls = results["config"]["true_spectrum"]
    N_gibbs = results["config"]["N_gibbs"]
    if not check_alm:
        plt.plot(path_cls[:, index])
        plt.show()
        plt.close()
        plt.hist(path_cls[1000:, index], bins=100)
        plt.axvline(x=realized_cls[index], color='k', linestyle='dashed', linewidth=1)
        plt.axvline(x=true_cls[index], color='k', linewidth=1)
        plt.show()
    else:
        plt.plot(path_alms[:, index].imag)
        plt.show()
        plt.close()
        plt.hist(path_alms[1000:, index].imag, bins=50)
        plt.axvline(x=obs_alms[index].imag, color='k', linestyle='dashed', linewidth=1)
        #plt.axvline(x=true_cls[index], color='k', linewidth=1)
        plt.show()


    print(true_cls)
    print(obs_alms[:10])
    print(realized_cls)


def compare(conj_path, crank_path, mala_path, index):
    results_conj = np.load(conj_path)[()]
    results_crank = np.load(crank_path)[()]
    results_mala = np.load(mala_path)[()]
    path_cls_conj = np.array(results_conj["path_cls"])
    path_cls_crank = np.array(results_crank["path_cls"])
    path_cls_mala = np.array(results_mala["path_cls"])
    obs = results_conj["obs_map"]

    obs_alms = hp.map2alm(obs, lmax=results_conj["config"]["L_MAX_SCALARS"])
    realized_cls = hp.anafast(obs, lmax=results_conj["config"]["L_MAX_SCALARS"])
    true_cls = results_conj["config"]["true_spectrum"]
    N_gibbs = results_conj["config"]["N_gibbs"]

    plt.plot(path_cls_conj[:, index])
    plt.plot(path_cls_crank[:, index])
    plt.plot(path_cls_mala[:, index])
    plt.show()
    plt.close()
    plt.hist(path_cls_conj[1000:, index], bins=25, density = True, alpha=0.2, label="CG")
    plt.hist(path_cls_crank[1000:, index], bins=25, density=True, alpha=0.2, label="CN")
    plt.hist(path_cls_mala[1000:, index], bins=25, density=True, alpha=0.2, label="MALA")
    plt.axvline(x=realized_cls[index], color='k', linestyle='dashed', linewidth=1)
    plt.axvline(x=true_cls[index], color='k', linewidth=1)
    plt.legend(loc="upper right")
    plt.show()

    print("N_gibbs")
    print(N_gibbs)
    print("N_mala")
    print(results_mala["config"]["N_crank"])


def get_l_major(list_l):
    indexes = []
    for l in list_l:
        for m in range(l+1):
            idx = hp.sphtfunc.Alm().getidx(config.L_MAX_SCALARS, l, m)
            indexes.append(idx)

    print(indexes)
    return indexes


def extend_cls_l_major(cls, l_min):
    all_cls = []
    for l, cl in enumerate(cls):
        e = [cl/2 for _ in range(l + l_min + 1)]
        e[0] = cl
        all_cls.append(e)

    return all_cls


def proposal_cls(observed_cls):
    l = np.array(range(2,config.L_MAX_SCALARS+1))
    sampled_cls = invgamma.rvs(a=(2 * l - 1) / 2, scale=(2 * l + 1) * (1e3/(l**2)) / 2)
    all_cls = np.concatenate((np.zeros(2), sampled_cls))
    return all_cls