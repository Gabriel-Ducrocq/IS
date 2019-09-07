import numpy as np
import utils
import config
import matplotlib.pyplot as plt
import healpy as hp
from scipy.stats import invgamma
import time
import json


theta_ = utils.generate_theta()
cls_ = utils.generate_cls(theta_)
d_lm = utils.generate_sky_map(cls_, noise=True)
observed_cls = hp.alm2cl(d_lm, lmax=config.L_MAX_SCALARS)



def main(list_l):
    indexes = utils.get_l_major(list_l)
    d_lm_cut = d_lm[indexes]
    history_log_w = []
    for i in range(config.N_IS):
        print(i)
        #theta = utils.generate_theta()
        #cls = utils.generate_cls(theta)
        cls = utils.proposal_cls(observed_cls)
        s = utils.generate_sky_map(cls, noise=False)
        s_cut = s[indexes]
        log_w = -(1/2)*np.sum((np.abs(d_lm_cut-s_cut)**2)/config.noise_covar[0])
        history_log_w.append(log_w)



    exponent_w = np.exp(history_log_w - np.max(history_log_w))
    normalized_weights = exponent_w/np.sum(exponent_w)
    print(normalized_weights)
    ESS = 1/np.sum(normalized_weights**2)
    return ESS


if __name__ == "__main__":
    """
    print(cls_[30])
    print(config.noise_covar[0])
    all_l = np.arange(0, 1024, 10)
    all_l[0] = 2
    all_l = np.concatenate((all_l, np.array([1024])))
    h_ess = []
    start = time.time()
    for l in all_l:
        ess = main([l])
        h_ess.append(ess)

    print(time.time() - start)
    results = {"ess":h_ess, "l":all_l.tolist(), "noise_level": config.noise_covar[0], "sample_size":config.N_IS}
    with open("resultats.json", "w") as f:
        json.dump(results, f)
   """
    with open("resultats.json", "r") as f:
       r =  json.load(f)

    plt.plot(r["ess"])
    plt.show()
    #sampled_cls = invgamma.rvs(a=(2 * 1024 - 1) / 2, scale=(2 * 1024 + 1) * (1e3/(1024**2)) / 2, size = 1000000)
    #plt.hist(sampled_cls, bins=1000, density=True)
    #plt.axvline(x=cls_[1024], color="green")
    #plt.axvline(x=(1e3/(1024**2)) , color="red")
    #plt.show()
