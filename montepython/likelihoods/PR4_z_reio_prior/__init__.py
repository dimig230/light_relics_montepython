import os
from montepython.likelihood_class import Likelihood_prior


class PR4_z_reio_prior(Likelihood_prior):

    # initialisation of the class is done within the parent Likelihood_prior. For
    # this case, it does not differ, actually, from the __init__ method in
    # Likelihood class.
    def loglkl(self, cosmo, data):

        z_reio = cosmo.z_reio()
        loglkl = -0.5 * (z_reio - self.z_reio) ** 2 / (self.sigma ** 2)
        return loglkl
