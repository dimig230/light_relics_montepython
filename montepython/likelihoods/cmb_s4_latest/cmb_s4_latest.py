import os
import numpy as np
from montepython.likelihood_class import Likelihood

class cmb_s4_latest(Likelihood):
    """
    CMB-S4 mock likelihood (TT+EE, diagonal, no TE).

    Reads dataset selector from cmb_s4_latest.data (or .param):
      data_file = '<basename>'  -> resolves data/cmb_s4_latest/<basename>.data

    The dataset .data must define at minimum:
      npy_file = /abs/path/to/s4wide_... .npy  (contains keys: 'el', 'cl_residual', 'fsky_val', ...)
      lmax = <int>                              (optional, inferred from 'el' if missing)

    Optional keys in dataset .data:
      lmin = <int>              (default: 10 if present in npy param_dict, else 2)
      use_TE = yes|no           (default: no)
      spectra = TT,EE           (subset control, default: TT,EE)
      cl_units = Cl|Dl          (theory conversion, default: Cl)

    Likelihood model (per-ℓ, diagonal approx):
      Var_TT ≈ 2/(2ℓ+1)/f_sky * (C_TT_th + N_TT)^2, and similarly for EE.
      -2 ln L = Σℓ [(C_th - C_obs)^2 / Var].

    Observed spectra C_obs are taken from a fiducial file if available:
      data/cmbs4_loverde_fiducial.dat  (columns: ℓ, C_TT, C_EE) else set C_obs = 0.
    """
    def __init__(self, path, data, command_line):
        # path -> .../likelihoods/cmb_s4_latest/cmb_s4_latest.data
        Likelihood.__init__(self, path, data, command_line)

        # dataset basename can be set here or in the .param
        dataset = getattr(self, 'data_file', None)
        if not dataset:
            raise ValueError("cmb_s4_latest: set `data_file = <basename>` in cmb_s4_latest.data or in the .param")

        # Locate the dataset config in the data tree
        data_cfg = os.path.join(data.path['data'], 'cmb_s4_latest', dataset + '.data')
        if not os.path.exists(data_cfg):
            raise IOError(f"cmb_s4_latest: dataset config not found: {data_cfg}")

        # Parse keys from the dataset .data
        opts = {}
        with open(data_cfg, 'r') as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith('#'):
                    continue
                if '=' in s:
                    k, v = s.split('=', 1)
                    opts[k.strip()] = v.strip().strip('"').strip("'")

        npy_file = opts.get('npy_file')
        if not npy_file or not os.path.exists(npy_file):
            raise IOError(f"cmb_s4_latest: npy_file missing or not found (from {data_cfg}): {npy_file}")

        self.npy_file = npy_file
        # Load mock
        mock = np.load(self.npy_file, allow_pickle=True).item()
        el = mock.get('el', None)
        if el is None or len(el) == 0:
            raise ValueError("cmb_s4_latest: 'el' missing from npy file")
        self.el = np.asarray(el, dtype=int)
        self.fsky = float(mock.get('fsky_val', 1.0))

        # Residual (noise) power spectra (dict with 'TT','EE')
        cl_resid = mock.get('cl_residual', {}) or {}
        self.nl_tt = np.asarray(cl_resid.get('TT', np.zeros_like(self.el)), dtype=float)
        self.nl_ee = np.asarray(cl_resid.get('EE', np.zeros_like(self.el)), dtype=float)

        # lmin, lmax and unit options
        lmax_opt = int(opts.get('lmax', '0')) if 'lmax' in opts else 0
        lmax_from_file = int(self.el[-1])
        self.lmax = lmax_opt or lmax_from_file
        # Prefer explicit lmin in dataset, else try from mock param_dict, else 2
        if 'lmin' in opts:
            self.lmin = int(opts['lmin'])
        else:
            pd = mock.get('param_dict', {}) or {}
            self.lmin = int(pd.get('lmin', 2))

        self.use_TE = str(opts.get('use_TE', 'no')).lower() in ('yes', 'true', '1')
        # Default to Dl to match provided fiducial files in data/
        self.cl_units = str(opts.get('cl_units', 'Dl')).strip()

        # retain mock for optional fields needed later
        self._mock = mock

        # Try to load a fiducial observed Cl file (optional)
        self.obs_ell = None
        self.obs_tt = None
        self.obs_ee = None
        fid_path = os.path.join(data.path['data'], 'cmbs4_loverde_fiducial.dat')
        if os.path.exists(fid_path):
            try:
                # Columns: ell, C_TT, C_EE (assumed in same units as theory conversion below)
                arr = np.loadtxt(fid_path)
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    self.obs_ell = arr[:, 0].astype(int)
                    self.obs_tt = arr[:, 1].astype(float)
                    self.obs_ee = arr[:, 2].astype(float)
            except Exception:
                # If parsing fails, fall back to zeros
                self.obs_ell = None

        # Ensure CLASS computes required spectra up to lmax
        self.need_cosmo_arguments(data, {
            'l_max_scalars': self.lmax,
            'output': 'tCl,pCl,lCl',
            'lensing': 'yes',
            'modes': 's'
        })

        # Precompute index mask for used multipoles intersecting all sources
        self._build_l_masks()

    def _build_l_masks(self):
        # Valid ℓ from 2..lmax
        l = np.arange(self.lmax + 1)
        mask = (l >= max(self.lmin, 2)) & (l <= self.lmax)

        # Interpolate arrays to full ℓ-grid if needed
        src_l = self.el if self.el[0] == 0 else np.arange(len(self.el))

        def interp_to_full(y, left=0.0, right=None):
            y = np.asarray(y, dtype=float)
            # Handle length mismatches gracefully
            try:
                if right is None:
                    right = y[-1]
                return np.interp(l, src_l, y, left=left, right=right)
            except Exception:
                return np.zeros_like(l, dtype=float)

        # Noise/residuals: if working in Dl use dl_residual_* if present, otherwise convert
        if self.cl_units.lower() == 'dl' and (
            ('dl_residual_TT' in getattr(self, '_mock', {})) or ('dl_residual_EE' in getattr(self, '_mock', {}))
        ):
            dl_tt = getattr(self, '_mock', {}).get('dl_residual_TT', None)
            dl_ee = getattr(self, '_mock', {}).get('dl_residual_EE', None)
            if dl_tt is not None and dl_ee is not None:
                self._nl_tt_full = interp_to_full(dl_tt, left=0.0, right=dl_tt[-1])
                self._nl_ee_full = interp_to_full(dl_ee, left=0.0, right=dl_ee[-1])
            else:
                # Fallback: convert Cl residuals to Dl
                factor = l * (l + 1.0) / (2.0 * np.pi)
                self._nl_tt_full = interp_to_full(self.nl_tt) * factor
                self._nl_ee_full = interp_to_full(self.nl_ee) * factor
        else:
            self._nl_tt_full = interp_to_full(self.nl_tt)
            self._nl_ee_full = interp_to_full(self.nl_ee)

        # Observed fiducial spectra (optional)
        if self.obs_ell is not None and self.obs_tt is not None and self.obs_ee is not None:
            try:
                self._obs_tt_full = np.interp(l, self.obs_ell, self.obs_tt, left=0.0, right=self.obs_tt[-1])
                self._obs_ee_full = np.interp(l, self.obs_ell, self.obs_ee, left=0.0, right=self.obs_ee[-1])
            except Exception:
                self._obs_tt_full = np.zeros_like(l)
                self._obs_ee_full = np.zeros_like(l)
        else:
            self._obs_tt_full = np.zeros_like(l)
            self._obs_ee_full = np.zeros_like(l)

        self._ells = l
        self._mask = mask

    def _get_theory_cls(self, cosmo):
        # Fetch lensed Cl from CLASS up to lmax
        cl = cosmo.lensed_cl(self.lmax)
        # cl dict keys: 'tt','ee','te','bb','pp','tp' with arrays starting at ℓ=2
        # Build full arrays from ℓ=0..lmax
        l = self._ells
        tt = np.zeros_like(l, dtype=float)
        ee = np.zeros_like(l, dtype=float)
        # Insert from index 2
        tt[2:] = cl['tt'][2:]
        ee[2:] = cl['ee'][2:]

        if self.cl_units.lower() == 'dl':
            # Convert to D_ell = ℓ(ℓ+1)/2π C_ell
            factor = l * (l + 1.0) / (2.0 * np.pi)
            tt = tt * factor
            ee = ee * factor
        # else keep as Cl
        return tt, ee

    def loglkl(self, cosmo, data):
        # Theory C_ell
        th_tt, th_ee = self._get_theory_cls(cosmo)

        # Slice to used multipoles
        m = self._mask
        ells = self._ells[m]

        # Observed and noise
        obs_tt = self._obs_tt_full[m]
        obs_ee = self._obs_ee_full[m]
        nl_tt = self._nl_tt_full[m]
        nl_ee = self._nl_ee_full[m]

        # Guardrails
        fsky = max(self.fsky, 1e-3)
        # Avoid zero variances
        eps = 1e-30

        # Variances (Gaussian, diagonal)
        prefac = 2.0 / (2.0 * ells + 1.0) / fsky
        var_tt = prefac * np.maximum(th_tt[m] + nl_tt, eps) ** 2
        var_ee = prefac * np.maximum(th_ee[m] + nl_ee, eps) ** 2

        # Chi-square
        chi2_tt = np.sum((th_tt[m] - obs_tt) ** 2 / var_tt)
        chi2_ee = np.sum((th_ee[m] - obs_ee) ** 2 / var_ee)
        chi2 = chi2_tt + chi2_ee

        return -0.5 * chi2