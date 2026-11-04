import math


class RTKInitializer:
    """
    Generates initial RTK unit hydrograph parameters and bounds for use in
    optimizing RDII (Rainfall-Derived Inflow and Infiltration) in EPA SWMM.

    The RTK approach represents RDII as the sum of up to three triangular unit
    hydrographs, each defined by:
        R — fraction of rainfall volume entering the sewer as RDII
        T — time to peak of the triangle (hours)
        K — ratio of recession time to time to peak (dimensionless, no upper bound)

    Triangles are ordered fast -> slow, representing:
        Fast   : direct inflow (roof drains, downspouts, sump pumps)
        Medium : near-surface infiltration through defective joints/pipes
        Slow   : deep groundwater infiltration through cracked pipes

    The sum of all R values must be <= 1.0 — enforce this as a constraint in
    your optimizer (e.g. scipy LinearConstraint with SLSQP). Individual R
    upper bounds are set to 1.0 so any single triangle can carry the full RDII
    response if the data supports it. R values do not need to sum to exactly
    1.0 — the remainder represents rainfall that never enters the sewer system.

    Supports catchment areas from ~5 acres (rooftop) to ~500,000 acres (HUC-10).

    Usage
    -----
    from rtk_initializer import RTKInitializer

    rtk = RTKInitializer(n_triangles=3, area_acres=75_000)

    # Structured parameter dicts
    params = rtk.params()

    # Flat format for scipy.optimize.minimize
    # iterate over and create lower upper bounds (MD)
    x0, bounds = rtk.flat_params()

    # Print a summary table
    rtk.print_params()
    """

    # K is a recession time ratio with no hard upper bound. A value of 1.0 means
    # symmetric triangle (recession = rise time). Values > 1 are physically normal,
    # especially for slow infiltration components where groundwater drains gradually
    # over many times the rise period. Typical range is 0.5 to 4.0+.
    # A small lower bound > 0 is enforced to prevent a zero-duration recession limb.
    K_MIN = 0.1

    # Regime anchor points defining fast and slow RDII response endpoints at
    # representative areas across five orders of magnitude. Parameters for a
    # given catchment are blended in log-area space between the two bracketing
    # regimes so transitions are smooth rather than abrupt.
    #
    # Each regime has:
    #   area_ref : representative area in acres
    #   fast     : anchor for the fastest (direct inflow) triangle
    #   slow     : anchor for the slowest (groundwater infiltration) triangle
    #
    # T values are in hours. R is dimensionless [0, 1]. K is dimensionless,
    # unbounded above — values >> 1 are physically meaningful for slow components.
    REGIMES = [
        {
            "name": "micro",
            "area_ref": 50.0,           # acres — rooftop / lot scale
            "fast": {
                "R_init": 0.15,
                # Very short time to peak: minutes to low single-digit hours
                "T_init": 0.25, "T_lo": 0.05, "T_hi": 2.0,    # hours
                # Near-symmetric triangle: inflow drains quickly after peak
                "K_init": 0.60, "K_lo": 0.20, "K_hi": 1.50,
            },
            "slow": {
                "R_init": 0.03,
                # Slow component still resolves within hours at lot scale
                "T_init": 6.0,  "T_lo": 1.0,  "T_hi": 24.0,
                # Longer recession than rise: subsurface drainage takes time
                "K_init": 1.50, "K_lo": 0.50, "K_hi": 3.00,
            },
        },
        {
            "name": "small",
            "area_ref": 500.0,          # acres — farm field / small urban catchment
            "fast": {
                "R_init": 0.12,
                "T_init": 1.0,  "T_lo": 0.25, "T_hi": 8.0,
                # K slightly above 1: modest recession tail on fast inflow
                "K_init": 0.70, "K_lo": 0.20, "K_hi": 5.0,
            },
            "slow": {
                "R_init": 0.04,
                # Slow component extends to days at this scale
                "T_init": 24.0, "T_lo": 6.0,  "T_hi": 96.0,
                # Long recession: infiltrating groundwater drains slowly
                "K_init": 2.00, "K_lo": 0.50, "K_hi": 8.0,
            },
        },
        {
            "name": "medium",
            "area_ref": 10_000.0,       # acres — HUC-12 scale
            "fast": {
                "R_init": 0.10,
                # Channel routing adds to travel time; hours to half a day
                "T_init": 4.0,  "T_lo": 0.5,  "T_hi": 24.0,
                "K_init": 0.80, "K_lo": 0.20, "K_hi": 5.0,
            },
            "slow": {
                "R_init": 0.05,
                # Slow component spans days to weeks
                "T_init": 96.0, "T_lo": 24.0, "T_hi": 360.0,
                # High K: deep infiltration recedes very gradually
                "K_init": 2.50, "K_lo": 0.50, "K_hi": 10.0,
            },
        },
        {
            "name": "large",
            "area_ref": 85_000.0,       # acres — original reference scale for this model
            "fast": {
                "R_init": 0.08,
                # Fast response still takes hours to days at this scale
                "T_init": 10.0, "T_lo": 1.0,  "T_hi": 72.0,
                "K_init": 1.00, "K_lo": 0.20, "K_hi": 8.0,
            },
            "slow": {
                "R_init": 0.05,
                # Slow component can take weeks to months to fully recede
                "T_init": 270., "T_lo": 100., "T_hi": 1200.0,
                # Very high K: regional groundwater contributions drain over months
                "K_init": 3.00, "K_lo": 1.00, "K_hi": 15.0,
            },
        },
        {
            "name": "huc10",
            "area_ref": 500_000.0,      # acres — ~780 sq mi, upper HUC-10 range
            "fast": {
                # Even fast inflow has significant basin travel time at this scale
                "R_init": 0.06,
                "T_init": 24.0, "T_lo": 4.0,  "T_hi": 120.0,
                "K_init": 1.20, "K_lo": 0.30, "K_hi": 10.0,
            },
            "slow": {
                "R_init": 0.04,
                # Deep groundwater discharge; recession spans months
                "T_init": 720., "T_lo": 240., "T_hi": 4320.0,  # up to ~6 months
                # K can be very large: recession many times longer than rise
                "K_init": 4.00, "K_lo": 1.00, "K_hi": 20.0,
            },
        },
    ]

    # Precomputed log-space area values aligned with REGIMES list,
    # used by _blend_regimes() to locate bracketing regimes efficiently
    _LOG_AREAS = [math.log(r["area_ref"]) for r in REGIMES]

    def __init__(self, n_triangles: int, area_acres: float = 85_000.0):
        """
        Parameters
        ----------
        n_triangles : int
            Number of RTK triangles (>= 1). Each represents one distinct RDII
            pathway. EPA SWMM supports up to 3; typical usage is 2-3.
        area_acres : float
            Catchment area in acres. Defaults to 85,000 ac (original reference).
        """
        if n_triangles < 1:
            raise ValueError("n_triangles must be >= 1")
        if area_acres <= 0:
            raise ValueError("area_acres must be positive")

        self.n_triangles = n_triangles
        self.area_acres  = area_acres

        # Compute and cache parameters on construction so repeated calls to
        # params() and flat_params() do not recompute
        self._params = self._build_params()

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def params(self) -> list[dict]:
        """
        Return structured RTK parameters, one dict per triangle.

        Each dict has the form:
            {
                "R": {"init": float, "bounds": (float, float)},
                "T": {"init": float, "bounds": (float, float)},  # hours
                "K": {"init": float, "bounds": (float, float)},  # dimensionless ratio, can exceed 1.0
            }

        R values are scaled so inits sum to a physically reasonable total RDII
        fraction (<= 1.0). They do not need to sum to exactly 1.0.
        """
        return self._params

    def flat_params(self) -> tuple[list[float], list[tuple[float, float]]]:
        """
        Return parameters in the flat format expected by scipy.optimize.minimize.

        Parameter order within x0: [R1, T1, K1, R2, T2, K2, ...]

        Returns
        -------
        x0 : list of float
            Initial parameter vector, length 3 * n_triangles.
        bounds : list of (float, float)
            Bound pairs aligned with x0.

        Example
        -------
        from scipy.optimize import minimize, LinearConstraint
        import numpy as np

        rtk = RTKInitializer(n_triangles=3, area_acres=75_000)
        x0, bounds = rtk.flat_params()

        # Enforce sum(R) <= 1.0 — R values are at indices 0, 3, 6, ...
        A = np.zeros((1, len(x0)))
        A[0, [i * 3 for i in range(rtk.n_triangles)]] = 1.0
        sum_r_constraint = LinearConstraint(A, lb=0.0, ub=1.0)

        # SLSQP supports both bounds and constraints; L-BFGS-B does not
        result = minimize(objective, x0, bounds=bounds,
                          constraints=sum_r_constraint, method='SLSQP')
        """
        x0, bounds = [], []
        for p in self._params:
            x0     += [p["R"]["init"],   p["T"]["init"],   p["K"]["init"]]
            bounds += [p["R"]["bounds"], p["T"]["bounds"], p["K"]["bounds"]]
        return x0, bounds

    def print_params(self) -> None:
        """Print a formatted summary table of parameters to stdout."""
        print(f"\n--- {self.area_acres:,.0f} ac  |  {self.n_triangles} triangle(s) ---")
        hdr = (
            f"  {'#':<4} {'R init':>8}  {'R bounds':>18}"
            f"  {'T init (hr)':>12}  {'T bounds (hr)':>24}"
            f"  {'K init':>8}  {'K bounds':>18}"
        )
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for i, p in enumerate(self._params):
            r, t, k = p["R"], p["T"], p["K"]
            print(
                f"  {i+1:<4}"
                f" {r['init']:>8.4f}  ({r['bounds'][0]:.4f}, {r['bounds'][1]:.4f})"
                f" {t['init']:>12.3f}  ({t['bounds'][0]:.2f}, {t['bounds'][1]:.1f})"
                f" {k['init']:>8.4f}  ({k['bounds'][0]:.4f}, {k['bounds'][1]:.4f})"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _blend_regimes(self) -> dict:
        """
        Return a blended regime anchor by interpolating between the two regimes
        that bracket self.area_acres in log-area space.

        Log-space interpolation is used because catchment area spans five orders
        of magnitude. Linear blending would give near-identical results to the
        large-end regime for most of the range.

        Areas outside the defined regime range are clamped to the nearest endpoint
        rather than extrapolated — extrapolating T in log space would quickly
        produce physically unreasonable values.
        """
        log_a = math.log(self.area_acres)

        # Clamp to the defined regime range rather than extrapolating
        if log_a <= self._LOG_AREAS[0]:
            return self.REGIMES[0]
        if log_a >= self._LOG_AREAS[-1]:
            return self.REGIMES[-1]

        # Find the two regimes that bracket this area
        for i in range(len(self.REGIMES) - 1):
            if self._LOG_AREAS[i] <= log_a <= self._LOG_AREAS[i + 1]:
                lo, hi = self.REGIMES[i], self.REGIMES[i + 1]

                # Fractional position between the two regimes in log-area space
                t = (log_a - self._LOG_AREAS[i]) / (self._LOG_AREAS[i + 1] - self._LOG_AREAS[i])

                blended = {
                    "area_ref": self.area_acres,
                    "name": f"blended({lo['name']}-{hi['name']})",
                }

                for key in ("fast", "slow"):
                    blended[key] = {}
                    # R_init and K: dimensionless — linear blend is appropriate.
                    # R bounds are not stored in regimes; they are applied uniformly
                    # in _build_params so any triangle can carry the full response.
                    for param in ("R_init", "K_init", "K_lo", "K_hi"):
                        blended[key][param] = (
                            lo[key][param] + t * (hi[key][param] - lo[key][param])
                        )
                    # T: spans orders of magnitude across regimes — blend in log space
                    for param in ("T_init", "T_lo", "T_hi"):
                        blended[key][param] = math.exp(
                            math.log(lo[key][param])
                            + t * (math.log(hi[key][param]) - math.log(lo[key][param]))
                        )

                return blended

        return self.REGIMES[-1]  # unreachable but satisfies static analysis

    @staticmethod
    def _log_interp(a: float, b: float, t: float) -> float:
        """Interpolate in log space; used for T values spanning orders of magnitude."""
        return math.exp(math.log(a) + t * (math.log(b) - math.log(a)))

    @staticmethod
    def _lin_interp(a: float, b: float, t: float) -> float:
        """Linear interpolate; used for dimensionless R and K values."""
        return a + t * (b - a)

    def _build_params(self) -> list[dict]:
        """
        Construct the full parameter list for all triangles.
        Called once at construction and cached in self._params.
        """
        regime = self._blend_regimes()
        fast, slow = regime["fast"], regime["slow"]

        params = []
        for i in range(self.n_triangles):
            # Fractional position along fast->slow spectrum.
            # With n=1, t=0 gives the fast anchor — captures the dominant
            # direct inflow response.
            t = i / max(self.n_triangles - 1, 1)

            # R: init interpolated between fast and slow anchors; bounds are
            # uniform (0.001, 1.0) for all triangles regardless of regime so
            # the optimizer is free to assign all RDII to any single triangle.
            # The physical constraint sum(R) <= 1.0 must be enforced externally
            # as a scipy LinearConstraint — bounds alone cannot enforce a sum.
            R_init = self._lin_interp(fast["R_init"], slow["R_init"], t)
            R_lo   = 0.001  # small floor prevents optimizer zeroing a triangle
            R_hi   = 1.0

            # T: log — spans orders of magnitude across fast->slow spectrum
            T_init = self._log_interp(fast["T_init"], slow["T_init"], t)
            T_lo   = self._log_interp(fast["T_lo"],   slow["T_lo"],   t)
            T_hi   = self._log_interp(fast["T_hi"],   slow["T_hi"],   t)

            # K: linear — recession time ratio, no upper bound.
            # K = 1.0 is a symmetric triangle; K > 1 means recession is longer
            # than rise, which is physically normal for infiltration components.
            # Lower bound enforced at K_MIN to prevent zero-duration recession.
            K_init = self._lin_interp(fast["K_init"], slow["K_init"], t)
            K_lo   = max(self._lin_interp(fast["K_lo"], slow["K_lo"], t), self.K_MIN)
            K_hi   = self._lin_interp(fast["K_hi"],   slow["K_hi"],   t)

            params.append({
                "R": {"init": round(R_init, 4), "bounds": (round(R_lo, 4), round(R_hi, 4))},
                "T": {"init": round(T_init, 3), "bounds": (round(T_lo, 3), round(T_hi, 3))},
                "K": {"init": round(K_init, 4), "bounds": (round(K_lo, 4), round(K_hi, 4))},
            })

        # Scale R inits so they sum to 0.20 (20% of rainfall as RDII) as a
        # conservative starting point — the optimizer will adjust from here.
        # Bounds are set to (0.001, 1.0) for every triangle so the optimizer
        # is free to assign all RDII to any single component if the data supports it.
        r_sum = sum(p["R"]["init"] for p in params)
        target_r_sum = 0.20
        r_scale = target_r_sum / r_sum
        for p in params:
            p["R"]["init"] = round(p["R"]["init"] * r_scale, 4)

        return params
