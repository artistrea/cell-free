from dataclasses import dataclass
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
BOLTZMANN_CONSTANT = 1.381e-23


def fspl(dist: float, freq: float):
    return (
        20 * np.log10(dist)
        + 20 * np.log10(freq)
        + 20 * np.log10(4*np.pi/3e8)
    )


def noise_for_equipment(temperature: float, noise_fig_db: float, bw: float):
    return 10 * np.log10(BOLTZMANN_CONSTANT * temperature * bw) + noise_fig_db


def pow_from_snr_and_noise(snr: float, noise: float) -> float:
    """(dB, dB) -> dB"""
    return snr + noise


@dataclass(
    frozen=True
)
class CellFreeSimulationResults:
    cell_free_ref: "CellFree"
    num_geometries: int
    num_blocks_per_geometry: int
    num_ue: int
    statistical_sinr: np.ndarray
    instantaneous_sinr: np.ndarray
    bw: float

    @property
    def achievable_rate_statistical_knowledge(self):
        assert self.statistical_sinr.shape == (
            self.num_geometries, self.num_ue
        )
        return self.bw * np.log2(1 + self.statistical_sinr)

    @property
    def achievable_rate_instantaneous_knowledge(self):
        assert len(self.instantaneous_sinr.shape) == 3
        assert self.instantaneous_sinr.shape == (
            self.num_geometries, self.num_blocks_per_geometry, self.num_ue
        )
        return self.bw * np.mean(np.log2(1 + self.instantaneous_sinr), axis=1)


@dataclass(
    frozen=True
)
class CellFree:
    fc: float  # [Hz]
    bw: float  # [Hz]
    noise_fig_db: float  # [dB]
    ap_h: float  # [m]
    ue_h: float  # [m]
    pilot_pow_db: float  # [dB]
    downlink_pow_db: float  # [dB]

    noise_temperature: float  # [K]
    area_dimensions: tuple[float, float]  # ([m], [m])

    # per drop simulation
    num_ue: int
    tau_cf: int  # number of samples for channel estimation
    num_ap: int

    close_in_fs_exponent: float = 2.8  # 2 means fspl
    shadowing_std_db: float = 8  # [dB]

    @property
    def noise_pow_db(self) -> float:
        return noise_for_equipment(
            self.noise_temperature, self.noise_fig_db, self.bw
        )

    @property
    def noise_pow_lin(self) -> float:
        return 10 ** (self.noise_pow_db/10)

    @property
    def noise_std_lin_per_component(self) -> float:
        return np.sqrt(self.noise_pow_lin / 2)

    @property
    def M(self) -> int:
        # per slide 43, each drop has M APs
        return self.num_ap

    @property
    def K(self) -> int:
        # per slide 43, each drop has K UEs
        return self.num_ue

    @property
    def pilot_pow_lin(self) -> float:
        return 10**(self.pilot_pow_db/10)

    @property
    def downlink_pow_lin(self) -> float:
        return 10**(self.downlink_pow_db/10)

    def __post_init__(self):
        assert self.tau_cf >= self.num_ue

    def simulate(
        self, rng: np.random.Generator,
        geometry_instances: int, drops_per_geom: int
    ) -> CellFreeSimulationResults:
        statistical_sinr = np.zeros((geometry_instances, self.K))
        instantaneous_sinr = np.zeros((geometry_instances, drops_per_geom, self.K))

        for i in tqdm(list(range(geometry_instances))):
            pos_ue_x = rng.uniform(0, self.area_dimensions[0], size=self.K)
            pos_ue_y = rng.uniform(0, self.area_dimensions[1], size=self.K)

            pos_ap_x = rng.uniform(0, self.area_dimensions[0], size=self.M)
            pos_ap_y = rng.uniform(0, self.area_dimensions[1], size=self.M)

            # (M, K)
            ap_to_ue_dist = np.sqrt(
                (pos_ap_x[:, None] - pos_ue_x[None, :])**2
                + (pos_ap_y[:, None] - pos_ue_y[None, :])**2
                + (self.ue_h - self.ap_h)**2
            )
            ap_to_ue_dist = np.maximum(ap_to_ue_dist, 1.0)

            shadowing_db = rng.normal(0, self.shadowing_std_db, size=(self.M, self.K))
            ap_to_ue_pathloss_db = (
                fspl(1., self.fc)
                + self.close_in_fs_exponent * 10 * np.log10(ap_to_ue_dist)
                + shadowing_db
            )
            ap_to_ue_pathloss_lin = 10**(ap_to_ue_pathloss_db / 10)
            # NOTE: kinda Eq. (30)
            # FIXME: talk to professor about unreliable notation
            # Omega is gain, not path loss. So the slides are wrong
            Omega = 1/ap_to_ue_pathloss_lin

            # SINR and achievable rate based on statistics only:
            # Eq. (35)
            C = (
                np.sqrt(self.tau_cf * self.pilot_pow_lin)  * Omega
                / (
                    self.tau_cf * self.pilot_pow_lin * Omega
                    + self.noise_pow_lin
                )
            )

            # Eq. (36)
            # NOTE: this commented implementation is according to the professor
            # equation, but it seems wrong
            # The correct one is from Ngo's article Eq. (8)
            # Gamma = np.sqrt(self.tau_cf * self.downlink_pow_lin) * Omega * C
            Gamma = np.sqrt(self.tau_cf * self.pilot_pow_lin) * Omega * C
            eta = (np.sum(Gamma, axis=-1, keepdims=True))**-1

            # Eq. (37)
            # shape = (K,)
            sinr_statistic_only = self.downlink_pow_lin * np.sum(
                np.sqrt(eta) * Gamma, axis=0
            )**2 / (
                self.downlink_pow_lin * np.sum(Omega * np.sum(
                    eta * Gamma, axis=1, keepdims=True
                ), axis=0) + self.noise_pow_lin
            )
            assert sinr_statistic_only.shape == (self.K,)
            statistical_sinr[i] = sinr_statistic_only

            # NOTE: rayleigh assumed
            H_all_drops = (
                rng.normal(0, 1/np.sqrt(2),
                           size=(drops_per_geom, self.M, self.K))
                + 1j * rng.normal(0, 1/np.sqrt(2),
                                  size=(drops_per_geom, self.M, self.K))
            )
            noise_all_drops = (
                rng.normal(0, self.noise_std_lin_per_component,
                           size=(drops_per_geom, self.M, self.K))
                + 1j * rng.normal(0, self.noise_std_lin_per_component,
                                  size=(drops_per_geom, self.M, self.K))
            )
            for j in range(drops_per_geom):
                H = H_all_drops[j]
                # UL Training (Chan. Estimation)
                G = np.sqrt(Omega) * H
                # NOTE: since we assume tau_cf >= K, there's no pilot
                # contamination, and we don't even need to define each pilot
                # we can go straight to projection
                # Eq. (33)
                noise = noise_all_drops[j]
                Y_pilot_proj = (
                    np.sqrt(self.tau_cf * self.pilot_pow_lin) * G
                    + noise
                )
                # Eq. (34)
                G_est = C * Y_pilot_proj

                # Eq. (21)
                # shape == (M, K, K)
                # (signal from AP, meant to user k', arriving at user k)
                signal_ap_to_ue = (
                    np.sqrt(eta)[..., None] * G[..., None]
                    * np.conj(G_est)[:, None, :]
                )
                pow_at_ue_meant_for_ue = np.abs(np.sum(
                    signal_ap_to_ue, axis=0
                ))**2
                desired_at_ue = np.diag(pow_at_ue_meant_for_ue)
                assert desired_at_ue.shape == (self.K,)
                # we can only do it this way because professor's equation is not
                # done according to the article. The sum axis is swapped.
                interf_at_ue = (
                    np.sum(pow_at_ue_meant_for_ue, axis=1) - desired_at_ue
                )
                sinr = (
                    self.downlink_pow_lin * desired_at_ue
                    / (self.downlink_pow_lin * interf_at_ue
                        + self.noise_pow_lin)
                )
                assert sinr.shape == (self.K,)
                instantaneous_sinr[i][j] = sinr

        return CellFreeSimulationResults(
            cell_free_ref=self,
            num_geometries=geometry_instances,
            num_blocks_per_geometry=drops_per_geom,
            num_ue=self.K,
            bw=self.bw,
            statistical_sinr=statistical_sinr,
            instantaneous_sinr=instantaneous_sinr,
        )


def ecdf(data: np.ndarray):
    """Empirical CDF over all values in `data` (any shape, flattened)."""
    x = np.sort(data.ravel())
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y

def plot_scenarios(results: list[tuple[str, "CellFreeSimulationResults"]]):
    """
    Plota as CDFs de SINR e Taxa Alcançável.

    Linha pontilhada: ECSI (Statistical CSI)
    Linha contínua: PCSI (Instantaneous CSI)
    """

    fig, (ax_sinr, ax_rate) = plt.subplots(1, 2, figsize=(14, 5))

    for label, res in results:
        # ----- SINR -----
        x, y = ecdf(10 * np.log10(res.statistical_sinr))
        (line,) = ax_sinr.plot(
            x,
            y,
            linestyle=":",
            label=f"ECSI {label}",
        )

        x, y = ecdf(10 * np.log10(res.instantaneous_sinr))
        # x, y = ecdf(10 * np.log10(np.mean(res.instantaneous_sinr, axis=1)))
        ax_sinr.plot(
            x,
            y,
            linestyle="-",
            color=line.get_color(),
            label=f"PCSI {label}",
        )

        # ----- Achievable Rate -----
        x, y = ecdf(res.achievable_rate_statistical_knowledge / 1e6)
        (line,) = ax_rate.plot(
            x,
            y,
            linestyle=":",
            label=f"ECSI {label}",
        )

        x, y = ecdf(res.achievable_rate_instantaneous_knowledge / 1e6)
        ax_rate.plot(
            x,
            y,
            linestyle="-",
            color=line.get_color(),
            label=f"PCSI {label}",
        )

    ax_sinr.set(
        # title="CDF da SINR",
        xlabel="SINR [dB]",
        ylabel="ECDF",
    )
    ax_sinr.grid(True)
    ax_sinr.legend()

    ax_rate.set(
        # title="CDF da Taxa Alcançável",
        xlabel="Taxa Alcançável [Mbit/s]",
        ylabel="ECDF",
    )
    ax_rate.grid(True)
    ax_rate.legend()

    fig.tight_layout()

    return fig


def main():
    basic_params = {
        "fc": 3e9, "bw": 20e6,
        "noise_fig_db": 9,
        "noise_temperature": 296.15,
        "area_dimensions": (1e3, 1e3),
        "ap_h": 15, "ue_h": 1.65,
        "num_ue": 10,
        "tau_cf": 10,
        "num_ap": 100,

        "pilot_pow_db": 10 * np.log10(0.1),
        "downlink_pow_db": 10 * np.log10(0.2),
    }

    all_params = [
        {**basic_params, 'num_ap': x} for x in [100, 150, 200]
    ]
    results = []
    for par in all_params:
        cell_free_sim = CellFree(**par)
        rng = np.random.default_rng(12331)
        res = cell_free_sim.simulate(rng, 300, 100)
        label = f"M={par['num_ap']}"
        results.append((label, res))

    plot_scenarios(results)

    plt.show()


if __name__ == "__main__":
    main()
