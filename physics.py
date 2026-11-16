"""Physics-specific analysis methods for Simulation.

PhysicsMixin has no dependency on the core: it only uses the public
BaseSimulation API through ``self`` (history, config, outputs, ...).
Combined with BaseSimulation into ``Simulation`` in ``__init__.py``; it
comes first in the MRO, so it can add methods and override core hooks.
"""
import numpy as np


class PhysicsMixin:

    # --- core hook overrides ----------------------------------------------

    def _postprocess_history(self, df):
        """Add v_rms (from KE) and B_rms (from ME), using E = 1/2 X_rms^2 V.

        Best effort: each column is skipped silently if it can't be computed.
        """
        df = super()._postprocess_history(df)
        for energy, rms in (("KE", "v_rms"), ("ME", "B_rms")):
            try:
                if energy in df.columns and rms not in df.columns:
                    df[rms] = np.sqrt(2.0 * df[energy] / self._domain_volume())
            except Exception:
                pass
        return df

    # --- plots / analyses ---------------------------------------------------

    def plot_energy_history(self, ax=None):
        """Example template: kinetic and magnetic energy vs time."""
        import matplotlib.pyplot as plt
        h = self.history
        if ax is None:
            _, ax = plt.subplots()
        for col in ("KE", "ME"):
            if col in h.columns:
                ax.plot(h["time"], h[col], label=col)
        ax.set_xlabel("time")
        ax.set_ylabel("energy")
        ax.legend()
        return ax
