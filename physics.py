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

    @staticmethod
    def compare(sims, method, labels=None, linestyles=("-", "--", ":", "-."),
                ax=None, run_legend_loc="lower left", **kwargs):
        """Call the plotting method `method` on each sim, overlaid on one axes.

        `method` is a method name ("evolution_plot") or the function itself
        (Simulation.evolution_plot). Extra kwargs go to every call.
        """
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        if ax is None:
            _, ax = plt.subplots()
        name = method if isinstance(method, str) else method.__name__
        if labels is None:
            labels = [getattr(s, "name", f"sim {i}") for i, s in enumerate(sims)]
        styles = [linestyles[i % len(linestyles)] for i in range(len(sims))]

        for i, (sim, ls) in enumerate(zip(sims, styles)):
            getattr(sim, name)(ax=ax, ls=ls, legend=(i == 0), **kwargs)

        # Keep the quantity legend from the first call, then add a run legend
        quantity_legend = ax.get_legend()
        handles = [Line2D([], [], color="k", ls=ls) for ls in styles]
        ax.legend(handles, labels, loc=run_legend_loc)
        if quantity_legend is not None:
            ax.add_artist(quantity_legend)

    def evolution_plot(self, ax=None, xlim=(300, 5000), ylim=(1e-2, 1e1),
                    colors=("blue", "orange", "green"), ls="-", alpha=1, legend=True):
        """Plots B_rms, v_rms and Clumping factor (Drho) against redshift"""
        import matplotlib.pyplot as plt
        h = self.history
        if ax is None:
            ax = plt.gca()

        def lab(s):
            return s if legend else "_nolegend_"

        ax.loglog(h["redshift"], h["ClumpingFactor"], label=lab(r"clumping factor $D\rho$"), color=colors[0], ls=ls, alpha=alpha)
        ax.loglog(h["redshift"], h["B_rms"], label=lab(r"$B_\mathrm{rms}$"), color=colors[1], ls=ls, alpha=alpha)
        ax.loglog(h["redshift"], h["v_rms"], label=lab(r"$v_\mathrm{rms}$"), color=colors[2], ls=ls, alpha=alpha)
        if legend:
            ax.legend()
        ax.set_xlabel("redshift")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)