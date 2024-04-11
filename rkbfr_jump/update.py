import numpy as np
from eryn.utils import Update


class AdjustStretchScaleCombineMove(Update):
    def __init__(
        self,
        idx_moves=None,
        target_acceptance=0.22,
        supression_factor=0.1,
        max_factor=0.5,
        min_a=1.1,
        verbose=False,
    ):
        if idx_moves is None:
            self.idx_moves = [0]
        else:
            self.idx_moves = idx_moves

        self.target_acceptance = target_acceptance
        self.supression_factor = supression_factor
        self.max_factor = max_factor
        self.min_a = min_a
        self.verbose = verbose

        self.time = 0
        self.previously_accepted = {}

    def __call__(self, iter, last_sample, sampler):
        for idx in self.idx_moves:
            move = sampler.moves[0].moves[idx]
            change = 1.0
            mean_af = 0.0

            if self.time > 0:
                # cold chain -> T=0
                mean_af = np.mean(
                    (move.accepted[0] - self.previously_accepted[idx])
                    / (sampler.backend.iteration - self.previous_iter)
                )

                if mean_af > self.target_acceptance:
                    factor = self.supression_factor * (mean_af / self.target_acceptance)
                    if factor > self.max_factor:
                        factor = self.max_factor
                    change = 1 + factor

                else:
                    factor = self.supression_factor * (self.target_acceptance / mean_af)
                    if factor > self.max_factor:
                        factor = self.max_factor
                    change = 1 - factor

                if move.a * change > self.min_a:
                    move.a *= change
                else:
                    move.a = self.min_a

            self.previously_accepted[idx] = move.accepted[0].copy()

            if self.verbose:
                print(
                    f"[{move.__class__.__name__}] iter={sampler.backend.iteration} a={move.a:.2f}"
                )

        self.time += 1
        self.previous_iter = sampler.backend.iteration