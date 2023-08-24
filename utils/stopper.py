from ray.tune import Stopper
from collections import defaultdict
from typing import Dict
import numpy as np


class EarlyPlateauStopper(Stopper):
    """Early stop the experiment when a metric plateaued across trials.

    Stops the entire experiment when the metric has plateaued
    for more than the given amount of iterations specified in
    the patience parameter or when the max number of iterations 
    are reached.

    Args:
        metric: The metric to be monitored.
        std: The minimal standard deviation after which
            the tuning process has to stop.
        top: The number of best models to consider.
        min_iter: The minimum number of iterations required.
        max_iter: The max number of iterations allowed.
        mode: The mode to select the top results.
            Can either be 'min' or 'max'.
        patience: Number of epochs to wait for
            a change in the top models.

    Raises:
        ValueError: If the mode parameter is not 'min' nor 'max'.
        ValueError: If the top parameter is not
            a strictly positive integer.
        ValueError: If the patience parameter is not
            a strictly positive integer.
    """

    def __init__(
        self,
        metric: str,
        top: int = 10,
        min_iter: int = 0,
        max_iter: int = 100,
        min_trials: int = 1,
        mode: str = 'min',
        patience: int = 0,
    ):
        if mode not in ('min', 'max'):
            raise ValueError('The mode parameter can only be either min or max.')
        if not isinstance(top, int) or top < 1:
            raise ValueError(
                'Top results to consider must be'
                ' a strictly positive integer.'
            )
        if not isinstance(patience, int) or patience < 0:
            raise ValueError('Patience must be a strictly positive integer.')
        if not isinstance(min_trials, int) or min_trials < 0:
            raise ValueError(
                'The min number of trials before evaluating must be a'
                ' strictly positive integer.'
            )
        self._mode = mode
        self._metric = metric
        self._iter = defaultdict(lambda: 0)
        self._history = defaultdict(lambda: [])
        self._min_trials = min_trials
        self._min_iter = min_iter
        self._max_iter = max_iter
        self._patience = patience
        self._iterations = 0
        self._top = top
        self._top_values = []

    def __call__(self, trial_id, result):
        
        # Check for passing max iterations
        self._iter[trial_id] += 1
        trial_i = self._iter[trial_id]
        
        # Check for passing min iterations
        if trial_i < self._min_iter:
            return False

        # Check for passing max iterations
        if trial_i >= self._max_iter:
            return True

        # Add current value to history
        self._history[trial_i].append(result[self._metric])

        return self.stop_all()

    def stop_all(self):
        best_value = float('inf') if self._mode == 'min' else -float('inf')
        iter_since_improvment = 0

        # Iterate over history
        for i in self._history:
            history_i_arr = self._history[i]

            # Make sure at least min_trials have completed this 
            # iteration before evaluating
            if len(history_i_arr) >= self._min_trials:

                # Get mean of n best trials
                if self._mode == 'min':
                    n_best = sorted(history_i_arr)[:self._top]
                else:
                    n_best = sorted(history_i_arr)[-self._top:]
                value_i = np.mean(n_best)

                is_improved = (self._mode == 'min' and value_i < best_value) or \
                              (self._mode == 'max' and value_i > best_value)

                # If mean has improved, reset counter. Else increase        
                if is_improved:
                    best_value = value_i
                    iter_since_improvment = 0
                else:
                    iter_since_improvment += 1

                # Stop if no improvment to 'n best means' in 'patience' steps
                if iter_since_improvment > self._patience:
                    return True

        return False
