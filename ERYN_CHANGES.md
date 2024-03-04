(ALREADY FIXED IN DEV BRANCH IN GITHUB --> commit #a8466ad)

1. Deleted "model_0" from priors.setter when input is a dictionary and the item has the attr logpdf (line 670 in ensemble.py):
   "self._priors[key] = test"
2. Corrected group move calculations in iterations (group.py, line 137):
   "if self.iter == 0 or self.iter % self.n_iter_update == 0"
3. Applied changes to "get_a_sample" and "get_last_sample" in backend.py
4. Applied changes to all_moves_tmp, initial_state, temperature control and update_fn in ensemble.py
5. Applied changes to multipletry.py
6. Applied changes to prior.py