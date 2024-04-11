from copy import deepcopy

import numpy as np
from eryn.moves import DistributionGenerateRJ, GroupStretchMove, MultipleTryMove
from eryn.prior import ProbDistContainer
from numba import njit, prange
from scipy.spatial.distance import cdist


class GroupMoveRKHS(GroupStretchMove):
    def __init__(self, dist_measure="norm", **kwargs):
        super(GroupMoveRKHS, self).__init__(**kwargs)
        if dist_measure == "norm":
            self.idx_reference = None
        elif dist_measure == "beta":
            self.idx_reference = 0
        elif dist_measure == "tau":
            self.idx_reference = 1
        else:
            raise ValueError(f"Incorrect value {dist_measure} for dist_measure")

    def setup_friends(self, branches):
        self.coords_friends = branches["components"].coords[branches["components"].inds]
        if self.idx_reference is None:
            self.reference_values = self.coords_friends
        else:
            self.reference_values = self.coords_friends[:, self.idx_reference]

    @staticmethod
    @njit(cache=True, fastmath=True, parallel=True)
    def _random_nearest_friends(dist, nfriends, inds_choice):
        n_idx = len(dist)
        keep = np.empty(n_idx, dtype=np.int64)

        for i in prange(n_idx):
            dist_row = dist[i]
            partitioned_indices = np.argpartition(dist_row, nfriends)[:nfriends]
            sorted_partitioned_row = np.argsort(dist_row[partitioned_indices])
            sorted_partitioned_indices = partitioned_indices[sorted_partitioned_row]
            idx = inds_choice[i]
            keep[i] = sorted_partitioned_indices[idx]
        return keep

    def find_friends(self, name, s, s_inds):
        """For each parameter, assign a random friend from the nfriends closest ones
        in terms of distance."""
        friends = np.zeros_like(s)
        coords_here = s[s_inds]
        n_idx = len(coords_here)

        if self.idx_reference is None:
            dist = cdist(coords_here, self.coords_friends, "sqeuclidean")
        else:
            dist = cdist(
                coords_here[:, self.idx_reference, None],
                self.coords_friends[:, self.idx_reference, None],
                "sqeuclidean",
            )

        inds_choice = np.random.randint(0, self.nfriends, size=n_idx)
        keep = self._random_nearest_friends(dist, self.nfriends, inds_choice)
        friends[s_inds] = self.coords_friends[keep]

        return friends


class RJMoveRKHS(DistributionGenerateRJ):
    def __init__(self, priors, *args, **kwargs):
        self.priors = priors
        generate_dist = {  # unused; only for compatibility
            "components": ProbDistContainer({(0, 1): priors})
        }
        super(RJMoveRKHS, self).__init__(generate_dist, *args, **kwargs)

    @staticmethod
    @njit(cache=True, fastmath=True)
    def get_model_change_proposal(
        inds, change, nleaves_min, nleaves_max, nleaves, seed
    ):
        """Helper function for changing the model count by 1.

        This helper function works with nested models where you want to add or remove
        one leaf at a time.

        Args:
            inds (np.ndarray): ``inds`` values for this specific branch with shape
                ``(ntemps, nwalkers, nleaves_max)``.
            random (object): Current random state of the sampler.
            nleaves_min (int): Minimum allowable leaf count for this branch.
            nleaves_max (int): Maximum allowable leaf count for this branch.

        Returns:
            inds_birth_array (np.ndarray), inds_death_array (np.ndarray):
                    The indexing information is a 2D array with shape ``(number changing, 3)``.
                    The length 3 is the index into each of the ``(ntemps, nwalkers, nleaves_max)``.

        """
        np.random.seed(seed)
        ntemps, nwalkers = inds.shape[:2]

        # fix edge cases
        change = (
            change * ((nleaves != nleaves_min) & (nleaves != nleaves_max))
            + (+1) * (nleaves == nleaves_min)
            + (-1) * (nleaves == nleaves_max)
        )

        # setup storage for this information
        num_increases = np.sum(change == +1)
        num_decreases = np.sum(change == -1)
        inds_birth_array = np.zeros((num_increases, 3), dtype=np.int64)
        inds_death_array = np.zeros((num_decreases, 3), dtype=np.int64)

        increase_i = 0
        decrease_i = 0
        for t in range(ntemps):
            for w in range(nwalkers):
                # check if add or remove
                change_tw = change[t][w]
                # inds array from specific walker
                inds_tw = inds[t][w]

                # adding
                if change_tw == +1:
                    # find where leaves are not currently used
                    inds_false = np.where(inds_tw == False)[0]  # noqa: E712
                    # decide which spot to add
                    ind_change = np.random.choice(inds_false)
                    # put in the indexes into inds arrays
                    inds_birth_array[increase_i] = np.array(
                        [t, w, ind_change], dtype=np.int64
                    )
                    # count increases
                    increase_i += 1

                # removing
                elif change_tw == -1:
                    # find which leaves are used
                    inds_true = np.where(inds_tw == True)[0]  # noqa: E712
                    # choose which to remove
                    ind_change = np.random.choice(inds_true)
                    # add indexes into inds
                    inds_death_array[decrease_i] = np.array(
                        [t, w, ind_change], dtype=np.int64
                    )
                    decrease_i += 1
                    # do not care currently about what we do with discarded coords, they just sit in the state

        return inds_birth_array, inds_death_array

    def get_proposal(
        self, all_coords, all_inds, nleaves_min_all, nleaves_max_all, random, **kwargs
    ):
        """Make a proposal

        Args:
            all_coords (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim]. These are the curent
                coordinates for all the walkers.
            all_inds (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max]. These are the boolean
                arrays marking which leaves are currently used within each walker.
            nleaves_min_all (dict): Minimum values of leaf ount for each model. Must have same order as ``all_cords``.
            nleaves_max_all (dict): Maximum values of leaf ount for each model. Must have same order as ``all_cords``.
            random (object): Current random state of the sampler.
            **kwargs (ignored): For modularity.

        Returns:
            tuple: Tuple containing proposal information.
                First entry is the new coordinates as a dictionary with keys
                as ``branch_names`` and values as
                ``double `` np.ndarray[ntemps, nwalkers, nleaves_max, ndim] containing
                proposed coordinates. Second entry is the new ``inds`` array with
                boolean values flipped for added or removed sources. Third entry
                is the factors associated with the
                proposal necessary for detailed balance. This is effectively
                any term in the detailed balance fraction. +log of factors if
                in the numerator. -log of factors if in the denominator.

        """
        # get input information for the components branch
        coords_components = all_coords["components"]
        inds_components = all_inds["components"]
        nleaves_min_components = nleaves_min_all["components"]
        nleaves_max_components = nleaves_max_all["components"]

        # put in base information
        ntemps, nwalkers = coords_components.shape[:2]
        q = deepcopy(all_coords)
        new_inds = deepcopy(all_inds)
        factors = np.zeros((ntemps, nwalkers))

        # skip if no movement allowed
        if nleaves_min_components == nleaves_max_components:
            return q, new_inds, factors

        # choose whether to add or remove
        nleaves = inds_components.sum(axis=-1)
        if self.fix_change is None:
            change = random.choice([-1, +1], size=nleaves.shape)
        else:
            change = np.full(nleaves.shape, self.fix_change)

        # get the inds adjustment information
        # For birth and deaths, each row is the index of (ntemp, nwalker, nleaves_max) that changes
        seed = random.get_state()[1][0]
        inds_birth_array, inds_death_array = self.get_model_change_proposal(
            inds_components,
            change,
            nleaves_min_components,
            nleaves_max_components,
            nleaves,
            seed,
        )

        # adjust deaths from True -> False
        inds_death = tuple(inds_death_array.T)  # multi-index for numpy arrays
        new_inds["components"][inds_death] = False

        # factor is +log q()
        factors[inds_death[:2]] += +1 * self.priors.logpdf_components(
            q["components"][inds_death]
        )

        # adjust births from False -> True
        inds_birth = tuple(inds_birth_array.T)  # multi-index for numpy arrays
        new_inds["components"][inds_birth] = True

        # add coordinates for new leaves
        num_inds_change = len(inds_birth[0])
        q["components"][inds_birth] = self.priors.rvs(num_inds_change)

        # factor is -log q()
        factors[inds_birth[:2]] += -1 * self.priors.logpdf_components(
            q["components"][inds_birth]
        )

        return q, new_inds, factors


# CAMBIO: Adaptar la llamada a get_model_change_proposal para usar nuestra función con njit
class MultipleTryMoveRJRKHS(MultipleTryMove):
    def get_proposal(
        self,
        branches_coords,
        branches_inds,
        nleaves_min_all,
        nleaves_max_all,
        random,
        **kwargs,
    ):
        """Make a proposal

        Args:
            all_coords (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max, ndim]. These are the curent
                coordinates for all the walkers.
            all_inds (dict): Keys are ``branch_names``. Values are
                np.ndarray[ntemps, nwalkers, nleaves_max]. These are the boolean
                arrays marking which leaves are currently used within each walker.
            nleaves_min_all (list): Minimum values of leaf ount for each model. Must have same order as ``all_cords``.
            nleaves_max_all (list): Maximum values of leaf ount for each model. Must have same order as ``all_cords``.
            random (object): Current random state of the sampler.
            **kwargs (ignored): For modularity.

        Returns:
            tuple: Tuple containing proposal information.
                First entry is the new coordinates as a dictionary with keys
                as ``branch_names`` and values as
                ``double `` np.ndarray[ntemps, nwalkers, nleaves_max, ndim] containing
                proposed coordinates. Second entry is the new ``inds`` array with
                boolean values flipped for added or removed sources. Third entry
                is the factors associated with the
                proposal necessary for detailed balance. This is effectively
                any term in the detailed balance fraction. +log of factors if
                in the numerator. -log of factors if in the denominator.

        """

        if len(list(branches_coords.keys())) > 1:
            raise ValueError("Can only propose change to one model at a time with MT.")

        # get main key
        key_in = list(branches_coords.keys())[0]
        self.key_in = key_in

        if branches_inds is None:
            raise ValueError("In MT RJ proposal, branches_inds cannot be None.")

        ntemps, nwalkers, nleaves_max, ndim = branches_coords[key_in].shape

        # get temperature information
        betas_here = np.repeat(
            self.temperature_control.betas[:, None], nwalkers, axis=-1
        ).flatten()

        # current Likelihood and prior information
        ll_here = self.current_state.log_like.flatten()
        lp_here = self.current_state.log_prior.flatten()

        # do rj setup
        assert len(nleaves_min_all) == 1 and len(nleaves_max_all) == 1
        nleaves_min = nleaves_min_all[key_in]
        nleaves_max = nleaves_max_all[key_in]

        if nleaves_min == nleaves_max:
            raise ValueError("MT RJ proposal requires that nleaves_min != nleaves_max.")
        elif nleaves_min > nleaves_max:
            raise ValueError("nleaves_min is greater than nleaves_max. Not allowed.")

        # get the inds adjustment information
        # choose whether to add or remove
        inds_components = branches_inds[key_in]
        nleaves = inds_components.sum(axis=-1)
        if self.fix_change is None:
            change = random.choice([-1, +1], size=nleaves.shape)
        else:
            change = np.full(nleaves.shape, self.fix_change)

        # get the inds adjustment information
        # For birth and deaths, each row is the index of (ntemp, nwalker, nleaves_max) that changes
        seed = random.get_state()[1][0]
        inds_birth_array, inds_death_array = self.get_model_change_proposal(
            inds_components,
            change,
            nleaves_min,
            nleaves_max,
            nleaves,
            seed,
        )

        all_inds_for_change = {"+1": inds_birth_array, "-1": inds_death_array}

        # preparing leaf information for going into the proposal
        inds_leaves_rj = np.zeros(ntemps * nwalkers, dtype=int)
        coords_in = np.zeros((ntemps * nwalkers, ndim))
        inds_reverse_rj = None

        # prepare proposal dictionaries
        new_inds = deepcopy(branches_inds)
        q = deepcopy(branches_coords)
        for change in all_inds_for_change.keys():
            if change not in ["+1", "-1"]:
                raise ValueError("MT RJ is only implemented for +1/-1 moves.")

            # get indicies of changing leaves
            temp_inds = all_inds_for_change[change][:, 0]
            walker_inds = all_inds_for_change[change][:, 1]
            leaf_inds = all_inds_for_change[change][:, 2]

            # leaf index to change
            inds_leaves_rj[temp_inds * nwalkers + walker_inds] = leaf_inds
            coords_in[temp_inds * nwalkers + walker_inds] = branches_coords[key_in][
                (temp_inds, walker_inds, leaf_inds)
            ]

            # adjustment of indices
            new_val = {"+1": True, "-1": False}[change]

            # adjust indices
            new_inds[key_in][(temp_inds, walker_inds, leaf_inds)] = new_val

            if change == "-1":
                # which walkers are removing
                inds_reverse_rj = temp_inds * nwalkers + walker_inds

        # setup reversal coords and inds
        # need to determine Likelihood and prior of removed binaries.
        # this goes into the multiple try proposal as previous ll and lp
        temp_reverse_coords = {}
        temp_reverse_inds = {}

        for key in self.current_state.branches:
            (
                ntemps_tmp,
                nwalkers_tmp,
                nleaves_max_tmp,
                ndim_tmp,
            ) = self.current_state.branches[key].shape

            # coords from reversal
            temp_reverse_coords[key] = self.current_state.branches[key].coords.reshape(
                ntemps_tmp * nwalkers_tmp, nleaves_max_tmp, ndim_tmp
            )[inds_reverse_rj][None, :]

            # which inds array to use
            inds_tmp_here = (
                new_inds[key]
                if key == key_in
                else self.current_state.branches[key].inds
            )
            temp_reverse_inds[key] = inds_tmp_here.reshape(
                ntemps * nwalkers, nleaves_max_tmp
            )[inds_reverse_rj][None, :]

        # calculate information for the reverse
        lp_reverse_here = self.current_model.compute_log_prior_fn(
            temp_reverse_coords, inds=temp_reverse_inds
        )[0]
        ll_reverse_here = self.current_model.compute_log_like_fn(
            temp_reverse_coords, inds=temp_reverse_inds, logp=lp_here
        )[0]

        # fill the here values
        ll_here[inds_reverse_rj] = ll_reverse_here
        lp_here[inds_reverse_rj] = lp_reverse_here

        # get mt proposal
        generated_points, factors = self.get_mt_proposal(
            coords_in,
            random,
            betas=betas_here,
            ll_in=ll_here,
            lp_in=lp_here,
            inds_leaves_rj=inds_leaves_rj,
            inds_reverse_rj=inds_reverse_rj,
        )

        # for reading outside
        self.mt_ll = self.mt_ll.reshape(ntemps, nwalkers)
        self.mt_lp = self.mt_lp.reshape(ntemps, nwalkers)

        # which walkers have information added
        inds_forward_rj = np.delete(np.arange(coords_in.shape[0]), inds_reverse_rj)

        # updated the coordinates
        temp_inds = all_inds_for_change["+1"][:, 0]
        walker_inds = all_inds_for_change["+1"][:, 1]
        leaf_inds = all_inds_for_change["+1"][:, 2]
        q[key_in][(temp_inds, walker_inds, leaf_inds)] = generated_points[
            inds_forward_rj
        ]

        return q, new_inds, factors.reshape(ntemps, nwalkers)


# Cambio: usar nuestro priors all_models_together y sus funciones específicas (logpdf_components, rvs).
# No nos hace falta en este caso pasar los índices.
class MTRJMoveRKHS(MultipleTryMoveRJRKHS, RJMoveRKHS):
    def __init__(self, priors, *args, **kwargs):
        """Perform a reversible-jump multiple try move based on a distribution.

        Distribution must be independent of the current point.

        This is effectively an example of the mutliple try class inheritance structure.

        Args:
            generate_dist (dict): Keys are branch names and values are :class:`ProbDistContainer` objects
                that have ``logpdf`` and ``rvs`` methods. If you
            *args (tuple, optional): Additional arguments to pass to parent classes.
            **kwargs (dict, optional): Keyword arguments passed to parent classes.

        """
        kwargs["rj"] = True
        MultipleTryMoveRJRKHS.__init__(self, **kwargs)
        RJMoveRKHS.__init__(self, priors, *args, **kwargs)

    def special_generate_logpdf(self, generated_coords):
        """Get logpdf of generated coordinates.

        Args:
            generated_coords (np.ndarray): Current coordinates of walkers.

        Returns:
            np.ndarray: logpdf of generated points.
        """
        return self.priors.logpdf_components(generated_coords)

    def special_generate_func(
        self, coords, random, size=1, fill_tuple=None, fill_values=None
    ):
        """Generate samples and calculate the logpdf of their proposal function.

        Args:
            coords (np.ndarray): Current coordinates of walkers.
            random (obj): Random generator.
            *args (tuple, optional): additional arguments passed by overwriting the
                ``get_proposal`` function and passing ``args_generate`` keyword argument.
            size (int, optional): Number of tries to generate.
            fill_tuple (tuple, optional): Length 2 tuple with the indexing of which values to fill
                when generating. Can be used for auxillary proposals or reverse RJ proposals. First index is the index into walkers and the second index is
                the index into the number of tries. (default: ``None``)
            fill_values (np.ndarray): values to fill associated with ``fill_tuple``. Should
                have size ``(len(fill_tuple[0]), ndim)``. (default: ``None``).
            **kwargs (tuple, optional): additional keyword arguments passed by overwriting the
                ``get_proposal`` function and passing ``kwargs_generate`` keyword argument.

        Returns:
            tuple: (generated points, logpdf of generated points).

        """
        nwalkers = coords.shape[0]

        if not isinstance(size, int):
            raise ValueError("size must be an int.")

        generated_coords = self.priors.rvs(size=(nwalkers, size))

        if fill_values is not None:
            generated_coords[fill_tuple] = fill_values

        generated_logpdf = self.special_generate_logpdf(
            generated_coords.reshape(nwalkers * size, -1)
        ).reshape(nwalkers, size)

        return generated_coords, generated_logpdf

    def set_coords_and_inds(self, generated_coords, inds_leaves_rj=None):
        """Setup coordinates for prior and Likelihood

        Args:
            generated_coords (np.ndarray): Generated coordinates with shape ``(number of independent walkers, num_try)``.
            inds_leaves_rj (np.ndarray): Index into each individual walker giving the
                leaf index associated with this proposal. Should only be used if ``self.rj is True``. (default: ``None``)

        Returns:
            dict: Coordinates for Likelihood and Prior.

        """
        coords_in = np.repeat(
            self.current_state.branches[self.key_in].coords.reshape(
                (
                    1,
                    -1,
                )
                + self.current_state.branches[self.key_in].coords.shape[-2:]
            ),
            self.num_try,
            axis=1,
        )
        coords_in[
            (
                np.zeros(coords_in.shape[0], dtype=int),
                np.arange(coords_in.shape[1]),
                np.repeat(inds_leaves_rj, self.num_try),
            )
        ] = generated_coords.reshape(-1, coords_in.shape[-1])
        inds_in = np.repeat(
            self.current_state.branches[self.key_in].inds.reshape(
                (
                    1,
                    -1,
                )
                + self.current_state.branches[self.key_in].inds.shape[-1:]
            ),
            self.num_try,
            axis=1,
        )
        inds_in[
            (
                np.zeros(coords_in.shape[0], dtype=int),
                np.arange(inds_in.shape[1]),
                np.repeat(inds_leaves_rj, self.num_try),
            )
        ] = True

        coords_in_dict = {}
        inds_in_dict = {}
        for key in self.current_state.branches.keys():
            if key == self.key_in:
                coords_in_dict[key] = coords_in
                inds_in_dict[key] = inds_in

            else:
                coords_in_dict[key] = self.current_state.branches[key].coords.reshape(
                    (1, -1) + self.current_state.branches[key].shape[-2:]
                )
                # expand to express multiple tries
                coords_in_dict[key] = np.tile(
                    coords_in_dict[key], (1, self.num_try, 1)
                ).reshape(
                    coords_in_dict[key].shape[0],
                    coords_in_dict[key].shape[1] * self.num_try,
                    coords_in_dict[key].shape[2],
                    coords_in_dict[key].shape[3],
                )
                inds_in_dict[key] = self.current_state.branches[key].inds.reshape(
                    (1, -1) + self.current_state.branches[key].shape[-2:-1]
                )
                # expand to express multiple tries
                inds_in_dict[key] = np.tile(
                    inds_in_dict[key], (1, self.num_try)
                ).reshape(
                    inds_in_dict[key].shape[0],
                    inds_in_dict[key].shape[1] * self.num_try,
                    inds_in_dict[key].shape[2],
                )

        return coords_in_dict, inds_in_dict

    def special_like_func(self, generated_coords, inds_leaves_rj=None, **kwargs):
        """Calculate the Likelihood for sampled points.

        Args:
            generated_coords (np.ndarray): Generated coordinates with shape ``(number of independent walkers, num_try)``.
            inds_leaves_rj (np.ndarray): Index into each individual walker giving the
                leaf index associated with this proposal. Should only be used if ``self.rj is True``. (default: ``None``)

        Returns:
            np.ndarray: Likelihood values with shape ``(generated_coords.shape[0], num_try).``

        """
        coords_in, inds_in = self.set_coords_and_inds(
            generated_coords, inds_leaves_rj=inds_leaves_rj
        )
        ll = self.current_model.compute_log_like_fn(coords_in, inds=inds_in)[0]
        ll = ll[0].reshape(-1, self.num_try)
        return ll

    def special_prior_func(self, generated_coords, inds_leaves_rj=None, **kwargs):
        """Calculate the Prior for sampled points.

        Args:
            generated_coords (np.ndarray): Generated coordinates with shape ``(number of independent walkers, num_try)``.
            inds_leaves_rj (np.ndarray): Index into each individual walker giving the
                leaf index associated with this proposal. Should only be used if ``self.rj is True``. (default: ``None``)

        Returns:
            np.ndarray: Prior values with shape ``(generated_coords.shape[0], num_try).``

        """
        coords_in, inds_in = self.set_coords_and_inds(
            generated_coords, inds_leaves_rj=inds_leaves_rj
        )
        lp = self.current_model.compute_log_prior_fn(coords_in, inds=inds_in)
        return lp.reshape(-1, self.num_try)