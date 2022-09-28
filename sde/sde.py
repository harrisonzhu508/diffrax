import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom


def normal_logprob(y, loc, scale):
    return -0.5 * ((y - loc) / scale) ** 2 - jnp.log(scale) - 0.5 * jnp.log(2 * jnp.pi)


def normal_kl_divergence(loc1, scale1, loc2, scale2):
    var_ratio = (scale1 / scale2) ** 2
    t1 = ((loc2 - loc1) / scale2) ** 2
    return 0.5 * (var_ratio + t1 - 1 - jnp.log(var_ratio))


class Encoder(eqx.Module):

    gru: eqx.nn.GRUCell
    linear: eqx.nn.Linear

    def __init__(self, input_size, hidden_size, output_size, *, key) -> None:
        super().__init__()
        gru_key, linear_key = jrandom.split(key)
        self.gru = eqx.nn.GRUCell(
            input_size=input_size, hidden_size=hidden_size, key=gru_key
        )
        self.linear = eqx.nn.Linear(
            in_features=hidden_size, out_features=output_size, key=linear_key
        )

    def __call__(self, x):
        def scan_fn(state, input):
            new_state = self.gru(input, state)
            return new_state, new_state

        init_state = jnp.zeros(self.gru.hidden_size)
        _, out = jax.lax.scan(scan_fn, init_state, x)
        out = jax.vmap(self.linear)(out)
        return out


class DriftPosterior(eqx.Module):

    net: eqx.nn.MLP

    def __init__(self, latent_size, context_size, hidden_size, *, key) -> None:

        self.net = eqx.nn.MLP(
            # in_size=latent_size + context_size,
            in_size=3,
            width_size=hidden_size,
            out_size=latent_size,
            depth=2,
            activation=jax.nn.softplus,
            key=key,
        )

    def __call__(self, t, y, args):
        if len(jnp.shape(t)) == 0:
            t = jnp.full_like(y, fill_value=t)
        feature = jnp.concatenate([jnp.sin(t), jnp.cos(t), y], axis=-1)
        # return self.net(y)
        return self.net(feature)


class DriftPrior(eqx.Module):

    net: eqx.nn.MLP

    def __init__(self, latent_size, hidden_size, *, key):
        self.net = eqx.nn.MLP(
            # in_size=latent_size,
            in_size=1,
            width_size=hidden_size,
            out_size=latent_size,
            depth=2,
            activation=jax.nn.softplus,
            key=key,
        )

    def __call__(self, t, y, args):
        # if len(jnp.shape(t)) == 0:
        #     t = jnp.full_like(y, fill_value=t)
        # feature = jnp.concatenate([jnp.sin(t), jnp.cos(t), y])
        # return self.net(feature)
        return self.net(y)


class Diffusion(eqx.Module):

    # nets: List[eqx.nn.MLP]
    sigma: jnp.ndarray

    def __init__(self, latent_size, hidden_size, *, key):

        # keys = jrandom.split(key, latent_size)
        # self.nets = [
        #     eqx.nn.MLP(
        #         in_size=1,
        #         width_size=hidden_size,
        #         out_size=1,
        #         depth=1,
        #         activation=jax.nn.softplus,
        #         final_activation=jax.nn.sigmoid,
        #         key=i_key,
        #     )
        #     for i_key in keys
        # ]
        self.sigma = jnp.ones(shape=(latent_size,))

    def __call__(self, t, y, args):
        # y = jnp.split(y, indices_or_sections=len(self.nets))
        # out = [net_i(y_i) for net_i, y_i in zip(self.nets, y)]
        # return jnp.concatenate(out, axis=0)
        return self.sigma


class LatentSDE(eqx.Module):

    encoder: eqx.Module
    posterior_drift: eqx.Module
    prior_drift: eqx.Module
    diffusion: eqx.Module
    qz0_net: eqx.nn.Linear
    projector: eqx.nn.Linear
    pz0_mean: jnp.ndarray
    pz0_logstd: jnp.ndarray
    # qz0_mean: jnp.ndarray
    # qz0_logstd: jnp.ndarray
    t0: float
    t1: float
    latent_size: int

    def __init__(
        self, data_size, latent_size, context_size, hidden_size, t0, t1, *, key
    ) -> None:
        super().__init__()
        self.t0, self.t1 = t0, t1
        self.latent_size = latent_size
        keys = jrandom.split(key, num=6)
        self.encoder = Encoder(
            input_size=data_size,
            hidden_size=hidden_size,
            output_size=context_size,
            key=keys[0],
        )
        self.qz0_net = eqx.nn.Linear(
            context_size, latent_size + latent_size, key=keys[1]
        )

        self.posterior_drift = DriftPosterior(
            latent_size=latent_size,
            context_size=0,
            # context_size=context_size,
            hidden_size=hidden_size,
            key=keys[2],
        )
        self.prior_drift = DriftPrior(
            latent_size=latent_size, hidden_size=hidden_size, key=keys[3]
        )

        self.diffusion = Diffusion(
            latent_size=latent_size, hidden_size=hidden_size, key=keys[4]
        )
        self.projector = eqx.nn.Linear(latent_size, data_size, key=keys[5])
        self.pz0_mean = jnp.zeros(shape=(1, latent_size))
        self.pz0_logstd = jnp.zeros(shape=(1, latent_size))
        # self.qz0_mean = jnp.zeros(shape=(1, latent_size))
        # self.qz0_logstd = jnp.zeros(shape=(1, latent_size))

    def integrate(self, y0, solver, context, key, saveat=None, dt=1e-2):
        """Solving SDE over latent space"""
        bm = diffrax.VirtualBrownianTree(
            t0=self.t0, t1=self.t1, shape=(self.latent_size,), tol=1e-3, key=key
        )

        control_term = diffrax.WeaklyDiagonalControlTerm(self.diffusion, bm)
        posterior_sde = diffrax.MultiTerm(
            diffrax.ODETerm(self.posterior_drift), control_term
        )
        prior_sde = diffrax.MultiTerm(diffrax.ODETerm(self.prior_drift), control_term)

        # get augmented SDEs
        aug_sde, aug_y0 = diffrax.sde_kl_divergence(
            sde1=posterior_sde, sde2=prior_sde, context=context, y0=y0
        )

        sol = diffrax.diffeqsolve(
            aug_sde, solver, t0=self.t0, t1=self.t1, dt0=dt, y0=aug_y0, saveat=saveat
        )
        return sol.ys

    def __call__(self, xs, ts, *, key, ts_saveat=None, num_samples=1):
        """
        This extracts contexts from data via a recurrent neural network (GRU).
        The contexts then are fed to SDE over latent space.
        The function returns the trajectories of models after
        re-projecting from latent space into data space.
        """
        solver = diffrax.Euler()
        if ts_saveat is not None:
            saveat = diffrax.SaveAt(ts=ts_saveat)
        else:
            saveat = None

        def solve(xs, ts, key, saveat):
            eps_key, bm_key = jrandom.split(key)
            ctx = self.encoder(jnp.flip(xs, axis=0))
            ctx = jnp.flip(ctx, axis=0)
            if saveat is None:
                saveat = diffrax.SaveAt(ts=ts)

            # def context(t):
            #     # find the index which is closet to the current time
            #     t_index = jnp.searchsorted(ts, t, side="right")
            #     # return the corresponding context
            #     return ctx[t_index]
            context = None
            qz0_mean, qz0_logstd = jnp.split(
                self.qz0_net(ctx[0]), indices_or_sections=2, axis=-1
            )
            eps = jrandom.normal(key=eps_key, shape=(num_samples, qz0_logstd.shape[0]))
            z0 = qz0_mean + jnp.exp(qz0_logstd) * eps
            bm_keys = jrandom.split(bm_key, num=num_samples)

            def integrate_solver(z0, bm_keys):
                zs, logpq_path = self.integrate(
                    z0, solver=solver, context=context, saveat=saveat, key=bm_keys
                )
                return zs, logpq_path

            zs, logpq_path = jax.vmap(integrate_solver)(z0, bm_keys)

            logpq0 = normal_kl_divergence(
                loc1=qz0_mean,
                scale1=jnp.exp(qz0_logstd),
                loc2=self.pz0_mean,
                scale2=jnp.exp(self.pz0_logstd),
            )
            logpq = logpq0.sum() + jnp.mean(logpq_path[:, -1], axis=0)
            return zs, logpq

        batch_solve = jax.vmap(solve, (0, 0, 0, None))
        keys = jrandom.split(key, num=xs.shape[0])
        zs, logpq = batch_solve(xs, ts, keys, saveat)

        # xs_pred = zs
        # if self.projector is None:
        #     xs_pred = zs
        # else:
        #     xs_pred = jax.vmap(jax.vmap(jax.vmap(self.projector)))(zs)
        xs_pred = jax.vmap(jax.vmap(jax.vmap(self.projector)))(zs)
        return xs_pred, logpq, zs

    def sample(self, batch_size, ts, dt=1e-2, *, key):
        """Sample from prior drift"""

        eps_key, bm_key = jrandom.split(key)

        solver = diffrax.Euler()
        saveat = diffrax.SaveAt(ts=ts)

        def solve(z0, key):
            bm = diffrax.VirtualBrownianTree(
                t0=self.t0, t1=self.t1, shape=(self.latent_size,), tol=1e-3, key=key
            )
            control_term = diffrax.WeaklyDiagonalControlTerm(self.diffusion, bm)
            sde = diffrax.MultiTerm(diffrax.ODETerm(self.prior_drift), control_term)
            sol = diffrax.diffeqsolve(
                sde, solver, t0=self.t0, t1=self.t1, dt0=dt, y0=z0, saveat=saveat
            )
            return sol.ys

        eps = jrandom.normal(shape=(batch_size, *self.pz0_mean.shape[1:]), key=eps_key)
        z0s = self.pz0_mean + jnp.exp(self.pz0_logstd) * eps
        bm_keys = jrandom.split(bm_key, num=batch_size)
        batch_solve = jax.vmap(solve)
        zs = batch_solve(z0s, bm_keys)
        # xs = zs

        # if self.projector is None:
        #     xs = zs
        # else:
        #     xs = jax.vmap(jax.vmap(jax.vmap(self.projector)))(zs)
        xs = jax.vmap(jax.vmap(self.projector))(zs)
        return xs, zs
