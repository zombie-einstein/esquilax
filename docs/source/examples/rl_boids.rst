RL Boids
========

We can naturally use esquilax simulations as reinforcement
learning environments, allowing training policies across
batches of agents, and with multiple policies.

Environment
-----------

We'll use the same updates and states from the :ref:`evo_boids`
example, but wrap them up in an environment class

.. testsetup:: rl_boids

   from typing import Callable, Tuple, Any
   import chex
   import evosax
   import flax
   import jax
   import jax.numpy as jnp
   import optax
   import esquilax
   from esquilax import ml

   @chex.dataclass
   class Boid:
       pos: chex.Array
       heading: float
       speed: float

   @chex.dataclass
   class Params:
       max_speed: float = 0.05
       min_speed: float = 0.025
       max_rotate: float = 0.1
       max_accelerate: float = 0.005
       close_range: float = 0.01
       collision_penalty: float = 0.1

   class MLP(flax.linen.Module):
       layer_width: int
       actions: int

       @flax.linen.compact
       def __call__(self, x):
           x = flax.linen.Dense(features=self.layer_width)(x)
           x = flax.linen.tanh(x)
           return x

   @esquilax.transforms.spatial(
       10,
       (jnp.add, jnp.add, jnp.add, jnp.add),
       (0, jnp.zeros(2), 0.0, 0.0),
       include_self=False,
   )
   def observe(_k: chex.PRNGKey, _params: Params, _a: Boid, b: Boid):
       return 1, b.pos, b.speed, b.heading

   @esquilax.transforms.amap
   def flatten_observations(_k: chex.PRNGKey, params: Params, observations):
       boid, n_nb, x_nb, s_nb, h_nb = observations

       def obs_to_nbs():
           _x_nb = x_nb / n_nb
           _s_nb = s_nb / n_nb
           _h_nb = h_nb / n_nb

           dx = esquilax.utils.shortest_vector(boid.pos, _x_nb)
           d = jnp.sqrt(jnp.sum(dx * dx)) / 0.1
           phi = jnp.arctan2(dx[1], dx[0]) + jnp.pi
           d_phi = esquilax.utils.shortest_vector(
               boid.heading, phi, 2 * jnp.pi
           ) / jnp.pi
           dh = esquilax.utils.shortest_vector(
               boid.heading, _h_nb, 2 * jnp.pi
           ) / jnp.pi
           ds = (_s_nb - boid.speed) / (params.max_speed - params.min_speed)

           return jnp.array([d, d_phi, dh, ds])

       return jax.lax.cond(
           n_nb > 0,
           obs_to_nbs,
           lambda: jnp.array([-1.0, 0.0, 0.0, 0.0]),
       )

   @esquilax.transforms.amap
   def update_velocity(
       _k: chex.PRNGKey, params: Params, x: Tuple[chex.Array, Boid]
   ):
       actions, boid = x
       rotation = actions[0] * params.max_rotate * jnp.pi
       acceleration = actions[1] * params.max_accelerate

       new_heading = (boid.heading + rotation) % (2 * jnp.pi)
       new_speeds = jnp.clip(
           boid.speed + acceleration,
           min=params.min_speed,
           max=params.max_speed,
       )

       return new_heading, new_speeds

   @esquilax.transforms.amap
   def move(_key: chex.PRNGKey, _params: Params, x):
       pos, heading, speed = x
       d_pos = jnp.array(
           [speed * jnp.cos(heading), speed * jnp.sin(heading)]
       )
       return (pos + d_pos) % 1.0

   @esquilax.transforms.spatial(
       5, jnp.add, 0.0, include_self=False,
   )
   def reward(_k: chex.PRNGKey, params: Params, a: chex.Array, b: chex.Array):
       d = esquilax.utils.shortest_distance(a, b, norm=True)

       reward = jax.lax.cond(
           d < params.close_range,
           lambda _: -params.collision_penalty,
           lambda _d: jnp.exp(-50 * _d),
           d,
       )
       return reward

.. testcode:: rl_boids

   class BoidEnv(esquilax.ml.rl.Environment):
       def __init__(self, n_agents: int):
           self.n_agents = n_agents

       @property
       def default_params(self) -> Params:
           return Params()

       def reset(
           self, key: chex.PRNGKey, params: Params
       ) -> Tuple[chex.Array, Boid]:
           k1, k2, k3 = jax.random.split(key, 3)

           boids = Boid(
               pos=jax.random.uniform(k1, (self.n_agents, 2)),
               speed=jax.random.uniform(
                   k2,
                   (self.n_agents,),
                   minval=params.min_speed,
                   maxval=params.max_speed,
               ),
               heading=jax.random.uniform(
                   k3, (self.n_agents,),
                   minval=0.0, maxval=2.0 * jnp.pi
               ),
           )
           obs = self.get_obs(boids, params=params, key=key)
           return obs, boids

       def step(
           self,
           key: chex.PRNGKey,
           params: Params,
           state: Boid,
           actions: chex.Array,
       ) -> Tuple[chex.Array, Boid, chex.Array, chex.Array]:
           headings, speeds = update_velocity(
               key, params, (actions, state)
           )
           pos = move(key, params, (state.pos, headings, speeds))
           rewards = reward(key, params, pos, pos, pos=pos)
           boids = Boid(pos=pos, heading=headings, speed=speeds)
           obs = self.get_obs(boids, params=params, key=key)
           return obs, state, rewards, False

       def get_obs(
           self, state, params=None, key=None,
       ) -> chex.Array:
           n_nb, x_nb, s_nb, h_nb = observe(
               key, params, state, state, pos=state.pos
           )
           obs = flatten_observations(
               key, params, (state, n_nb, x_nb, s_nb, h_nb)
           )
           return obs

This structure is reasonably standard for reinforcement learning
environments, with methods to reset the environment state, and
a step methods that accepts actions and consequently updates
the state of the environment. We've also included a convenience
observation function that generates a flattened observation from
the current environment state.

RL Agent
--------

We also define the RL agent. In this case the boid agents
will share a single policy (though we could also initialise
individual policies). We implement the shared policy agent
class :py:class:`esquilax.ml.rl.SharedPolicyAgent`

.. note::

   We'll not implement the full RL agent functionality here
   (for brevity). The agent can be used to implement
   specific RL algorithms.

.. testcode:: rl_boids

   class RLAgent(ml.rl.SharedPolicyAgent):
       def sample_actions(self, _k, observations):
           actions = ml.get_actions(
               self.apply_fn, True, self.params, observations
           )
           return actions, None

       def update(self, _k, trajectories):
           return self, -1

The sample actions functions generates actions given
observations, in this case we simply apply the agent
network across the set of observations.

The update function should update the parameters and
optimiser of the agent, given trajectories collected over
the course of training.

Training
--------

We can then run the training loop

.. testcode:: rl_boids

   def rl_boids(
       env_params: Params,
       n_agents: int,
       n_epochs: int,
       n_env: int,
       n_steps: int,
       layer_width: int = 16,
       show_progress: bool = True,
   ):
       k = jax.random.PRNGKey(451)
       k_init, k_train = jax.random.split(k)

       env = BoidEnv(n_agents)

       network = MLP(layer_width=layer_width, actions=2)
       opt = optax.adam(1e-4)
       agents = RLAgent.init(k_init, network, opt, (4,))

       trained_agents, rewards, _ = ml.rl.train(
           k_train,
           agents,
           env,
           env_params,
           n_epochs,
           n_env,
           n_steps,
           show_progress=show_progress,
       )

       return trained_agents, rewards

We initialise the environment and the RL agent from the
neural network. We can then run the training loop using the
built in :py:meth:`esquilax.ml.rl.train` function.

.. doctest:: rl_boids
   :hide:

   >>> _ = rl_boids(
   ...     Params(), 4, 2, 2, 5, layer_width=4, show_progress=False,
   ... )
   ...
