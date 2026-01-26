import jax.numpy as jnp
from Plant.AbstractPlant import AbstractPlant

class CournotCompetition(AbstractPlant):

    def __init__(self, target_Y, min_D, max_D, max_price, marginal_cost, dt=1):
        self.max_price = max_price
        self.marginal_cost = marginal_cost
        super().__init__(target_Y, min_D, max_D, dt)

    def get_V(self, H):
        return jnp.sqrt(2*9.81*jnp.maximum(H, 0))
    
    def get_init_state(self):
        return (0.5, 0.5)

    def step(self, state, U, D):
        qa_prev, qb_prev = state
        qa = jnp.clip(qa_prev + U * self.dt, 0, 1)
        qb = jnp.clip(qb_prev + D * self.dt, 0, 1)
        q = qa + qb
        price = self.max_price - q
        profit = qa * (price - self.marginal_cost)
        return profit, (qa, qb)