from geometry.geometry_base import BaseManifold, TopologyType
from config import AgentConfig, SystemConfig
from meta.emergence import MultiScaleSystem
from gradients.gradient_engine import compute_softmax_weights
import numpy as np

# Create point manifold
manifold = BaseManifold(shape=(), topology=TopologyType.PERIODIC)
system_cfg = SystemConfig(
    lambda_self=3.0, lambda_belief_align=2.0, lambda_prior_align=1.0,
    lambda_obs=0.0, lambda_phi=0.0, kappa_beta=1.0, kappa_gamma=1.0
)

system = MultiScaleSystem(manifold, max_emergence_levels=3)
system.system_config = system_cfg

agent_cfg = AgentConfig(K=3, observation_noise=0.1)
agent1 = system.add_base_agent(agent_cfg, agent_id='agent1')
agent2 = system.add_base_agent(agent_cfg, agent_id='agent2')

# Initialize
agent1.mu_q = np.random.randn(3) * 0.1
agent1.Sigma_q = np.eye(3)
agent2.mu_q = np.random.randn(3) * 0.1
agent2.Sigma_q = np.eye(3)

# Try to compute softmax - need adapter first
class SimpleAdapter:
    def __init__(self):
        self.agents = [agent1, agent2]
        self.config = system_cfg
        self.n_agents = 2
    def get_neighbors(self, idx):
        neighbors = []
        for j in range(2):
            if j != idx:
                neighbors.append(j)
        return neighbors

adapter = SimpleAdapter()
try:
    beta_fields = compute_softmax_weights(adapter, 0, mode='belief', kappa=1.0)
    print('beta_fields:', beta_fields)
    print('beta_fields[1] type:', type(beta_fields[1]))
    print('beta_fields[1] shape:', np.asarray(beta_fields[1]).shape)
    print('beta_fields[1] value:', beta_fields[1])

    # Try chi_ij too
    chi_ij = agent1.support.compute_overlap_continuous(agent2.support)
    beta_ij = beta_fields[1]
    lambda_belief = 2.0

    weight_field = lambda_belief * chi_ij * beta_ij
    print('\nweight_field type:', type(weight_field))
    print('weight_field shape:', weight_field.shape if hasattr(weight_field, 'shape') else 'no shape')
    print('weight_field value:', weight_field)
    print('Can convert to float?', end=' ')
    try:
        w = float(weight_field)
        print('YES:', w)
    except:
        print('NO')
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()
