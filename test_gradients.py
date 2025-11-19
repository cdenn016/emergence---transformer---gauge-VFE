import numpy as np
import sys
import traceback
from geometry.geometry_base import BaseManifold, TopologyType
from config import AgentConfig, SystemConfig
from meta.emergence import MultiScaleSystem, HierarchicalConfig, HierarchicalEvolutionEngine
from math_utils.so3_generators import generate_so3_generators
from gradients.gradient_engine import compute_natural_gradients

# Create point manifold
manifold = BaseManifold(shape=(), topology=TopologyType.PERIODIC)

# Create system config
system_cfg = SystemConfig(
    lambda_self=3.0,
    lambda_belief_align=2.0,
    lambda_prior_align=1.0,
    lambda_obs=0.0,
    lambda_phi=0.0,
    kappa_beta=1.0,
    kappa_gamma=1.0,
    identical_priors='off'
)

# Create multi-scale system
multi_scale_system = MultiScaleSystem(manifold, max_emergence_levels=3)
multi_scale_system.system_config = system_cfg

# Create agent config
agent_cfg = AgentConfig(K=3, observation_noise=0.1)

# Create 2 agents
generators = generate_so3_generators(3)
for i in range(2):
    agent = multi_scale_system.add_base_agent(agent_cfg, agent_id=f'agent_{i}')
    agent.generators = generators
    agent.mu_q = np.random.randn(3) * 0.1
    agent.Sigma_q = np.eye(3)
    agent.mu_p = np.random.randn(3) * 0.1
    agent.Sigma_p = np.eye(3)
    from math_utils.so3_frechet import so3_exp
    random_phi = np.random.randn(3) * 0.1
    agent.gauge.phi = so3_exp(random_phi)

# Simple adapter
class SimpleAdapter:
    def __init__(self, agents, cfg):
        self.agents = agents
        self.config = cfg
        self.n_agents = len(agents)
    def get_neighbors(self, idx):
        neighbors = []
        for j in range(self.n_agents):
            if j != idx:
                neighbors.append(j)
        return neighbors
    def compute_transport_ij(self, i, j):
        from math_utils.transport import compute_transport
        return compute_transport(
            self.agents[i].gauge.phi,
            self.agents[j].gauge.phi,
            self.agents[i].generators,
            validate=False
        )

# Try to compute gradients
try:
    active_agents = multi_scale_system.get_all_active_agents()
    print(f'Number of active agents: {len(active_agents)}')
    print(f'Agent 0 mu_q shape: {active_agents[0].mu_q.shape}')
    print(f'Agent 0 Sigma_q shape: {active_agents[0].Sigma_q.shape}')
    
    temp_system = SimpleAdapter(active_agents, system_cfg)
    gradients = compute_natural_gradients(temp_system, verbose=0)
    print('SUCCESS - gradients computed')
except Exception as e:
    print(f'ERROR: {e}')
    traceback.print_exc()
