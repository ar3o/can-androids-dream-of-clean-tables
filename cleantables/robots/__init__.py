"""Robot-side components: Franka kinematics, the release signal from the world model's contact head, and NVIDIA's Isaac-Lift
policies as plain torch modules.

  franka_kinematics    batched Franka Panda FK / IK and collision spheres
  release              where the hand lets go, from the world model's contact head
  lift_policy          the Isaac-Lift-KukaAllegro actor and its observation builder
  lift_policy_franka   the observation builder for the Isaac-Lift-Franka actor
Outputs go to $CLEANTABLES_DATA/franka_skill/.
"""
from cleantables import CT_DATA
OUT = CT_DATA + "/franka_skill"
