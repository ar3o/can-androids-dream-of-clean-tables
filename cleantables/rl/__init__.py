"""Bimanual RL in the physics twins, taught by the world model.

python -m cleantables.rl.train --run NAME [--init CKPT]           # rotating data-parallel PPO, held-out evaluations
python -m cleantables.rl.ppo --run NAME --scene EPISODE ...       # one training block on one twin (used by train)
python -m cleantables.rl.evaluate --checkpoint CKPT --scene EP    # policy vs frozen lift prior, optional RTX video
"""
