# Settings reference

The main README explains each stage and why it's built that way. This page lists the numbers behind it.

## Episodes and perception

| Setting | Value |
|:--|:--|
| Episode labeller | Gemini Flash, one generic prompt per video; 1 s margin added to each segment |
| Data | 20 videos, 45 episodes (14 fruit, 12 tools, 19 stacking) |
| Depth | MoGe-3 ViT-L, one field of view per video, table plane by RANSAC |
| Object discovery | Qwen3-VL-4B lists objects on table crops; SAM 3 keeps the names it can segment |
| Tracking | SAM 3 video tracking at 5 fps, one session per episode |
| Object size | 75th percentile of the tracked footprint, since occlusion only ever shrinks it |
| Video features | V-JEPA 2 ViT-L (`vjepa2-vitl-fpc64-256`), 16 frames, cached per keyframe |
| Appearance and names | DINOv2 crop embeddings, MiniLM sentence embeddings |

## Twins

| Setting | Value |
|:--|:--|
| Simulator | Isaac Sim 6.1 / Isaac Lab 3, PhysX |
| Meshes | Hunyuan3D-2.1 shape + paint, one per physical object; crop projected as texture where paint failed; apples are revolved shapes |
| Background | per-pixel colour and depth that stay still over the video; SAM 3 removes the person |
| Camera | the phone's pose and intrinsics |
| Robots | two Kuka iiwa + Allegro arms at the person's estimated shoulder positions |
| Build time | about 11 s per episode |
| Box mass | 2.5 to 4 kg, for stability when brushed by an arm |

## World model

| Setting | Value |
|:--|:--|
| Trunk | transformer, 6 layers, width 256, 8 heads, 13.1M parameters with the path decoder |
| Tokens | 4 CLS, instruction, up to 24 objects, 2 hands, 32 V-JEPA 2 tokens |
| Next object | pointer over objects; set loss over every object that still has to move |
| End position | destination pointer + 5-component mixture density over the offset from the destination |
| Paths | flow-matching decoder, 32 steps per path, 12 samples, anchored at the predicted end |
| Progress | moves done / task moves, binary cross-entropy, ranking loss, time-reversed clips as negatives |
| Latent target | next V-JEPA 2 latent |
| Modalities | video + 3D, 3D only, video only (dropout during training); 3D only in simulation |
| Ensemble | 3 seeds; destinations from the ensemble mean |
| Evaluation | 5 folds; each holds out 3-4 box videos whole and 2-3 fruit episodes |

## Physics feedback

| Setting | Value |
|:--|:--|
| Geometry | filled voxel occupancy of every twin mesh at 5 mm; container cavities stay empty |
| Score | penetration deeper than one voxel while sweeping the moving object along an imagined path |
| Update | path decoder trained towards the least-penetrating 25% of its own samples |
| Regulariser | the imitation losses on the human videos keep running |
| Data | randomised layouts of the training twins only |

## Bimanual RL

| Setting | Value |
|:--|:--|
| Actor | per arm, trainable copy of the Isaac Lab Kuka-Allegro lift MLP (1595-512-256-128-23, ELU) |
| New inputs | other arm's 10 link positions, 23 joint positions and velocities, arm-arm distance; nearest obstacles; 5 imagined-path points at 5, 10, 15, 20 and 30 cm ahead, in the arm's frame; all through a zero-initialised layer |
| Grip strength | zero-initialised extra output blending the hand towards open |
| KL leash | weight 1.0 decaying to 0.05 over 300 iterations |
| Critic warm-up | 15 iterations |
| Actor learning rate | 3e-5 |
| Assignment | learned attention pointer, entropy bonus 0.05 decaying to 0.01 |
| Critic | centralised, sees both arms |
| Teacher | world model in 3D-only mode with no hand input; re-plans on a new object, every 3 s, and when an object is more than 6 cm off its path (at most once a second) |
| Success | object at rest, no fingertip touching, within 5 cm (horizontal) of its goal and not above it |
| Costs | arm contact 0.01, arm proximity under 5 cm 0.005, touching other objects 0.02, knocked obstacle 0.5, time 0.001; path-progress shaping weight 10 |
| Scale | PPO, 2 data-parallel workers × 448 environments, 19 training twins rotated every 60 iterations, 7 held out |
| Randomisation | layout shuffled in half of the resets, object mass and friction, spawn check that rejects unstable layouts |
