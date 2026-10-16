# Attribution: what is ours and what isn't

## Third-party components

| Component | Role here | Licence |
|---|---|---|
| Isaac Sim 6.1 / Isaac Lab 3 (NVIDIA) | Physics and rendering, RL environments | Isaac Sim: NVIDIA licence; Isaac Lab: BSD-3-Clause |
| `Isaac-Lift-KukaAllegro` policy (NVIDIA) | Starting weights for each arm's actor (published rsl_rl checkpoint), then fine-tuned | see Isaac Lab |
| V-JEPA 2 ViT-L (Meta) | Frozen video features | see upstream repo |
| SAM 3 (Meta) | Object and hand masks, video tracking | SAM License |
| MoGe-3 (Microsoft) | Metric depth from a single frame | see upstream repo |
| Qwen3-VL (Alibaba) | Listing the objects on the table | see upstream repo |
| Gemini Flash (Google) | Cutting videos into episodes and writing their instructions | API terms |
| DINOv2 (Meta) | Object appearance features | Apache-2.0 |
| MiniLM sentence embeddings | Object-name and instruction embeddings | Apache-2.0 |
| Hunyuan3D-2.1 shape + paint (Tencent) | Object meshes and textures from video crops | Tencent Hunyuan 3D 2.1 Community License |
| FoundationPose (NVIDIA) | Tested for 6D carry tracking; not used in the final pipeline | NVIDIA Source Code License |

## Ours

1. **Phone video → episodes → 3D object tracks, automatically.** Generic episode cutting, open-vocabulary object discovery, per-episode tracking windows, lifting to a table-centred metric frame.
2. **Phone video → physics twin, automatically.** Per-episode twins from the first frame, a shared mesh library, the tabletop orthophoto, the static-background depth mesh that removes the person, the phone camera, robot placement from the person's shoulders, and the physics settings that keep cluttered scenes stable.
3. **World model.** V-JEPA 2 features plus 3D object tokens, hands and the instruction; next-object, destination, end-position, imagined-path, progress and latent-prediction heads; held-out evaluation against simple baselines.
4. **Physics feedback.** Imagined carries scored against the twins' voxelised geometry with hollow containers, and the path head trained towards its own non-penetrating samples.
5. **Bimanual policy taught by the world model.** NVIDIA's grasp policy made trainable with zero-initialised extra inputs, a decaying KL leash and a learned grip strength; world-model goals, imagined paths and success test as the teacher.

## Hand-chosen settings that remain

Nothing in the current policy scripts the grasp, the carry or the release. These choices are still made by hand and apply to every task alike:

- the weights of the generic costs (arm contact and proximity, touching other objects, knocking things over, time) and of the path-progress term;
- the KL leash schedule and the critic warm-up;
- the success test: an object at rest, no fingertip touching, within 5 cm (horizontal) of its world-model goal and not above it;
- twin approximations flagged per scene: procedural apples, heavy boxes for stability.

The earlier reference run also used a scripted release and a hand-designed lift arc; both were removed.

## Tried and dropped

- 6D carry tracking with FoundationPose (the hand hides the object).
- A binary "open hand" action (replaced by a continuous grip strength).
- An LLM prior for object destinations: dropped because it was scored on our own auto-labels, which is not a fair test.
