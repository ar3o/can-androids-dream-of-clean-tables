# Results

The world model and the twins are trained and evaluated. The bimanual policy that learns from the world model alone is still training, and its row stays empty until it's evaluated on the held-out twins. Brackets are 95% bootstrap intervals over episodes.

## World model

Data: 45 episodes from 20 phone videos (14 fruit, 12 tools, 19 stacking). Five folds: a box video is held out as a whole (its tools and stacking episodes together), and fruit episodes, which all come from one video, are held out one by one. Three seeds per fold, averaged as an ensemble.

### Main table (video + 3D)

| Evaluation | Fruit | Tools | Stacking |
|:--|:-:|:-:|:-:|
| Predicted next object still has to be moved | 0.68 [0.52, 0.83] | 0.61 [0.46, 0.74] | 0.84 [0.72, 0.94] |
| Chance (random object still on the table) | 0.49 | 0.30 | 0.60 |
| Predicted next object is exactly the one moved next | 0.32 [0.17, 0.47] | 0.33 [0.20, 0.47] | 0.49 [0.40, 0.56] |
| Chance | 0.18 | 0.12 | 0.22 |
| Progress, rank correlation within an episode | 0.88 | 0.85 | 0.94 |
| Same, frames shuffled (control) | 0.04 | 0.05 | 0.01 |
| Real clip scored above the same clip played backwards | 90% | 87% | 79% |

Progress is ordered correctly but compressed: the mean prediction is about 0.35 before an episode starts and about 0.69 after it ends, rather than 0 and 1.

### Where the model does not beat simple baselines

| Evaluation (cm, lower is better) | Fruit | Tools | Stacking |
|:--|:-:|:-:|:-:|
| End position, given the true destination | 7.3 | 6.0 | 10.7 |
| Baseline: centre of the true destination | 6.9 | 5.7 | 10.7 |
| Mean imagined path vs the tracked carry | 16.7 | 9.7 | 10.7 |
| Baseline: straight line to the true end point | 10.7 | 7.1 | 5.7 |
| Baseline: object stays where it is | 23.9 | 17.2 | 15.6 |

Given the right destination, the predicted end position is no better than the destination's centre. The imagined paths beat "stays where it is" but not a straight line to the true end point. That baseline is given the answer, so it is a ceiling rather than a fair competitor, but the paths learned from video are clearly not accurate carries. The physics-feedback stage below is what makes them usable.

### Which input matters (all tasks pooled)

| Input | Next object still to move | Next object exact | Progress rank correlation | Mean path error |
|:--|:-:|:-:|:-:|:-:|
| Video + 3D | 0.77 | 0.42 | 0.89 | 11.4 cm |
| 3D only | 0.77 | 0.42 | 0.84 | 11.6 cm |
| Video only | 0.75 | 0.37 | 0.89 | 17.1 cm |

The 3D object tokens carry the paths and the choice of object; the video features carry most of the progress signal. In simulation the model runs on 3D only.

### As a goal-setter

| Check | Fruit | Tools |
|:--|:-:|:-:|
| Held-out episodes: median distance from the predicted goal to where the person put the object | 5.9 cm (53 objects) | 13.4 cm (45 objects) |
| Baseline: object left in place | 30.7 cm | 31.4 cm |
| Held-out twins, randomised layouts: goal inside the correct container | 99% (266 scenes) | 88% (181 scenes) |

The tool mistakes are tools sent to a nearby box instead of the toolbox.

### Physics feedback

Imagined carries on held-out twins (fruit: 176, tools: 80), before and after the physics-feedback stage. Each carry is checked against the twins' mesh geometry in 5 mm voxels, with containers kept hollow.

| Imagined carries | Before | After |
|:--|:-:|:-:|
| Fruit passing through another object | 98% (173 / 176) | 13% (23 / 176) |
| Fruit, median lift above start and end | 0.9 cm | 13.0 cm |
| Tools passing through another object | 100% (80 / 80) | 70% (56 / 80) |
| Tools, penetration into other objects (voxels per step) | 92 | 18 |
| Tools, median lift above start and end | 1.6 cm | 12.4 cm |

These are for the second physics-feedback round, which the robot learns from (carries peak around 18 to 25 cm above the table). A third round on tools only cut tool penetration to 2.7 but pushed every carry to 65 to 75 cm, because the score has no reason to stop climbing, so it is not used.

## Bimanual policy

Final policy: a trainable copy of NVIDIA's lift policy per arm, taught by the world model, with success counted as being put down in the destination the world model chose. Evaluation: 7 held-out twins (fruit IMG_2516 t11-t14, tools IMG_2546/2547/2548 t01), 32 episodes per twin, 30 s each, and both controllers start from identical layouts that are checked for stability before either robot acts. An object counts when it rests inside its labelled container at the end of the episode, which does not depend on the world model's choice of destination. Brackets are 95% Wilson intervals.

| Controller | Fruit in container | Tools in container | Arm-contact steps per episode (fruit / tools) |
|:--|:-:|:-:|:-:|
| NVIDIA lift policy on both arms, greedy nearest-object assignment | 14.6% (103 / 704) [12.2, 17.4] | 10.6% (34 / 320) [7.7, 14.5] | 551 / 719 |
| World-model-taught policy | 20.2% (142 / 704) [17.4, 23.3] | 18.8% (60 / 320) [14.9, 23.4] | 380 / 516 |

| Twin | Ours | Baseline |
|:--|:-:|:-:|
| IMG_2516_t11 | 53 / 160 | 28 / 160 |
| IMG_2516_t12 | 36 / 160 | 34 / 160 |
| IMG_2516_t13 | 33 / 192 | 27 / 192 |
| IMG_2516_t14 | 20 / 192 | 14 / 192 |
| IMG_2546_t01 | 10 / 64 | 13 / 64 |
| IMG_2547_t01 | 21 / 128 | 13 / 128 |
| IMG_2548_t01 | 29 / 128 | 8 / 128 |

A one-arm comparison (the arm that can reach the containers works, the other is moved away, and every layout is reachable) was still training at the deadline; its results will be posted on the project page.

An earlier reference run used a frozen NVIDIA policy with a scripted release and a hand-designed lift arc. It roughly doubled the frozen policy's placements on held-out twins, but those two pieces would have to be rewritten for every task, so it is not part of the pipeline.

## Things that didn't work

| Experiment | Result | Takeaway |
|---|---|---|
| 6D tracking of carries with FoundationPose | Did not recover the lift of carried objects | The hand hides the object; fixed instead by physics feedback in the twins |
| Mixed scenes (tools and fruit on one table) | The world model ignores the instruction and goes for the tools | Every training scene held one task's objects, so the instruction never had to decide anything |
| Binary "open hand" action | Random opening dropped objects mid-carry, and the policy learned never to open | Replaced by a continuous grip-strength output |
| World-model goals with a frozen grasp policy and no lift target | No placements: the arms retreated or never let go | Led to the current design, a trainable grasp policy that sees the imagined path |

## Known limitations

- **Instruction embedding.** The world model was trained with a hashed embedding of the instruction text, while the twins give the robot's teacher a MiniLM sentence embedding. Its goals still land in the right container (fruit 100%, tools 83% on the training twins, 100% on the held-out ones), and switching the twins to the training embedding helps some scenes and hurts others, so the release keeps it and a retrain with one embedding is the fix.
- **Perception on the tools videos.** The 19 box and tool videos were processed with an earlier, box-specific geometry step; the repository ships the general pipeline that processed the fruit video. Re-running the general pipeline on a tools video gives boxes within a few millimetres of the released ones, but it can miss an object (one video lost the wrench).
- **Object names.** One fruit-video object ("ginger") was added to the object list by hand during development. The released pipeline instead adds any object named in the episode labels that the VLM listing missed.
- **Held-out scope.** The held-out twins are new episodes and layouts for the robot, but the world model that teaches it was trained on every episode, and the held-out fruit twins come from the same video as the training ones.
- **Success test.** "Released" means no fingertip touches the object, so an object resting in the palm would count. The put-down test does not check the container, only the world model's goal; evaluation reports both.
- **Stacking twins.** The stacking twins were built by an earlier version of the twin builder and are not used for robot training.
