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

### One arm (after the deadline)

The same policy fine-tuned with one arm: the arm that can reach the container in each twin works, the other is moved 1.5 m back, and every layout is reachable by the working arm. Checkpoint at iteration 119 of the one-arm run; 128 episodes per twin with layouts drawn from seed 777 (unstable spawns are redrawn separately for each controller). An object counts when it is put down inside the destination the world model chose, so these numbers are not directly comparable with the two-arm table. Brackets are 95% Wilson intervals.

| Twin | Ours | NVIDIA lift policy, same arm |
|:--|:-:|:-:|
| IMG_2516_t11 | 205 / 640 | 71 / 640 |
| IMG_2516_t12 | 128 / 640 | 124 / 640 |
| IMG_2516_t13 | 244 / 768 | 82 / 768 |
| IMG_2516_t14 | 49 / 768 | 38 / 768 |
| Fruit pooled | 22.2% [20.7, 23.8] | 11.2% [10.1, 12.4] |
| IMG_2546_t01 | 23 / 256 | 16 / 256 |
| IMG_2547_t01 | 55 / 512 | 59 / 512 |
| IMG_2548_t01 | 81 / 512 | 44 / 512 |
| Tools pooled | 12.4% [10.7, 14.3] | 9.3% [7.8, 11.0] |

After iteration 199 the run continues with a fixed learning rate for the new layers (adapter and grip output); before that the KL-adaptive rule had held them at its 1e-5 floor since about iteration 15.

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

## Earlier robot results (before the deadline)

These are the first two-arm and one-arm evaluations with the four-finger hand, kept here so the main README can stay short. The one-arm runs were the ones that taught me to change one thing at a time.

#### Results

To test the policy fairly I held out 7 twins built from episodes it never trained on (4 fruit, 3 tools) and ran 32 episodes on each, with both controllers starting from exactly the same randomised layouts. The baseline is NVIDIA's original lift policy on both arms with a greedy rule that sends each free arm to the nearest object still to move, so the comparison isolates what the world-model-taught training adds. I count an object as done if it is inside its labelled container at the end of the episode, which is stricter than the training signal because it doesn't trust the world model's choice of destination.

<p align="center">
  <img src="assets/results/rl_heldout.png" alt="Held-out results: objects inside their container at the end, fruit 20.2% for our policy against 14.6% for the baseline and tools 18.8% against 10.6%; arm-on-arm contact drops from about 550 to 380 steps per episode for fruit and from 720 to 520 for tools." width="90%">
</p>

| Held-out twins | Ours | NVIDIA lift policy + greedy | Arm contact (steps/episode), ours vs baseline |
|:--|:-:|:-:|:-:|
| Fruit, 4 twins (704 fruit) | **20.2%** [17.4, 23.3] | 14.6% [12.2, 17.4] | 380 vs 551 |
| Tools, 3 twins (320 tools) | **18.8%** [14.9, 23.4] | 10.6% [7.7, 14.5] | 516 vs 719 |

The trained policy puts more objects away than NVIDIA's policy on both tasks, by about 40% for fruit and 75% for tools, and the arms bump into each other less, but the absolute numbers are still low.

<p align="center">
  <img src="assets/media/robot_two_arm_fruit.gif" alt="Held-out fruit twin, the first episode of the evaluation. Top: the trained two-arm policy, with both arms reaching for apples next to the bowl and crowding each other. Bottom: NVIDIA's lift policy with the greedy assignment from the same starting layout." width="100%">
  <br><sub>Held-out fruit twin IMG_2516_t13, the first episode of the evaluation (not picked). Top: the trained two-arm policy. Bottom: NVIDIA's lift policy from the same layout. Left of each: the phone camera, right: an overview.</sub>
</p>

Watching the videos, most of the failures are the two arms going for objects next to each other and ending up stuck on each other at the bowl or the toolbox, and when an arm does get an object over the container it usually loosens its grip until the object slips rather than opening the hand.

#### One arm

To separate the crowding from everything else, I trained a one-arm version after the deadline. It starts from the two-arm policy's weights, the second arm is moved 1.5 m back from the table, the working arm is whichever arm can reach the container in that twin, and every layout is drawn so that the working arm can reach each object and its container. The first version always used the left arm, and in 14 of the 26 twins that arm could not reach the container at all, which I only found by checking why some twins never scored anything. The baseline is NVIDIA's lift policy on the same arm with layouts drawn from the same seed (the one-arm evaluation redraws unstable spawns separately for each controller, so individual layouts can differ), and here an object counts when it is put down inside the destination the world model chose, so the numbers are not directly comparable with the two-arm table above.

| Held-out twins, one arm (iteration 119, 128 episodes per twin) | Ours | NVIDIA lift policy |
|:--|:-:|:-:|
| Fruit, 4 twins (2816 fruit) | **22.2%** [20.7, 23.8] | 11.2% [10.1, 12.4] |
| Tools, 3 twins (1280 tools) | **12.4%** [10.7, 14.3] | 9.3% [7.8, 11.0] |

Per twin, fruit is 205 vs 71 (t11), 128 vs 124 (t12), 244 vs 82 (t13) and 49 vs 38 (t14), and tools are 23 vs 16, 55 vs 59 and 81 vs 44 objects.

<p align="center">
  <img src="assets/media/robot_one_arm_fruit.gif" alt="Held-out fruit twin IMG_2516_t13. Top: the trained one-arm policy at iteration 219 carrying fruit towards the bowl. Bottom: NVIDIA's lift policy on the same arm." width="100%">
  <br><sub>The same held-out fruit twin with one arm, the first episode of each controller (not picked), played at double speed. Top: trained policy at iteration 219. Bottom: NVIDIA's lift policy on the same arm.</sub>
</p>

With the other arm out of the way the policy puts away about twice as much fruit as the baseline, but on tools it is only a little better than the baseline, and the main thing still missing is letting go: in the videos it often brings an object to the container and then keeps holding it, or lets it slip next to it. I think there are three reasons for this. NVIDIA's policy was trained only to lift and hold, so the starting point never opens its hand, and opening the hand anywhere except right above the container loses reward, which makes holding on a safe local optimum that RL has trouble leaving. The grip output's exploration noise was also resampled every control step, so the hand flickered and never stayed open long enough to drop anything, which I changed to noise that is held for half a second. And after the deadline I found that the automatic learning-rate rule had pushed the new layers, including the grip output, down to its floor of 1e-5 after about 15 iterations and kept them there, so the part of the policy that has to learn to let go was barely training at all. That run is now continuing from iteration 199 with a fixed learning rate for the new layers. The scores on the training twins are also much higher than on the held-out ones, which means some of what it learns is specific to the layouts it trains on, and with only a few million environment steps in the one-arm runs so far it has had very little training compared with what PPO usually needs for a dexterous hand.

