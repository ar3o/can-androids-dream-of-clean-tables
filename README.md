<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.svg">
    <img src="assets/banner-light.svg" alt="Can Androids Dream of Clean Tables?" width="100%">
  </picture>
</p>

<p align="center">
  <a href="https://developer.nvidia.com/isaac/sim"><img src="assets/badges/isaac_sim.svg" alt="NVIDIA Isaac Sim 6.1" height="28"></a>
  <a href="https://github.com/isaac-sim/IsaacLab"><img src="assets/badges/isaac_lab.svg" alt="NVIDIA Isaac Lab 3" height="28"></a>
  <a href="https://github.com/facebookresearch/vjepa2"><img src="assets/badges/vjepa2.svg" alt="V-JEPA 2 by Meta FAIR" height="28"></a>
  <img src="assets/badges/sam3.svg" alt="SAM 3 by Meta" height="28">
  <img src="assets/badges/pytorch.svg" alt="PyTorch" height="28">
  <a href="LICENSE"><img src="assets/badges/license.svg" alt="MIT licence" height="28"></a>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/hero_strip-dark.jpg">
    <img src="assets/hero_strip-light.jpg" alt="Three views of the same fruit-to-bowl episode: a frame of my phone video, the physics twin built automatically from it, and the two robot arms learning the task in the twin." width="100%">
  </picture>
</p>

<p align="center"><b>An automated pipeline that turns unlabelled phone videos of a person doing a task into robot policies trained in simulation.</b></p>

<p align="center">My submission for the <b>Humanoid Robot Learning internship challenge</b>, October 2026.</p>

<p align="center"><b>Live write-up:</b> I am continuing this project after the challenge deadline, and the latest version of this write-up lives at <a href="https://ar3o.github.io/clean-tables/">ar3o.github.io/clean-tables</a>.</p>

## Why I built this

My idea for this project was to build something that would actually matter for the way physical AI companies are trying to train robots right now. In my own past projects I have worked with reinforcement learning and with LLM planning for robot arms, and the same problem kept coming up: every new task needs a person to build the scene, write and tune a reward, and babysit the training. It works, but it stops at one task, because the next task needs someone to do all of it again, and that just doesn't scale to the amount of data you need to train a generalist humanoid.

So I wanted to attempt something that I knew would be a lot harder, but more rewarding and with more open problems in it: a fully automated pipeline with no human labelling and no hand tuning per task. You put in egocentric videos of people doing things, which companies are now collecting in huge amounts, and you get out policies in simulation that can adapt to new configurations of that task. If that works, every new hour of human video becomes training data for robots without anyone writing task-specific code, and the cost of teaching a robot a new task drops to the cost of recording someone doing it.

I also picked a few harder options on purpose. The robot uses a four-finger Allegro hand instead of a parallel gripper, which would have been the classic choice for simplicity, because dexterous hands are what a generalist humanoid will actually have and I wanted to learn how to work with them. For the same reason I went for two arms working on the same table instead of one, which means the policy also has to learn to share the space. The scope is kept quite wide on purpose: the pipeline is designed to support any task that is about moving objects to where they belong, so sorting, tidying, packing and stacking all fit, while something like pulling a lever or turning a tap would probably not work and is out of scope.

I built this as my submission for the Humanoid Robot Learning internship challenge, starting at the beginning of October 2026, and I am carrying on with it after the deadline.

<p align="center">
  <img src="assets/figures/fig_pipeline.png" alt="Overview of the whole automated pipeline: phone videos are cut into episodes by a video-language model, every object is tracked in 3D and rebuilt as a physics twin in Isaac Sim, a world model learns the task from the videos and is then corrected with physics in the twins, and two robot arms are trained with reinforcement learning in thousands of randomised twins with the world model as their teacher." width="100%">
  <br><sub>The full pipeline. Nothing between the phone videos and the trained policy is labelled by hand or written for a specific task.</sub>
</p>

## 1. Recording human data

I recorded a number of videos of myself putting fruit and tools in semi-random setups across my table and then sorting them at a natural speed into containers, the fruit into a bowl and the tools into a toolbox, which are basic tasks within the scope. The tool demonstrations also have cardboard boxes and other clutter on the table, because I wanted to test whether this could work in a less structured setting, which is one of the biggest challenges for humanoids. I also recorded myself stacking boxes, which the world model learns from but the robot doesn't train on yet. In total that is 20 videos and 45 episodes: 14 fruit, 12 tools and 19 stacking.

I wanted the whole workflow to run without human labelling, and understanding what the tasks in a video even are needs some general knowledge about the world, so I used Gemini Flash, which has decent visual understanding, to mine the videos. It cuts long recordings into clips, so you can feed in long human videos without anyone pressing start and stop, and for each clip it returns a generic task instruction, the objects involved and where they ended up. I kept the prompt task agnostic, so it doesn't mention fruit, tools or anything else that would lead it, and it should work on any video you give it.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/media/episodes-dark.gif">
    <img src="assets/media/episodes-light.gif" alt="Clips from the fruit video, each with the instruction and time range that Gemini wrote for that segment." width="80%">
  </picture>
  <br><sub>Episodes cut and described automatically. The text at the top is Gemini's output.</sub>
</p>

## 2. From real video to simulation

The idea now is to recreate the real human data in simulation. That lets us expand a handful of demonstrations into many more examples of the task through domain randomisation, it gives an environment for massively parallel learning that would be far too expensive to collect in the real world, and it lets the world model learn about physics in a way you just can't get from monocular phone video. So I want to rebuild the same objects and the same environment in a simulator, and for this I use NVIDIA Isaac Sim, partly because I'm familiar with it and partly because it has a lot of what this needs: Isaac Lab for parallel RL, domain randomisation, realistic RTX rendering that will matter for closing the sim-to-real gap when training vision policies later, and of course everything runs on CUDA.

Because Gemini has already told us which objects matter in each clip, we can follow them through the video. MoGe-3 gives a metric depth map for every frame and I fit the table plane with RANSAC, which gives a table-centred frame for everything else. Qwen3-VL lists what is on the table, SAM 3 checks each name and keeps the ones it can actually find, and SAM 3's video tracking then follows every object and my hands through the episode, so the masks and the depth can be lifted to 3D boxes over time. My first version detected objects separately in each keyframe and linked them with matching, which kept inventing objects: whenever a box moved, a new "ghost" box appeared, and one stacking video ended up with 12 objects when there were really 7. SAM 3's video mode keeps one identity per object for the whole episode and fixed that.

<p align="center">
  <img src="assets/figures/fig_real2sim.png" alt="Real-to-sim subsystem: depth and table frame from MoGe-3, object discovery with Qwen3-VL and SAM 3, SAM 3 video tracking lifted to 3D boxes, Hunyuan3D meshes, a static background plate with the person removed, and the assembled twin rendered from the phone camera with two robot arms." width="100%">
</p>

From the tracks, each episode becomes its own Isaac Sim scene of my actual table at the start of that episode. Most meshes come from Hunyuan3D-2.1 (the cardboard boxes are textured cuboids built straight from the video), made once per physical object from its video crops and reused across episodes, with the object's own crop projected on as texture where Hunyuan's texturing failed. Because my phone doesn't move during a recording, I didn't need Gaussian splatting for the background: for every pixel I keep the colour and depth that stay the same over the whole video, so anything that moves, including me, drops out, and SAM 3 removes whatever is left of me, like my feet when they stood still for a while. The twin's camera uses the phone's real pose and intrinsics so the render lines up with the video, and the two Kuka arms with Allegro hands are mounted roughly where my shoulders were. A twin takes about 11 seconds to build, so all 45 can be rebuilt whenever the builder changes. My first attempt used stand-in YCB objects and a made-up toolbox, which looked like a table with stuff on it but wasn't my table, so I threw it away.

<p align="center">
  <img src="assets/media/twin_side_by_side.jpg" alt="Left: my phone frame at the start of a fruit episode. Right: the twin of the same moment from the same camera, with me removed and the robot arms added." width="100%">
  <br><sub>A real start frame (left) and its twin rendered from the same camera (right).</sub>
</p>

## 3. The world model

The world model is the part I spent the most time on, because it is what replaces the person who would normally write the reward. It has to look at a scene and an instruction and say which object should move next, where it should end up, how it should get there and how far along the task is, and it has to learn all of that only from my videos.

<p align="center">
  <img src="assets/figures/fig_worldmodel.png" alt="World model architecture: frozen V-JEPA 2 video tokens, one token per object built from its 3D box, a frozen MiniLM name embedding and a frozen DINOv2 appearance embedding, two hand tokens and the instruction go into a trained transformer, which predicts the next object, its destination and end position, imagined carry paths with flow matching, task progress and the next video latent. Panel (b) shows the physics feedback stage." width="100%">
</p>

It is a 13M-parameter transformer that reads frozen V-JEPA 2 features from the last second of video, one token per object (its 3D box plus embeddings of its name and its appearance), my two hands and the instruction. One model covers all three tasks, and the instruction tells it which one it is doing. I trained it with modality dropout so it works from video and 3D together, from 3D alone or from video alone, which matters because in simulation the robot's teacher only gets exact 3D boxes and has no phone video to look at. Three copies trained with different seeds form an ensemble.

A few of the design choices came from things going wrong first. I don't tidy in a consistent order, so asking the model for the exact next object mostly taught it noise, and the next-object loss now accepts any object that still has to move. The end position is predicted relative to the destination it picks, so when the bowl is somewhere else in a randomised layout the goal moves with it. The imagined carries come from a flow-matching decoder anchored at the predicted end point, because my first version generated paths freely and they wandered all over the table. And task progress is trained on the number of moves done, with the same clips played backwards as negatives, because otherwise it just learns to read the clock from the video.

To evaluate it fairly, everything below is on held-out episodes using five folds, where each fold holds out three or four of the box videos completely (their tools and stacking episodes together) and two or three fruit episodes, and every number is compared with a simple baseline.

<p align="center">
  <img src="assets/results/wm_heldout.png" alt="Bar charts on held-out episodes: next object still to move, world model 0.68, 0.61, 0.84 against random 0.49, 0.30, 0.60 for fruit, tools and stacking; exact next object 0.32, 0.33, 0.49 against 0.18, 0.12, 0.22; task progress rank correlation 0.88, 0.85, 0.94 with a shuffled-frames control near zero, and the real clip scored above the reversed one 90%, 87% and 79% of the time." width="100%">
</p>

It picks an object that still needs moving well above chance on all three tasks, and progress is ordered correctly within an episode (rank correlation 0.85 to 0.94, against roughly zero for the same frames shuffled). When it sets goals in twins with random layouts it hasn't seen, the goal lands in the correct container for 99% of fruit and 88% of tools, with the misses being tools sent to a nearby cardboard box. There are two honest weaknesses. The progress value is squashed, rising from about 0.35 to 0.7 rather than 0 to 1 even on episodes it trained on, so it is underfit rather than failing to generalise; a monotone calibration fitted on the training folds recovers part of the range and is the green line in the video below. The imagined paths learned from video alone are also not accurate carries, which is what section 4 fixes.

<p align="center">
  <img src="assets/results/wm_inputs.png" alt="Input ablation pooled over all tasks: next object exact 0.42 with video and 3D, 0.42 with 3D only, 0.37 with video only, against chance 0.19; imagined path error 11.4, 11.6 and 17.1 cm against 17.4 cm for the object staying put." width="70%">
  <br><sub>Which input matters. The 3D object tokens carry the choice of object and the paths, and the video features carry most of the progress signal, which is why the robot's teacher can run on 3D alone.</sub>
</p>

Here it is on a fruit episode it never trained on. At the start it imagines a carry for every object on the table, each in its own colour, and then the real video plays with my actual carries drawn as dots in the same colours, so you can compare them object by object. Objects the tracker never saw moving still get an imagined carry but no dots; the ginger in this episode is missing entirely, because it is small and close to the table's colour and was never detected, which also means the world model never got a training example of it being moved.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/media/wm_plan_IMG_2516_t05-dark.gif">
    <img src="assets/media/wm_plan_IMG_2516_t05-light.gif" alt="Held-out fruit episode: at the start the world model draws an imagined carry for each fruit into the bowl, then the real video plays with the real carries drawn in the same colours and the predicted task progress underneath." width="80%">
  </picture>
</p>

<details>
<summary>Same thing on a held-out tools episode</summary>
<br>
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/media/wm_plan_IMG_2542_t01-dark.gif">
    <img src="assets/media/wm_plan_IMG_2542_t01-light.gif" alt="Held-out tools episode with imagined carries for each tool and the real carries drawn in the same colours." width="80%">
  </picture>
</p>
</details>

## 4. Teaching the world model some physics

Once I put the world model inside the twins, the problem with its paths was obvious: most of its imagined carries went straight through the side of the bowl or through a box. The carries it learned from barely lift off the table, because my hand covers the object for most of a carry and the tracker can't see what's under my fingers. I first tried fixing the data with FoundationPose for 6D object poses during carries, but it lost the object under my hand as well, so that went nowhere.

What worked was letting the world model check its own ideas against the twins. Every twin mesh is voxelised at 5 mm with containers kept hollow, so ending inside the bowl is fine and going through its rim is not. The world model then imagines lots of carries in lots of random layouts, each one is scored by how deep the moving object goes into anything else, and the path decoder is fine-tuned towards its own least-penetrating quarter of samples, which is basically reward-weighted regression applied to flow matching. The normal training on my videos keeps running at the same time so it doesn't forget where things go.

<p align="center">
  <img src="assets/results/physics_feedback.png" alt="Physics feedback results on held-out twins: fruit carries passing through another object drop from 98% to 13% and tools from 100% to 70%; penetration drops from 58 to 2 voxels per step for fruit and from 92 to 18 for tools; median lift rises from 0.9 to 13.0 cm for fruit and 1.6 to 12.4 cm for tools." width="90%">
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/media/physics_feedback-dark.gif">
    <img src="assets/media/physics_feedback-light.gif" alt="The same twin with no robot. Before physics feedback the imagined path skims the table into the side of the bowl; after, it lifts the object over the rim." width="100%">
  </picture>
  <br><sub>The world model alone in a twin, moving each object along its own imagined path, before and after physics feedback. There is no robot here.</sub>
</p>

One thing I learned the hard way: I ran a third round of physics feedback on tools only, and it ended up carrying everything 65 to 75 cm into the air, which no arm should do. I only caught it by watching the paths in the twins, so the robot learns from the second round, and watching the imagined paths is now the first thing I check on any new world model.

## 5. Training two robot arms with the world model as teacher

For the robot I didn't want to train grasping from scratch, because that would eat all my compute on a problem that is already reasonably solved, so each arm starts as a copy of NVIDIA's Isaac Lab policy for lifting objects with exactly this Kuka arm and Allegro hand, and all of its weights are then trained. The extra things each arm needs to know about (where the other arm is, the nearest obstacles and five points along the world model's imagined path) come in through zero-initialised layers, the same trick ControlNet uses, so on the first iteration the policy is exactly NVIDIA's and RL only has to learn what to change. A KL penalty to the frozen original keeps the grasp from falling apart early on, a learned attention pointer decides which arm takes which object (there is no rule like "left arm for the left half of the table"), and one critic sees both arms.

<p align="center">
  <img src="assets/figures/fig_rl.png" alt="RL training: each arm's actor is a trainable copy of NVIDIA's lift policy with a zero-initialised adapter and a grip-strength head, tied to the frozen original by a KL leash; a pointer assigns objects to arms and a shared critic sees both; 896 randomised twins run in parallel and the world model provides goals, imagined carry paths and the reward." width="100%">
</p>

During training the world model takes the place of a hand-written reward: at every reset it decides where each object should end up and how it should get there, the arm's goal slides along that imagined path, and the reward is progress along the path plus a bonus when an object is put down in the destination the world model chose, with small costs for the arms touching each other, knocking things over and taking too long. Letting go is learned too: rather than scripting a release, the policy has a grip-strength output that blends the hand towards open. Training runs with PPO across 896 twins at once on one RTX 4090, rotating through 19 training twins with randomised layouts, masses and friction, while 7 twins from other episodes are held out for evaluation.

Getting this to actually learn took most of the last few days, and almost every time the problem turned out to be the setup rather than the RL itself. The path reward used to reset to zero every time an arm moved on to its next object or the world model re-planned, so finishing an object cost about six times what it paid, which I fixed by keeping the potential a function of the state as potential-based shaping requires (Ng et al., 1999). The world model was also getting the robot's palms in the inputs it had learned from my hands, which on some tool scenes sent the goals outside the toolbox, so it now gets no hand input in the twins, which is also how every offline check had been run. Success first meant landing within 5 cm of the exact goal point, which in a 24 cm bowl rejected most of the fruit the robot had actually put in the bowl (only 11% counted while 38% were in it), so it now means being put down in the object the world model chose as the destination. And the new grip-strength layer was stuck on the same tiny learning rate that protects NVIDIA's pretrained weights, so it barely moved until I gave the new layers their own rate.

### Results

To test the policy fairly I held out 7 twins built from episodes it never trained on (4 fruit, 3 tools) and ran 32 episodes on each, with both controllers starting from exactly the same randomised layouts. The baseline is NVIDIA's original lift policy on both arms with a greedy rule that sends each free arm to the nearest object still to move, so the comparison isolates what the world-model-taught training adds. I count an object as done if it is inside its labelled container at the end of the episode, which is stricter than the training signal because it doesn't trust the world model's choice of destination.

<p align="center">
  <img src="assets/results/rl_heldout.png" alt="Held-out results: objects inside their container at the end, fruit 20.2% for our policy against 14.6% for the baseline and tools 18.8% against 10.6%; arm-on-arm contact drops from about 550 to 380 steps per episode for fruit and from 720 to 520 for tools." width="90%">
</p>

| Held-out twins | Ours | NVIDIA lift policy + greedy | Arm contact (steps/episode), ours vs baseline |
|:--|:-:|:-:|:-:|
| Fruit, 4 twins (704 fruit) | **20.2%** [17.4, 23.3] | 14.6% [12.2, 17.4] | 380 vs 551 |
| Tools, 3 twins (320 tools) | **18.8%** [14.9, 23.4] | 10.6% [7.7, 14.5] | 516 vs 719 |

The trained policy puts more objects away than NVIDIA's policy on both tasks, by about 40% for fruit and 75% for tools, and the arms bump into each other less, but the absolute numbers are still low, and watching the videos the main reason is the two arms crowding each other at the bowl and toolbox, with the grip still mostly releasing by loosening rather than opening.

I ran out of time to take the robot policy as far as I wanted before the deadline. The plan was to compare the two-arm policy against NVIDIA's lift policy, and also against a one-arm version, to see whether learning from the world model can beat the NVIDIA baseline once the arms stop getting in each other's way. For now the robot results are best read as a qualitative demonstration that the pipeline closes the loop from phone videos to a trained policy, and I will post updated scores, including the one-arm comparison, on the [project page](https://ar3o.github.io/clean-tables/) after the deadline.

## Limitations and what's next

The biggest limitation right now is the two arms getting in each other's way: they often pick objects whose carries cross and end up stuck on each other, and the grip output still mostly releases by loosening rather than deliberately opening. The world model's instruction embedding at training time didn't match the one the twins give the teacher, which explains some of the wrong tool goals, and retraining with one embedding is the fix. Next I want to distil the RL policies into a vision-language-action model trained on camera images from the twins so the end product runs from pixels, add videos of mixed tables so the instruction actually has to decide what to move, and feed in the full-table clean-ups I have already recorded. I will keep posting progress on the [project page](https://ar3o.github.io/clean-tables/). The full result tables and settings are in [docs/results.md](docs/results.md) and [docs/method.md](docs/method.md).

## Running it

Setup is in [docs/installation.md](docs/installation.md) and every stage is explained in [docs/running.md](docs/running.md).

```bash
export PYTHONPATH=$PWD
$VISION_PY -m cleantables.episodes.segment && $VISION_PY -m cleantables.episodes.prepare   # videos -> episodes
VISION_PY=$VISION_PY bash scripts/run_perception.sh                                       # episodes -> 3D tracks
bash scripts/build_twins.sh                                                              # -> one physics twin per episode
PY=$VISION_PY bash scripts/worldmodel.sh                                                 # -> world model + physics feedback
$SIM_PY -m cleantables.rl.train --run main                                               # -> two-arm policy
$SIM_PY -m cleantables.rl.evaluate --checkpoint runs/rl/main/ckpt/last.pt --scene IMG_2546_t01
```

```
cleantables/
  episodes/     cutting videos into episodes with Gemini
  perception/   depth, table frame, objects, SAM 3 tracking, 3D boxes, events, V-JEPA 2 features
  twin/         background plates, mesh library, Isaac Sim scenes, renders, world-model replay
  worldmodel/   dataset, model, training, evaluation, physics feedback, the teacher used by the robot
  rl/           two-arm environment, policy, PPO, training launcher, evaluation
configs/        one YAML per stage, defaults = the settings behind the results
scripts/        shell drivers for the longer stages
docker/         container definitions for the simulation and vision environments
```

<details markdown="1">
<summary>Ideas borrowed from other work</summary>

| Idea | Where it's from | Used for |
|:--|:--|:--|
| Self-supervised video features | V-JEPA 2 (Assran et al., 2025) | world-model video input and next-latent target |
| Flow matching | Lipman et al., 2023 | imagined carry paths |
| Mixture density networks | Bishop, 1994 | end-position distribution |
| Time-reversed negatives for progress | Arrow of time (Wei et al., 2018); Rank2Reward (Yang et al., 2024) | task-progress head |
| Modality dropout | ModDrop (Neverova et al., 2015) | running the world model on 3D alone |
| Deep ensembles | Lakshminarayanan et al., 2017 | three-seed ensemble |
| Reward-weighted regression | Peters & Schaal, 2007 | physics feedback |
| Zero-initialised input layers | ControlNet (Zhang et al., 2023) | adding inputs to NVIDIA's policy |
| KL penalty to a reference policy | RLHF fine-tuning (Ouyang et al., 2022) | keeping the grasp while fine-tuning |
| Each hand observes the other | Bi-DexHands (Chen et al., 2022) | two-arm coordination |
| Centralised critic | MAPPO (Yu et al., 2022) | one critic for both arms |
| Pointer attention | Pointer Networks (Vinyals et al., 2015) | assigning objects to arms |
| Potential-based shaping | Ng et al., 1999 | path-progress reward |
| Domain randomisation | Tobin et al., 2017 | layouts, masses, friction |

</details>

## Built with

| Component | Used for |
|:--|:--|
| Isaac Sim 6.1 and Isaac Lab 3 (NVIDIA) | twins, physics, rendering, RL environments |
| `Isaac-Lift-KukaAllegro` policy (NVIDIA) | starting weights for each arm |
| V-JEPA 2, SAM 3, DINOv2 (Meta) | video features, masks and tracking, appearance |
| MoGe-3 (Microsoft) | metric depth |
| Qwen3-VL (Alibaba), Gemini Flash (Google) | object listing, episode mining |
| Hunyuan3D-2.1 (Tencent) | object meshes |
| MiniLM | name and instruction embeddings |

## License

The code in this repository is MIT licensed. Third-party models and assets keep their own licences.
