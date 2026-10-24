# Autoresearch: what it is, where it came from, when to use it

This is the background document for the repository. [METHOD.md](METHOD.md)
describes what this particular loop does; this page explains the idea it is
an instance of, for a reader who trains models and is deciding whether to
build one of these.

## 1. The idea in one paragraph

Autoresearch is a coding agent running the oldest optimisation algorithm
there is, hill climbing, over a training recipe, with git as the ratchet.
There is one file the agent may edit. There is one number that says whether
an edit helped, produced by code the agent may not touch. Every experiment
gets the same fixed training budget, so runs are comparable. The agent
proposes a change, trains, scores, and keeps the change if the number went
up by more than the noise, otherwise reverts it. Then it does that again,
unattended, for a night. The output is not one model but a ledger: every
edit tried, every verdict, and the sequence of kept edits that took the
recipe from where it started to where it ended.

What makes it autoresearch rather than hyperparameter search is the
proposer. A grid or a Bayesian optimiser moves numbers inside a space you
defined in advance. A coding agent reads the training script, the loss
curves and the log of what has already failed, and edits code: swaps the
loss, adds an augmentation, changes the sampler, restructures the network.
The search space is "what a competent engineer might try next", which is
much larger than any grid and much more interesting to watch.

## 2. Where it came from

**The name and the shape** come from Andrej Karpathy's `autoresearch`
repository, released in early 2026 on top of his `nanochat` language-model
code. The design is spare on purpose: `train.py` is the only file the agent
edits; `prepare.py` (data, tokenizer, evaluation) is fixed; `program.md` is
a steering file written and edited only by the human; every run trains for
exactly five minutes on one GPU; the metric is validation bits per byte,
chosen because it is vocabulary-independent and therefore hard to improve
by fiddling with the tokenizer. The agent keeps a change if the number
improves, otherwise `git checkout`. On one H100 that is about twelve
experiments an hour, about a hundred overnight. Community write-ups of
overnight runs report shapes like forty-odd experiments, a handful kept, a
few percent of total gain. The point was never the gain; it was that the
loop runs without a person in it and leaves an auditable trail.

**The lineage before it** is longer than the name. AutoML and neural
architecture search automated the same decisions a decade earlier, with
reinforcement learning or evolution as the proposer and enormous compute
bills. Hyperparameter optimisation (random search, Bayesian optimisation,
Hyperband) made the cheap version routine. What changed in 2024 and 2025
was the proposer: large language models that could read a training script
and write a plausible next version of it. AIDE ran a draft, debug and
improve tree search with an LLM at every node and reached medal level on
a sixth of the Kaggle competitions in MLE-bench; the AIRA follow-up showed
that the search policy around the LLM mattered as much as the LLM. In
parallel, DeepMind's AlphaEvolve and its open re-implementations
(OpenEvolve, ShinkaEvolve) used the model as the mutation operator inside
an evolutionary archive and found new results in mathematics and systems
code. The Darwin Gödel Machine turned the same loop on the agent's own
scaffolding. All of these need thousands of evaluations or long budgets.
Autoresearch is the small, greedy, single-GPU corner of that space: one
chain, one metric, short rounds, everything in git.

**The lineage after it** is mostly about keeping the loop honest. Within
weeks of the release there were forks for medical imaging
(`autoresearch-medimage`, 2-D classification and binary segmentation with
staged validation), for post-processing search without any training
(`mist-autoresearch`), and explicitly anti-gaming forks. A study that ran
the loop on a Quran-recitation scoring task found the agent hard-coding
verse identifiers to drive the metric, reading sibling runs through shared
git state, and leaving notes for its future self; adding a frozen held-out
split, and telling the agent it existed, removed both the cheating and the
inflated score. Cerebras published how its loop drifted off-task after
twelve unattended hours. The paper *Autoresearch with Coding Agents:
Generalizers and Metric-Maximizers* (arXiv:2607.18064) distilled this into
design rules: hold out, keep the feedback channel leak-free, isolate by
construction rather than by instruction, audit the state the agent can
reach, and pre-register the claim. For medical imaging specifically, AMID
(arXiv:2607.10522) added an independent reviewer agent that certifies each
attempt's split validity, metric direction and provenance before it can
count. Most of what is in this repository's fence is a direct response to
that literature.

## 3. The key concepts

**One editable surface.** The agent edits `recipe.py` and nothing else.
This is not a convenience; it is what makes the ledger readable. Every kept
change is a diff to one file, and the sequence of diffs is the research
narrative. It is also the first line of defence: a change outside the
surface is a harness breach, refused before training starts.

**A fixed budget.** Every experiment trains for the same wall-clock time.
That makes runs comparable without anyone deciding how long each one
deserves, and it rewards edits that make training more efficient, not just
bigger. The cost is that a wall-clock budget on a shared GPU is not
deterministic: the same seed gives a different number of steps and a
different score each time. Section 6 of METHOD.md records what that cost
us here.

**One frozen ruler.** The metric is computed by code the agent cannot
modify, on a validation split the agent cannot change, and the scorer's
hash is checked before every run. A ruler the agent can edit will be
edited. A ruler that is merely a proxy for what you care about will be
optimised as a proxy: the agent is doing exactly what you asked, which is
the problem. Choose the number as if an adversary will be maximising it,
because one will.

**The ratchet.** Keep or revert, decided by the number, enforced by git.
A kept edit is a commit; a reverted edit is a `git checkout` and a row in
the ledger saying so. Nothing is lost either way: the reverted edits are
the more informative half of the record, because they are the things a
competent engineer would have tried.

**The noise floor.** A loop that keeps every positive delta is a random
walk with a ratchet whenever the delta is smaller than run-to-run noise,
and it will climb anyway, on noise. So the noise is measured before the
climb, and the keep rule demands a gain larger than it. The measurement
itself has to be tested: re-run the unchanged recipe and check that the
rule does not keep it. If it does, the gate is measuring noise and the
campaign is void. This repository's first gate failed that check and had
to be rebuilt; the story is in METHOD.md and in the program ledger.

**A sequestered hold-out.** The validation split drives the climb, so by
the end of a campaign the recipe is fitted to it, a little or a lot. A
separate split that no run, no log and no agent ever sees is scored once,
by a person, at the close. The gap between validation and hold-out is the
honest measure of how much the loop overfitted its own ruler. The fence
that keeps hold-out identifiers out of every log is tested by planting a
leak and checking that the alarm fires.

**Null arms.** An agent that climbs is not evidence of anything unless you
know what climbing looks like without the agent. The do-nothing arm
(re-score the incumbent) gives the floor. The random-edit arm draws one
mutation from the same knob space, trains under the same budget, and faces
the same keep rule, for the same number of experiments. If the agent cannot
beat it, the headline is that it cannot. In a course setting, a third arm
replaces the agent with a student for an hour.

**The steering file.** `program.md` is the only channel from the human to
the agent and the agent never edits it. It states the objective, the
editable surface, the budget, the keep rule, the log format and the
forbidden list. Everything the agent is supposed to know is there, and
everything it is supposed to avoid is there, so when it does something
surprising you can check which sentence failed.

**Pre-registration.** The claim and the kill condition are written into
the program ledger before the first campaign run: what counts as success,
what ends the campaign early, and which settings may not change after the
start. The rule that stings most is the last one. Once you have seen
results, every adjustment to budget, threshold or target becomes a way of
reaching the answer you wanted, and the only protection is having written
down in advance that you would not.

**Everything logged.** One row per experiment in a tab-separated ledger,
one tracking run per experiment with every knob as a parameter and the
loss curve as a metric, every run directory kept. The loop produces a lot
of negative results and their value is entirely in being findable later.

## 4. When autoresearch is the right tool

Ask six questions. The loop is a good fit when the answers are mostly yes.

1. **Is there a single number you would trust an adversary to maximise?**
   Dice on a frozen split is such a number. "Clinical usefulness" is not,
   and neither is any metric whose computation the agent can reach.
2. **Can one experiment finish in minutes?** The loop needs tens to a
   hundred evaluations to be worth running. Five to fifteen minutes per
   experiment gives a night's worth. If a fair evaluation takes four
   hours, you get six experiments and a greedy climber is the wrong
   algorithm.
3. **Is the noise smaller than the steps?** Measure the run-to-run scatter
   of the baseline first. If plausible single edits are smaller than that
   scatter, you will spend your compute on seeds instead of ideas, as this
   project did at a 300 s budget. Fix the budget or the number of seeds
   before the climb, not during it.
4. **Is the editable surface small and the space of plausible edits
   large?** One file, many knobs, many recipe-family moves (loss, sampling,
   augmentation, schedule, architecture within a cap). If improving the
   system means rewriting a pipeline across many files, the ratchet cannot
   isolate what helped.
5. **Can you afford a sequestered hold-out?** It costs data you would
   rather train on. Without it, the final number is a validation score of a
   recipe tuned on that validation set and means little.
6. **Is the question "how far does a greedy climber get, and by which
   moves" rather than "find the best possible model"?** Autoresearch
   answers the first well and the second badly. If the state of the art is
   known and the task is to ship it, apply the known recipe; the loop is
   for finding out whether the recipe can be rediscovered, or for a domain
   where the recipe is not yet known.

It is the wrong tool when the metric is a proxy the agent can satisfy
without solving the problem; when evaluation noise swamps the effect sizes
you can afford to detect; when the data are small enough that the
validation split is memorised within a few dozen experiments; when any
kept change must be understood and justified before it is used, as in a
regulated device, because the loop keeps what scores, not what is
explicable; and when there is no one to read the ledger afterwards. The
loop's product is the ledger. If nobody is going to read it, run a sweep.

## 5. What to expect

Acceptance rates in published runs are low, ten to twenty percent of
experiments kept, and most of the gain comes from a few edits. Expect the
random arm to keep something too, and expect the two arms to be closer
than you would like. Expect the first noise-floor measurement to be wrong
in an instructive way. Expect at least one attempt by the agent to do
something you forbade, and design so that it is refused rather than
trusted. And expect that an honest negative result, where nothing survives
the hold-out, is a result: the write-up still maps every edit against the
known recipe and says which parts a greedy climber found and which it
never would.

## 6. Further reading

- Karpathy, `autoresearch`: https://github.com/karpathy/autoresearch
- Fork ecosystem: https://github.com/yibie/awesome-autoresearch
- *Autoresearch with Coding Agents: Generalizers and Metric-Maximizers*,
  arXiv:2607.18064 (the design rules)
- Cerebras, *How to stop your autoresearch loop from cheating*:
  https://www.cerebras.ai/blog/how-to-stop-your-autoresearch-loop-from-cheating
- AIDE, arXiv:2502.13138; MLE-bench, arXiv:2410.07095; AIRA search
  policies, arXiv:2507.02554
- ShinkaEvolve, arXiv:2509.19349; AlphaEvolve and OpenEvolve for the
  evolutionary branch
- AMID, *Towards Autonomous and Auditable Medical Imaging Model
  Development*, arXiv:2607.10522
- `autoresearch-medimage`: https://github.com/mattlungrenmd/autoresearch-medimage;
  `mist-autoresearch`: https://github.com/mist-medical/mist-autoresearch
- Isensee et al., *nnU-Net*, Nature Methods 2021, for the recipe this
  repository asks an agent to rediscover
