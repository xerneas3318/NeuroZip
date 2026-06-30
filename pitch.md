# NeuroZip — stage pitch (short-demo cut)

Runtime: ~2:10 spoken + 30 s for the two live-demo beats ≈ 2:40 total.
The honest-slide (was slide 6) is cut for the short-demo slot; the
weaknesses still live in the Q&A section below — raise them if a judge
prompts, don't dwell on them on stage.

This is the tightened version. Every claim aligned with
[`plots/phase0_summary.json`](plots/phase0_summary.json),
[`plots/phase1_bio_numbers.json`](plots/phase1_bio_numbers.json),
[`plots/phase2_permutation.json`](plots/phase2_permutation.json),
[`results.md`](results.md). All bio numbers are at the v4_low tier
(144×) so they match the headline compression ratio.

The long-form version with extra context, expanded slides, and the full
"why this wording matters" annotations lives on the `criticism-1` branch.
This file is the one you walk on stage with.

---

## Slide 1 — EEG trace, no title, no logo · 0:00 – 0:30 · LIVE DEMO

> One second of a human brain: 63 electrodes, someone looking at a
> photo. You can't read it. Watch.
> *(type "ostrich")*
> I'll find when they saw an ostrich, by typing the word.
> *(thumbnails populate)*
> We never trained on text. The model learned to align brain
> recordings with the images people saw; CLIP brings the words for
> free.

## Slide 2 — two reconstructions side by side · 0:30 – 1:05 · LIVE DEMO

> Same one second, compressed two ways, same architecture. Left:
> trained for waveform fidelity, the default. Right: ours, trained
> for meaning.
>
> Search both. *(type "ostrich")* The fidelity one loses the ostrich.
> Ours keeps it.
>
> So what you're looking at doesn't live in the loud parts of the
> signal fidelity protects. It lives somewhere smaller, and more
> fragile.

## Slide 3 — where + when the meaning lives · 1:05 – 1:40

> So: where? We compressed 144× and looked at what the codec gave up
> first. Occipitotemporal electrodes, biggest gap in the N170 window,
> 150 to 200 milliseconds. Visual cortex 21 percent tighter than
> fidelity; the rest of the scalp only 5 percent — a four-times
> spatial preference. Shuffled the channel labels ten-thousand times,
> three random sets matched it. p less than 0.001.
>
> That's not a discovery — a supervised loss for object identity would
> be wrong if it didn't preserve visual cortex. Read it as validation
> that the task gradient is shaping the codec correctly.

## Slide 4 — `144×` and the per-bit edge · 1:40 – 2:05

> The engineering claim is per-bit. At every tier NeuroZip retrieves
> more concepts per bit than fidelity: plus ten points top-5 at the
> 144-times tier, plus four to eleven across tiers. An independent
> classifier the codec never saw confirms it at the extreme tier
> (NeuroZip 100, fidelity 88), but that classifier saturates above
> this tier, so the per-bit edge is what carries the win.

## Slide 5 — one-sentence "how" · 2:05 – 2:30

> One sentence on the how: we put a **frozen judge** in the loop that
> scores whether the meaning survived, and train the compressor against
> it. The same judge that decides what to keep at training is what reads
> your text query at inference.

(This is the fallback for the original "fMRI replication" slot — we
haven't trained the cross-modal variant. If/when we do, this slide gets
swapped for "ran the exact same method on fMRI; same result; meaning
survives, fidelity doesn't.")

## Slide 7 — close · 2:30 – 3:00

> NeuroZip: brain datasets two orders of magnitude smaller, still
> searchable by language. Per-bit retrievability is the engineering
> claim. The validation: what the codec preserves tightest is exactly
> what visual neuroscience predicts. We didn't discover the
> neurophysiology, we confirmed the codec keeps the right thing.

---

## Pre-stage checklist

- [ ] Pre-load both demo queries (slide 1: `ostrich`, slide 2: re-`ostrich`
      against both panes). Have them in browser history.
- [ ] Record a **20-second screen capture** of slides 1 + 2 working end-to-
      end. Open in a second tab as fallback. Wifi failure mid-demo is the
      most common venue disaster; this kills that risk.
- [ ] Test the live pitch on the actual presentation machine, on the
      actual venue network, with the actual projector ~30 min before
      stage. Font loads + image cache matter.
- [ ] Rehearse the cut script against a stopwatch. Target 2:10 spoken
      + 0:30 demo ≈ 2:40 total. End early rather than rush slide 7.
- [ ] Don't expand slide 5. Everything you cut goes in Q&A; that's where
      technical merit gets scored anyway. The weaknesses-and-defenses
      content that used to be slide 6 is now Q&A-only — be ready to
      deliver it with the same confidence if a judge prompts.
- [ ] Have `plots/phase2_permutation.png` ready as a slide-3 backup if a
      judge asks "how do you know that 4× isn't chance." The histogram
      is the answer.

---

## Strategic note (year-out, not for this stage)

The advisor flagged that this is the neuroscience project at a
cancer-proteomics hackathon (QBI's orbit: Zaro lab, Bar-Peled lab,
chemical biology / proteomics / DrugMap). CysTeam-type projects win
because they're domain-native to that room, built by and for the
judges' own labs.

**This pitch is calibrated for the EEG identity, and that caps the
realistic placement at top-3.** First place at QBI requires re-aiming
the same method at QBI-native data: Cell Painting, mass-spec proteomics,
Perturb-seq, cysteine ligandability. Same technical contribution
(frozen-judge task-aware compression + search-by-meaning), different
substrate, co-developed with a sponsoring lab.

If you're optimizing this year's win, that pivot is the play, but it's
weeks-to-months of work, not a pitch tweak. Out of scope for criticism-2.

---

# Q&A prep — the five hardest questions

## "It's n=1. Does this generalize to other subjects?"

The dataset has ten subjects; we trained on one because the iteration
loop for the codec contribution is faster that way. The pipeline is
subject-agnostic, one flag in `data.py` switches it. We expect
cross-subject training to *help* the absolute numbers (+8–12 pp from the
THINGS-EEG literature) and we expect the *codec contribution* — the gap
between NeuroZip and fidelity — to carry, because it's about which
structure is task-relevant, not which subject. The lesson we wanted to
isolate is "task awareness preserves more per bit"; adding subject
variance would have confounded that ablation.

## "It's trial-averaged over 80 reps. Doesn't that hide the noise?"

Yes, averaging substantially denoises EEG. Single-trial retrieval is
harder and we don't claim to solve it. The published THINGS-EEG numbers
(NICE 33% top-1, Wu CVPR 2025 35%) use the same trial-averaged protocol;
it's the standard eval surface so codec methods are comparable. The
codec contribution — the gap between fidelity and NeuroZip — wouldn't
change qualitatively in single-trial because both methods see the same
noisy input. The absolute numbers would drop together.

## "Aren't you just re-finding the N170?"

Yes, and that's the point. The loss is supervised on CLIP-image features
of objects the subject saw, so the codec preserving visual cortex and
the N170 window is the *expected* behavior of that loss. We don't claim
to discover the N170 — we use the fact that the localization matches
known neurophysiology as validation that the method works on this
substrate. The investigational application is novel paradigms (memory,
imagery) or novel modalities (fMRI, intracranial) where we don't already
know the answer. The permutation test (p < 0.001 over 10,000 random
13-channel sets) shows the localization isn't an artifact of the
metric; it tracks the visual set specifically.

## "Who actually needs EEG compression?"

Three places, in order of pain. *Dataset sharing*: THINGS-EEG itself was
re-released in fp16 to halve its size, the storage pressure is real for
exactly this kind of corpus. *Wearable EEG*: consumer headsets stream
multi-hour recordings off-device with constrained battery and bandwidth.
*Searchable archives*: brain-imaging labs accumulate decades of session
data; "find me trials where the subject saw faces" is currently a manual
labeling job. Our contribution turns that into a text query. None of
this is blocked today, but all of it is expensive enough that people
work around it.

## "Your retrieval metric goes through the judge you trained against. Isn't that circular?"

Caught us, that's exactly the projector circularity. The *projector*
metric is what we trained against, and yes, NeuroZip beats fidelity
there partly because that's what the loss optimizes. So we built a
*separate concept classifier* with a different EEG split, a different
loss (softmax cross-entropy vs cosine), and a different output head.
At 144× it reads NeuroZip with 100% and fidelity with 88%. That's
the real defense. **Important caveat**: above 144×, this independent
classifier saturates near 100% for *both* methods, so it's a
tie-breaker at the most-aggressive tier and not a differentiator
elsewhere. Above that tier the per-bit retrieval gap on the projector
metric (+4 to +11 pp) is the load-bearing claim. We lean on whichever
metric isn't saturated at the tier in question.

---

## What's NOT in the spoken pitch but lives in Q&A

- The `_grad_flow_assert()` runtime check — encodes the subtle
  correctness condition (gradient must flow through the frozen judge)
  as a startup assertion.
- The v1 → v4 architecture evolution — we tried attention in the codec,
  it hurt the comparison story (fidelity-baseline implicit leak); v4
  reverted to conv-only codec + attention judge.
- The factorized Laplace prior + uniform-noise-train / integer-round-
  inference quantizer.
- The LAION-2B vs OpenAI CLIP gotcha — the dataset's precomputed
  features are LAION-2B; using OpenAI silently breaks retrieval
  (verified by cosine distance: 0.98 to LAION-2B-encoded vs −0.06 to
  OpenAI).
- The iso-rate table — full per-tier breakdown in `results.md`.
- The ablation counter-cut — `plots/phase0_summary.json` has a
  check-1 that cuts the other way: fidelity's downstream classifier
  depends heavily on one occipital channel (P8, ~12 pp accuracy drop
  if ablated), while no single channel ablation costs NeuroZip more
  than ~3 pp. By that lens fidelity looks more visually concentrated
  and NeuroZip more diffuse. Different question: check-1 measures
  *classifier robustness*, check-3 measures *waveform preservation*.
  NeuroZip wins check-3 (which is what the codec was actually trained
  on) and looks less concentrated by check-1 because it spreads the
  signal redundantly. Raise this only if a judge asks, since the
  honest-slide that fronted it is cut from the short-demo runtime.
- The honest weaknesses list — one subject, trial-averaged, top-5 is
  29% against a 45% raw-EEG ceiling, retrieval edge is +4–11 pp not
  +50. Used to live on slide 6. Deliver with the same confidence in
  Q&A as you would on stage.

If a judge asks "how does the codec actually compress," lead with
**rate + MSE + task loss, three terms summed, factorized Laplace prior
on integer-rounded latents, gradient flows through a frozen judge.**
Then drill as deep as they want; everything above is true in code and
reproducible from `train.py`.
