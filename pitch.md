# NeuroZip — stage pitch (3:00 cut)

Runtime: ~2:45 spoken + 30 s for the two live-demo beats. ~15 s buffer.

This is the tightened version. Every claim aligned with
[`plots/phase0_summary.json`](plots/phase0_summary.json),
[`plots/phase1_bio_numbers.json`](plots/phase1_bio_numbers.json),
[`results.md`](results.md).

The long-form version with extra context, expanded slides, and the full
"why this wording matters" annotations lives on the `criticism-1` branch.
This file is the one you walk on stage with.

---

## Slide 1 — EEG trace, no title, no logo  ·  0:00 – 0:30  ·  LIVE DEMO

> One second of a human brain — 63 electrodes, someone looking at a
> photo. You can't read it. Watch.
> *(type "ostrich")*
> I'll find when they saw an ostrich — by typing the word.
> *(thumbnails populate)*
> We never trained on text. The model learned to align brain
> recordings with images; the language model brings the words for
> free.

## Slide 2 — two reconstructions side by side  ·  0:30 – 1:05  ·  LIVE DEMO

> Same one second, compressed two ways, same architecture. Left:
> trained for waveform fidelity — the default. Right: ours, trained
> for meaning.
>
> Search both. *(type "ostrich")* The fidelity one loses the ostrich.
> Ours keeps it.
>
> So what you're looking at doesn't live in the loud parts of the
> signal fidelity protects. It lives somewhere smaller — and more
> fragile.

## Slide 3 — channel × time heatmap  ·  1:05 – 1:40

> So we asked: where? We compressed 144-fold — throwing away
> everything but meaning — and looked at what survived. It's not
> smeared across the brain. It concentrates at occipitotemporal
> electrodes, biggest right at the N170, 150 to 200 milliseconds —
> the window the brain identifies objects.
>
> Compression found that for us. We used compression as a microscope.

## Slide 4 — `144×` and `100%`  ·  1:40 – 2:05

> And that meaning is tiny. At 144× compression, a separate
> classifier — different data, different loss, different output head —
> still reads the object the subject saw with **100% accuracy**. The
> fidelity baseline drops to 88. The "what you saw" signal is
> low-dimensional and robust.

## Slide 5 — one-sentence "how"  ·  2:05 – 2:30

> One sentence on the how: we put a **frozen judge** in the loop that
> scores whether the meaning survived, and train the compressor against
> it — so the model that decides what to keep is what lets you search
> by typing.

(This is the fallback for the original "fMRI replication" slot — we
haven't trained the cross-modal variant. If/when we do, this slide gets
swapped for "ran the exact same method on fMRI; same result; meaning
survives, fidelity doesn't.")

## Slide 6 — the honest slide  ·  2:30 – 2:50

> Where we're weak: one subject, trial-averaged. Our edge is four to
> nine points, not fifty. And we tried to catch ourselves cheating —
> that 100-versus-88 is an independent judge we never trained against.
> The win survives where compression actually matters.

Deliver this **with confidence, not apology.** It's the slide that makes
the rest credible.

## Slide 7 — close  ·  2:50 – 3:00

> NeuroZip: brain datasets two orders of magnitude smaller, searchable
> by language — and a finding that object identity in the brain is
> low-dimensional, localized, and robust. Type a word — find a thought.

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
- [ ] Rehearse the cut script against a stopwatch. Target 2:45 spoken
      + 0:30 demo = 3:15 max, with the 15 s buffer eating any slop.
- [ ] Drill slide 6 specifically. It's the highest-value, easiest-to-
      under-deliver line.
- [ ] Don't expand slide 5. Everything you cut goes in Q&A; that's where
      technical merit gets scored anyway.

---

## Strategic note (year-out, not for this stage)

The advisor flagged that this is the neuroscience project at a
cancer-proteomics hackathon (QBI's orbit: Zaro lab, Bar-Peled lab,
chemical biology / proteomics / DrugMap). CysTeam-type projects win
because they're domain-native to that room — built by and for the
judges' own labs.

**This pitch is calibrated for the EEG identity, and that caps the
realistic placement at top-3.** First place at QBI requires re-aiming
the same method at QBI-native data: Cell Painting, mass-spec proteomics,
Perturb-seq, cysteine ligandability. Same technical contribution
(frozen-judge task-aware compression + search-by-meaning), different
substrate, co-developed with a sponsoring lab.

If you're optimizing this year's win, that pivot is the play — but it's
weeks-to-months of work, not a pitch tweak. Out of scope for criticism-2.

---

# Q&A prep — the five hardest questions

## "It's n=1. Does this generalize to other subjects?"

The dataset has ten subjects; we trained on one because the iteration
loop for the codec contribution is faster that way. The pipeline is
subject-agnostic — one flag in `data.py` switches it. We expect
cross-subject training to *help* the absolute numbers (~+8–12 pp from the
THINGS-EEG literature) and we expect the *codec contribution* — the gap
between NeuroZip and fidelity — to carry over, because it's about which
structure is task-relevant, not about which subject. The lesson we wanted
to isolate is "task awareness preserves more per bit"; adding subject
variance would have confounded that ablation.

## "It's trial-averaged over 80 reps. Doesn't that hide the noise?"

Yes — averaging substantially denoises EEG. Single-trial retrieval is
harder and we don't claim to solve it. The published THINGS-EEG numbers
(NICE 33% top-1, Wu CVPR 2025 35%) use the same trial-averaged protocol;
it's the standard eval surface so codec methods are comparable. The
codec contribution — the gap between fidelity and NeuroZip — wouldn't
change qualitatively in single-trial because both methods see the same
noisy input. The absolute numbers would drop together.

## "Aren't you just re-finding the N170?"

Yes, and that's the point. We're not claiming to discover the N170 —
we're claiming compression is a non-trivial tool for localizing
task-relevant signal. If our codec hadn't preserved the N170
disproportionately, the method would be wrong. The fact that an
unsupervised localization pipeline rediscovers known neurophysiology is
the validation that the method works on this substrate. The
investigational application is novel paradigms (memory, imagery) or
novel modalities (fMRI, intracranial) where we don't already know the
answer.

## "Who actually needs EEG compression?"

Three places, in order of pain. *Dataset sharing*: THINGS-EEG itself was
re-released in fp16 to halve its size — the storage pressure is real for
exactly this kind of corpus. *Wearable EEG*: consumer headsets stream
multi-hour recordings off-device with constrained battery and bandwidth.
*Searchable archives*: brain-imaging labs accumulate decades of session
data; "find me trials where the subject saw faces" is currently a manual
labeling job. Our contribution turns that into a text query. None of this
is blocked today, but all of it is expensive enough that people work
around it.

## "Your retrieval metric goes through the judge you trained against. Isn't that circular?"

Caught us — that's why slide 4 exists. The *projector* metric is the one
we trained against, and yes, NeuroZip beats fidelity there partly because
that's what the loss optimizes. The *held-out concept classifier* on slide
4 is the answer: different EEG split (60 of the 80 test reps), different
loss (softmax cross-entropy vs cosine to CLIP), different output head. At
144× compression it reads NeuroZip with 100% and fidelity with 88%. That
12-point gap is measured against a judge whose only relationship to our
training was that the codec was never asked to satisfy it. It's the
defense against "you're grading your own homework."

---

## What's NOT in the spoken pitch but lives in Q&A

- The `_grad_flow_assert()` runtime check — encoded the subtle
  correctness condition (gradient must flow through the frozen judge)
  as a startup assertion
- The v1 → v4 architecture evolution — we tried attention in the codec,
  it hurt the comparison story (fidelity-baseline implicit leak); v4
  reverted to conv-only codec + attention judge
- The factorized Laplace prior + uniform-noise-train / integer-round-
  inference quantizer
- The LAION-2B vs OpenAI CLIP gotcha — the dataset's precomputed features
  are LAION-2B; using OpenAI silently breaks retrieval (verified by
  cosine distance — 0.98 to LAION-2B-encoded vs −0.06 to OpenAI)
- The iso-rate table — full per-tier breakdown in `results.md`

If a judge asks "how does the codec actually compress," lead with
**rate + MSE + task loss, three terms summed, factorized Laplace prior
on integer-rounded latents, gradient flows through a frozen judge**.
Then drill as deep as they want; everything above is true in code and
reproducible from `train.py`.
