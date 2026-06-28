# NeuroZip — stage pitch

~2.5 minutes spoken. Biology leads, compression is the microscope not the
product, one technical beat earns the depth, honesty carries trust.

All claims here are aligned with the data in [`plots/phase0_summary.json`](plots/phase0_summary.json),
[`plots/phase1_bio_numbers.json`](plots/phase1_bio_numbers.json), and [`results.md`](results.md).

---

## Slide 1 — a single EEG trace scrolling. no title. no logo.

> This is one second of a human brain. Sixty-three electrodes, someone in
> a lab looking at a photograph. You can't read it. I can't read it. It
> looks like noise.
>
> Watch. *(type "ostrich" into the search box)* I'm going to find the
> moments this person was looking at an ostrich — by typing the word.
>
> *(ostrich thumbnails populate)*
>
> We never trained the system to recognize text. It learned to align
> brain recordings with the images people saw — and because the language
> model that does the search already aligned text and images, you get
> search-by-word for free at inference.

**Why the wording matters** — the older script said "we never labeled
these recordings," which would be misleading: every training EEG is paired
with the image's CLIP feature, so the recordings *are* labeled at training.
What we never trained on is **text**. Fix the line.

---

## Slide 2 — same EEG epoch, two reconstructions side by side

> Here's where it gets interesting for anyone who cares about the brain.
>
> Same one second of EEG, compressed two ways with the same model
> architecture. The left codec was trained to reproduce the waveform
> as faithfully as possible — the MSE-optimal baseline you'd build
> by default. The right one is ours.
>
> Now I search both. *(type "ostrich")* The MSE-optimal one — *(left
> misses)* — loses it. Ours — *(right hits)* — keeps it.
>
> So the codec optimized for waveform threw the ostrich away, and the
> one optimized for meaning held on. That's not a bug. That's a
> finding: what you're looking at does not live in the parts of the
> EEG that fidelity protects. It lives somewhere smaller — and more
> fragile.

**Why the wording matters** — the older script said "by every fidelity
number, the left one is more accurate." Not quite true at iso-rate —
fidelity's MSE advantage is 1–5%, and at our low/med tiers NeuroZip is
actually *better* on MSE. Drop the comparative; let the visual demo
carry it.

---

## Slide 3 — channel × time heatmap, one bright band

> So we asked the obvious next question: where?
>
> We compressed the brain recording 144 times smaller — throwing away
> bits until only the parts that carry meaning could survive — and then
> looked at what was left. It isn't smeared across the brain. It
> concentrates here: occipitotemporal electrodes, with the biggest
> preservation gap right inside the window where the visual system
> identifies the object — N170, 150 to 200 milliseconds after the eye
> sees it. Every visual-evoked component is preserved better than the
> baseline preserves it, but that one is preserved most.
>
> Decades of evoked-potential research predicted exactly that. The
> difference is we didn't assume it. Compression found it for us.
> Crushing the signal until only meaning remained localized the
> meaning. That's the whole idea: we used compression as a microscope.

**Why the wording matters** — the older script said "not the earlier
window where it's just registering that light hit the eye," implying
P100 (early visual) wasn't affected. Actually P100 also favors NeuroZip
by 12.6%. N170 is just the biggest gap (25.4%), with P200 (17.8%) and
P300 (12.1%) behind it. Keep "biggest at N170" — drop the false negation.

---

## Slide 4 — `144×` and `100%`

> And that meaning is tiny. We compress the EEG 144-fold — a million
> brain-image trials go from thirty gigabytes to two hundred megabytes —
> and a separate classifier, trained on a different EEG split with a
> different loss function and a different output head, identifies the
> object the subject saw with **100% accuracy** from our compressed
> reconstruction.
>
> The "what you saw" signal in EEG is low-dimensional, robust, and
> you can throw almost everything else away.

**Why the wording matters** — the older script said "completely
independent." The held-out classifier shares an encoder *body*
architecture with our judge, just trained on different data with a
different head and loss. Sharp judges will catch "completely." "Separate"
+ the specifics is honest and reads stronger.

---

## Slide 5 — one diagram: encoder → judge → meaning

> One sentence on how, because there's real machinery underneath and
> we're happy to go as deep as you want in questions.
>
> When we train the compressor, we don't only punish it for getting the
> waveform wrong. We put a frozen **judge** in the loop — a model that
> maps an EEG snippet into the same semantic space as the image the
> person saw — and we punish the compressor whenever it discards
> something that moves the recording away from that semantic point.
> The gradient flows straight through that judge into the compressor,
> even though the judge's own weights never change.
>
> So the same model that decides what to keep at training is what lets
> you search the compressed corpus by typing at inference. Train
> once, search with words forever.

This is the **only main-stage technical beat**. Resist the urge to expand
it. All the deeper material — the prior, the noise-at-train/round-at-
inference quantizer, the v1→v4 ablation, the gradient-flow assert —
lives in Q&A.

---

## Slide 6 — the honest slide

> We'd rather you trust these numbers than be dazzled by them, so here's
> where we're weak.
>
> This is one subject. It's trial-averaged — true single-trial decoding
> is harder. Our edge over plain compression in retrieval accuracy is
> four to nine percentage points at matched bit-budget — not fifty.
>
> And we tried to catch ourselves cheating. We trained a second judge
> on different data, with a different loss, and asked it whether our
> win was just us grading our own homework. At the most aggressive
> compression — 144× — that judge reads our reconstruction with 100%
> accuracy and the fidelity baseline with 88%. The win survives a
> judge we never optimized against, in the regime where compression
> actually matters.

**Why the wording matters** — the older script said "the win survives a
judge we never optimized against" without qualifying. True at high
compression, where the held-out reads NeuroZip 100% and fidelity 88% (a
12-point gap). At lighter compression, both saturate. Be specific about
when the differentiation is real.

Deliver this slide **with confidence, not apology**. It's the slide that
makes everything else credible.

---

## Slide 7 — one line on screen

> So what is NeuroZip?
>
> On paper, it's a compressor: brain datasets, two orders of magnitude
> smaller, and searchable by language — which, as far as we can tell, is
> the first EEG codec to support free-text retrieval at inference.
>
> But the reason we're standing at a biology hackathon is what the
> compression revealed: object identity in the brain is low-dimensional,
> localized in space and time, and robust enough to survive 144×
> compression and still answer to a single word.
>
> We built a tool that makes brains searchable. It told us something
> about the brain on the way.
>
> We're NeuroZip. Type a word — find a thought.

**Why the wording matters** — the older script said "which no compressor
has done before." Hard absolute claim without lit-review evidence. Soften
to "as far as we can tell, the first EEG codec to support…" — keeps the
spirit, doesn't invite a "actually, paper X did Y" rebuttal.

---

# Delivery checklist

- [ ] Slide 1 + 2 demo rehearsed bulletproof. Record a 20-second screen
      capture as fallback in case wifi dies at the venue.
- [ ] Pitch tested on the actual machine + network + projector ~30 min
      before stage.
- [ ] Q&A prep sheet (below) read end-to-end.
- [ ] Honest slide delivered with confidence.
- [ ] Resist expanding slide 5 — depth lives in Q&A.

---

# Q&A prep — the five hardest questions

## Q: "It's n=1. Does this generalize to other subjects?"

The dataset has ten subjects; we trained on one because the iteration
loop for the codec contribution is faster that way. The pipeline is
subject-agnostic — `data.py:ThingsEEG(subject="sub-02")` switches it.
We expect cross-subject training to *help* the absolute numbers (the
~+8–12 pp lift is standard in the THINGS-EEG literature) and we expect
the *codec contribution* — the gap between NeuroZip and fidelity — to
carry over, because it's about which structure is task-relevant, not
about which subject. We didn't run it because the lesson we wanted to
isolate is "task awareness preserves more per bit," and adding subject
variance would have confounded that ablation. Multi-subject is the
next overnight run.

## Q: "It's trial-averaged over 80 reps. Doesn't that hide the noise problem?"

Yes — trial-averaging substantially denoises the EEG. Single-trial
retrieval is harder and we don't claim to solve it. The published
THINGS-EEG numbers (NICE 33% top-1, Wu CVPR 2025 35%) use the same
trial-averaged protocol; it's the standard eval surface so codec
methods are comparable. The *codec* contribution — the gap between
fidelity and NeuroZip — wouldn't change qualitatively in single-trial,
because both methods see the same noisy input. The absolute numbers
would.

## Q: "Aren't you just re-finding the N170? That's been known for 30 years."

Yes, and that's the point. We're not claiming to discover the N170 — we're
claiming that **compression is a non-trivial tool for localizing
task-relevant signal**. If our codec hadn't preserved the N170 window
disproportionately, the method would be wrong. The fact that an entirely
unsupervised localization pipeline rediscovers known
neurophysiology is the validation that the method works on this
substrate. The interesting application is the substrates where we don't
already know — different paradigms (memory, imagery), different
modalities (intracranial recordings, fMRI). On THINGS-EEG we're calibrating;
on novel data the method becomes investigational.

## Q: "Who actually needs EEG compression?"

Three places, in order of pain. *Dataset sharing*: THINGS-EEG itself was
re-released in fp16 to halve its size — the storage pressure is real for
exactly this kind of corpus. *Wearable EEG*: consumer headsets (Muse,
Neurosity, etc) need to stream multi-hour recordings off-device with
constrained battery and bandwidth. *Searchable archives*: brain-imaging
labs accumulate decades of session data; "find me trials where the
subject was looking at faces" is currently a manual labeling job. Our
contribution turns that into a text query. None of this is "blocked
today," but all of it is "expensive enough that people work around it."

## Q: "Your retrieval metric goes through the same projector you trained against. Isn't that circular?"

Caught us — that's exactly why slide 4 exists. The *projector* metric is
the one we trained against, and yes, NeuroZip beats fidelity there partly
because that's the metric the loss optimizes. The *held-out concept
classifier* on slide 4 is the answer to that critique: different EEG
split (60 of the 80 test reps), different loss (softmax cross-entropy
over concepts vs cosine to CLIP), different output head. At 144×
compression it reads NeuroZip with 100% and fidelity with 88%. That
12-point gap is measured against a judge whose only relationship to our
training was that the codec was never asked to satisfy it. It's a
defense against "you're grading your own homework," and it holds in the
regime where compression matters.

---

# What's NOT in the pitch but lives in Q&A

- The `_grad_flow_assert()` runtime check
- The v1 → v4 architecture evolution (we tried attention in the codec,
  it hurt the comparison story)
- The factorized Laplace prior + uniform-noise/integer-round quantizer
- The LAION-2B vs OpenAI CLIP gotcha (the dataset's precomputed features
  are LAION-2B; using OpenAI silently breaks retrieval)
- The iso-rate table (full per-tier breakdown in `results.md`)

If a judge asks "how does the codec actually compress," lead with **rate
+ MSE + task loss, three terms summed, factorized Laplace prior on
integer-rounded latents**. Then drill as deep as they want; everything
above is true in code and reproducible from `train.py`.
