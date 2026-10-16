# NeuroZip: Task-Aware Neural Compression of Brain Recordings for Searchable Neuroimaging Archives

**Project co-leads:** Rian Butala (rianbutala@gmail.com) and Avinash Senthil (reachavinashs1@gmail.com)
**Origin:** QBI Hackathon 2026 (UCSF), 3rd place, Team NeuroZip | **Duration:** 9 months | **Requested:** $8,600

## Justification: why do this

Neuroimaging datasets that pair brain recordings with sensory stimuli now reach millions of trials and are the substrate for decoding and brain-computer-interface research. Storing them is expensive: one million EEG epochs is roughly 30 GB at float16, and THINGS-EEG was re-released in reduced precision purely to halve its footprint. The standard remedy, lossy compression that minimizes waveform reconstruction error, is silently harmful: it preserves the loud, high-variance parts of the signal while discarding the low-dimensional decodable content that is the reason the data were recorded. No existing codec has an objective that knows what a brain recording is for, so large archives are neither cheap to store nor searchable by content.

## What we will do

NeuroZip trains a codec against a frozen brain-to-CLIP projector that scores whether stimulus identity survived compression, so gradient flows through the frozen judge into the decoder. The same judge encodes free-text queries at inference, making the compressed corpus searchable in natural language. Five workstreams:

1. **Generalization.** Move from one subject to all ten THINGS-EEG subjects. Add leave-one-subject-out cross-subject training and single-trial (non-averaged) evaluation to show the per-bit retrieval advantage is not an artifact of 80-repetition averaging. Report per-subject variance, not just pooled means.

2. **Statistical rigor.** Formalize matched-bitrate rate-distortion comparison against fidelity-only baselines and published EEG decoders (NICE, Wu et al. 2025). Extend the independent-classifier control and the 10,000-permutation spatial test to every tier and subject, with multiple seeds and confidence intervals.

3. **Architecture and scaling.** The frozen-judge design is modality-agnostic and portable: the codec never touches CLIP, only a small projector trained once per modality. We will test:
   - (a) a learned entropy model with hyperprior conditioning to replace the factorized Laplace prior;
   - (b) multi-judge training, summing task terms from several frozen projectors (image-CLIP, text-CLIP, category classifier) so several kinds of meaning survive at once;
   - (c) a streaming/causal variant for continuous recordings, which is what wearable EEG produces;
   - (d) progressive tiers, one bitstream decodable at several quality levels.

   Crucially we will characterize the method across the full deployment range: from edge and CPU inference up to large-scale models on very large corpora. The edge case points at real-world use beyond research archives: continuous wearable biosignals (EEG, heart rate) today must be offloaded to a paired phone because the raw stream is too large to keep on-device, but a lightweight task-aware codec compressing at the source could keep far more on-device and transmit far less while preserving the parts that carry meaning. Showing the objective holds from a wearable to a big decoder over multi-million-trial archives is what turns a hackathon result into a method others adopt.

4. **Modality transfer.** Apply the identical objective where the answer is not known in advance, in priority order:
   - **NSD** (Natural Scenes Dataset, 7T fMRI): strongest first choice because it is large, public, and has precomputed CLIP-aligned features so only the projector changes;
   - **THINGS-MEG**: shares our exact stimulus set and isolates modality while holding semantics fixed;
   - **Public iEEG corpora**: high signal-to-noise ratio tests whether gains come from denoising or genuine task-awareness.

   We will complete NSD and THINGS-MEG and treat iEEG as a stretch goal.

5. **Integration with active UCSF labs.** We want to embed our compression layer directly into experiments that groups at UCSF are already running, rather than only testing on public archives. By adding NeuroZip as a drop-in layer in a live decoding, BCI, or stimulus-response study, we learn the real use cases, failure modes, and storage and latency constraints that a public dataset never surfaces, and the host lab gets a smaller, searchable version of their own data in return.

## Connection to the hackathon project

This directly continues the 3rd-place hackathon project by the same two people. On one subject, at 144x compression, the task-aware codec improved top-5 retrieval by 4 to 11 points over a fidelity baseline of matched architecture and budget, at under 5% cost in reconstruction MSE, and the preserved signal localized to occipitotemporal electrodes and the N170 window (p < 0.001, 10,000 permutations). The grant converts that prototype into a multi-subject, multi-modality, statistically controlled study spanning the full edge-to-large-scale range.

## Mentorship and connections we are seeking

This is the part of the grant we most want to maximize. We have published a few papers before, but they were all analyses of existing architectures; turning this into a peer-reviewed publication is different, because it requires usability feedback from the professors and researchers who would actually use the codec, and that communication is exactly what the QBI mini-grant is set up to provide. We are asking for:

- a faculty mentor for regular guidance on framing, experimental design, and venue choice (ideally 1 to 2 hours per month);
- a senior researcher or postdoc for closer review of the compression and statistics and hands-on help with figures and manuscript writing, which the program explicitly offers;
- introductions to one or two UCSF labs running live neural-recording experiments, to provide a concrete setting for workstream 5.

## Timeline (9 months)

| Months | Work |
|---|---|
| 1–2 | All 10 THINGS-EEG subjects, cross-subject and single-trial harness, secure mentor, begin lab outreach. |
| 2–4 | Matched-bitrate benchmarking and controls, entropy-model and multi-judge extensions. |
| 4–6 | NSD and THINGS-MEG transfer, large-scale scaling runs, partner-lab integration, figures, pre-print at month 6. |
| 7–9 | Venue submission, iEEG stretch goal, open codec and demo release. |

---

## Progress log

### 2026-10-15 — Workstream 1 kickoff (all 10 subjects, single-trial, LOSO)

Full tables: [tests/results/SUBJECTS.md](tests/results/SUBJECTS.md). Harnesses: `tests/exp_subjects.py` (`--holdout` for the classifier check), `tests/exp_loso.py`, `tests/exp_shipped_single.py`; summary: `tests/summarize_subjects.py`.

**Setup.** All 10 THINGS-EEG subjects downloaded. Fixed a loader bug: normalization stats were cached in one shared file, so every subject would have been z-scored with sub-01's stats; now cached per subject (sub-01 keeps the legacy path the shipped checkpoints use). Added `single_trial()` and `k_averaged(k)` test views next to `trial_averaged()`.

**Protocol.** Per subject: train that subject's frozen judge (12 epochs, fixed in advance for all subjects), then per rate tier train a fidelity codec and branch it into fidelity-continued vs. NeuroZip (task loss), same architecture, init and step count. Score on avg80 (paper protocol), avg4, and single trials (16,000 queries). Matched-bitrate comparisons interpolate each subject's fidelity rate curve at NeuroZip's actual bpp. Stats are paired across subjects (t-test and Wilcoxon).

**Results.**
- *Within-subject, 10 subjects, 3 seeds:* single-trial image top-5 at matched bitrate, NeuroZip beats fidelity by +6.8 to +7.0 pts (seed 0: 18.4% vs. 11.3%), +7.6 to +7.8 (seed 1), and +6.5 to +7.6 (seed 2). That holds on **10/10 subjects in every tier and every seed** (paired p between 1e-8 and 4e-5). On the 80-rep averages the gain is smaller and noisier (+2 to +5 pts, 7–9/10 subjects). The advantage is *larger* without averaging, so it is not an artifact of the 80-repetition protocol.
- *Non-circular check, all 10 subjects (seed 2):* a per-subject concept classifier, trained on raw EEG from 60 test reps and never exposed to the codec, scores reconstructions on the 20 reps it never saw. Single-trial top-1: NeuroZip 14.7–15.5% vs. fidelity 8.2–9.0% (raw 19.6%, chance 0.5%; 10/10 subjects, p ≈ 1e-7). 20-rep averages: 61.5–62.2% vs. 42.7–45.8% (raw 95.8%). The task-aware gain is not just the codec gaming its own judge.
- *Cost:* NeuroZip's single-trial reconstruction MSE is ~4–7% higher than fidelity's (e.g., 0.74 vs. 0.71); on 80-rep averages its MSE is lower (0.038 vs. 0.049).
- *Shipped v4 checkpoints, sub-01, single trials:* NeuroZip 14.6–15.0% vs. fidelity 8.2–9.9% image top-5 (judge). The independent held-out classifier agrees when scored on the 20 test reps it never trained on: single-trial top-1 9.2–9.9% vs. 4.6–6.2% (raw 8.5%, chance 0.5%); 20-rep averages 82–88% vs. 47–72%. NeuroZip's lowest-rate tier (0.11 bpp) beats fidelity's highest (0.25 bpp).
- *Found and fixed a leak in the classifier control:* the shipped held-out classifier trains on 60 of the 80 test reps per concept, so scoring it on all 80 reps (as a naive single-trial eval would) puts 75% of the scored trials in its training set. `exp_shipped_single.py` scores it only on the 20 unseen reps. `evaluate.py` scores the classifier on the 80-rep average, so it has the same leak for the headline holdout number; worth re-checking.
- *Leave-one-subject-out:* a judge trained on 9 subjects reaches only 12.9% avg80 / 7.9% single-trial top-5 on the held-out subject (vs. 39% / 20% within-subject). NeuroZip still wins on single trials for 10/10 held-out subjects, but by only +0.8 pts; no difference on averaged views. Cross-subject transfer is limited by the judge, not the codec.

**Caveats.**
- Judge-based retrieval is circular (same judge as training). The classifier check above addresses this, but the classifier is trained on test-split reps of the same 200 concepts. It is independent of the codec, not of the concepts.
- Judges are short-trained (12 epochs) and vary a lot by seed (sub-01 avg80 top-5: 24.5% seed 0 vs. 39.5% seed 1). The epoch count was chosen from sub-01's test curve, then frozen for all subjects.
- Rates come from the modeled factorized prior (≈0.12–0.25 bpp, ~65–130×), not a range-coded bitstream.

**Next.**
1. Re-check `evaluate.py`'s headline holdout number, which scores the classifier on 80-rep averages that include its 60 training reps.
2. Cross-subject judge: subject embeddings or per-subject adapter layers, since LOSO is judge-bound.
3. The 10,000-permutation spatial test per subject; comparison against published decoders (NICE, Wu et al. 2025) (WS2).
4. Mentor and lab outreach (months 1–2, non-code).
