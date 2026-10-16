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
