# NeuroZip — the long version (read this if you have ten minutes)

This is the plain-language companion to [`pitch.md`](pitch.md), the tight
3-minute thing we say on stage. This file slows down, explains every word, and
assumes you have *not* spent the last decade in a machine-learning lab. If
you're a judge who wants to actually understand what we built, start here.

We're high schoolers. Here's our promise: everything below is explained the
way we had to explain it to ourselves first. If a sentence needs a PhD to
read, we wrote it wrong.

Every number in this document comes straight from the files we ran:
[`plots/phase0_summary.json`](plots/phase0_summary.json),
[`plots/phase1_bio_numbers.json`](plots/phase1_bio_numbers.json),
[`plots/phase2_permutation.json`](plots/phase2_permutation.json), and
[`results.md`](results.md). All the biology numbers are measured at the
"144× compression" setting so they line up with the headline.

---

## Part 0 — the five words you need first

Here are the only five ideas you need. We use them the whole way down.

**EEG.** Electroencephalography. You put electrodes on someone's scalp and
record the tiny voltages the brain makes, many times a second. Our dataset has
**63 electrodes** sampled **250 times per second**. One "epoch" is one second
of that: a 63-by-250 grid of numbers. It looks like noise. You genuinely
cannot read it by eye.

**The dataset.** It's called THINGS-EEG. Scientists showed one person
thousands of photos of everyday objects (an ostrich, a basketball, a
dolphin…) and recorded their EEG each time. So every one-second brain
recording comes with a label: *this is the brain while the person looked at an
ostrich.* That pairing makes everything else possible.

**Compression / a "codec."** A codec is a program that shrinks data and then
rebuilds it. JPEG is a codec for photos; MP3 is a codec for music. It throws
away the parts you supposedly won't miss so the file gets smaller. The hard
part is choosing what to throw away. That choice is the whole story here.

**CLIP.** A famous AI model that turns both **images** and **text** into
points in the same "meaning space." In CLIP's world, the photo of an ostrich
and the word "ostrich" land near each other. This is what later lets you
search brain recordings by typing a word. We used CLIP frozen, off the shelf.
We did not build it.

**An "embedding."** A list of numbers (here, 512 of them) that stands for the
meaning of something. CLIP turns a picture into an embedding and a word into
an embedding, and nearby embeddings mean similar things. When we say "the
meaning of the EEG," we mean the embedding you'd get for the picture the
person was looking at.

That's it. Everything below is built from those five pieces.

---

## Part 1 — the whole project in one paragraph

Normal compression tries to rebuild a signal so it *looks* like the original.
For brain data, we think that's the wrong goal. We trained a codec to instead
keep whatever lets you still figure out *what the person was looking at*, even
after throwing away more than 99% of the data. Here's the payoff: because we
measure "meaning" with CLIP, you can afterward **type a word and pull up the
brain recordings that match it** — search a pile of compressed brain data with
plain English. And when we looked at *which* parts of the brain signal our
codec fought hardest to keep, they were exactly the parts brain scientists
already know carry "what object am I seeing." That's a good sign we kept the
right thing.

Now the slow version.

---

## Slide 1 — "type a word, find a thought" *(this is a live demo)*

**What you'll see us do.** A second of real EEG is on screen. It looks like
scribbles. We type "ostrich" into a box and hit search. The system pulls up
the brain recordings that best match the idea of an ostrich, and the ostrich
ones come up.

**Why it works.** We never trained the system on text. We trained it to line
up brain recordings with the **pictures** the person saw, using CLIP. CLIP
already knows the word "ostrich" and the picture of an ostrich live in the
same place in meaning-space. So once our model can hit the picture's meaning
from the EEG, the word comes along automatically, because the picture and the
word were already neighbors. We got text-search without asking for it. That's
CLIP doing the heavy lifting, and we're happy to say so.

**The honest size of it.** Typing a word lands the right concept in our top 5
about **29%** of the time. For comparison, even the *uncompressed* brain data
only gets there about **45%** of the time. Brains are noisy and this is hard.
29 out of a possible 45 is the real number. We'd rather show you a real 29
than a made-up 95.

---

## Slide 2 — same brain recording, two ways to compress it

This is the heart of the project, so we'll go slow.

We build **two** compressors with the **exact same internal design**. The only
difference is what we reward them for during training:

- **The "fidelity" codec** (the normal one): rewarded only for making the
  rebuilt signal look like the original waveform. This is what any engineer
  builds by default.
- **NeuroZip** (ours): rewarded for the same thing, plus one extra rule — keep
  whatever lets us still recognize what the person was looking at.

Then we give both the same one second of brain data, compress it down hard,
and search.

**What happens:** the normal codec loses the ostrich. Ours keeps it.

**Why that's surprising.** The normal codec did its job. It faithfully kept
the biggest, loudest parts of the signal, because that's what makes the
rebuild look most like the original. But the information about *what you're
seeing* doesn't live in the loud parts. It lives in smaller, more fragile
parts of the signal, and a fidelity codec happily throws those away to save
space. Our extra rule tells the codec to protect them. So at the same file
size, ours keeps the meaning and the normal one keeps the noise.

The whole thesis in one line: **for brain data, what's easy to compress and
what's worth keeping are two different things.**

---

## Slide 3 — *where* and *when* in the brain the meaning lives

We could only ask this because we had two codecs to compare: when our codec
decides what to protect, *which parts of the brain signal does it protect
hardest?*

EEG has a where (which electrode, meaning which part of the scalp) and a when
(how many milliseconds after the photo appeared). We checked both.

**Where.** Our codec preserves the electrodes over the **back of the head**
(the visual cortex, literally where vision is processed) about **21% more
tightly** than the normal codec. Over the rest of the scalp, the advantage is
only about **5%**. So it cares about the visual area roughly **4× more** than
the rest. It spends its effort right where vision happens.

**When.** The biggest gap shows up around **150–200 milliseconds** after the
photo appears. Brain scientists call that window the "N170," and they've known
for ~30 years that it's when the brain is busy identifying an object. With no
hint from us, our codec works hardest to preserve exactly that window.

**"How do you know that's not luck?"** Fair. So we ran a stress test: we
randomly shuffled which electrodes count as "visual" **10,000 times** and
checked how often a random group looked as visual-focused as the real one.
Answer: almost never, about 3 times in 10,000. In statistics that's
**p < 0.001**, which means it's very unlikely to be a coincidence.

**The part we're careful about.** We did **not** discover the N170. It's
textbook, and we couldn't have discovered it, because we *trained* the codec
to keep object-identity information. So of course it keeps the object-identity
part of the brain. Finding that it did is a **check that our method works**,
and we read it that way. If our codec had protected the front of the head
instead, that would have meant something was broken. Being precise about this
difference is part of doing it honestly.

---

## Slide 4 — the actual engineering claim: more meaning per bit

Slides 1–3 are the visible, cool stuff. This slide is the claim we'd defend in
a paper.

"Bits" means file size. The real test is this: **for the same number of bits,
does NeuroZip keep more meaning than the normal codec?** If you just spend more
storage you can keep more of anything; that's not clever. Keeping more *per
bit* is the clever part.

It does. At the 144× compression setting, NeuroZip lands the right concept in
the top 5 about **10 percentage points more often** than the fidelity codec at
a matched bit budget. Across the different compression settings the gap runs
from about **+4 to +11 points**. Same file size, more meaning kept.

**The independent referee.** A skeptic could say we measured "meaning" with
the same CLIP-based tool we trained against, so of course we win — we graded
our own homework. Good objection. So we built a **completely separate** judge:
a different model, trained on a different slice of the data, with different
math, that the codec never tried to please. We asked it to identify the object
from the compressed brain data. At the most aggressive compression, it reads
**NeuroZip correctly 100% of the time and the normal codec only 88%**. That's
an outside referee agreeing with us.

**The honest footnote.** That outside referee is so good that at gentler
compression settings it scores ~100% for *both* codecs and can't tell them
apart, because both kept enough. So we don't pretend it proves our case
everywhere. It's the tie-breaker at the extreme setting, and the "more meaning
per bit" number carries the rest. We quote whichever ruler isn't maxed out at
the setting we're talking about.

---

## Slide 5 — how it actually works, in one breath

One mechanical idea, plainly.

While the codec is learning, we sit a **frozen "judge"** next to it. The
judge's only job is to look at the rebuilt brain signal and answer: could I
still tell what the person was looking at? When the answer gets worse, the
codec gets a penalty and adjusts. "Frozen" means the judge itself never
changes, so it's a fixed ruler the codec has to satisfy.

Here's the neat part: **that same judge is the search engine.** The model we
use during training to score "did the meaning survive" is the same model that,
at the end, takes your typed word and finds the matching brain recordings. One
model, two jobs: the teacher while we build the codec, the librarian when you
search.

*(We were going to put a second result here — the same method on fMRI, a
different kind of brain scan — to show this works beyond EEG. We didn't finish
training that, so we won't show a slide that implies we did. If we ever do, it
earns its place by being real.)*

---

## Slide 6 — where we're weak (and why we still trust it)

We'd rather hand you the weaknesses than have you find them.

- **One person.** We trained and tested on a single subject. The code can
  switch subjects with one setting; we just didn't run all ten, because we
  wanted to test one idea cleanly (does the meaning-rule help?) without mixing
  in "do different brains behave differently?"
- **Averaged trials.** Each concept was shown ~80 times and we averaged those
  repeats to cut the noise. Single-shot brain reading is much harder and we do
  **not** claim to solve it. We average because that's the standard way this
  dataset is scored, so our numbers can be compared to published work.
- **Modest absolute numbers.** 29% top-5, against a 45% ceiling. We lead with
  the per-bit improvement over the baseline because that's the honest,
  defensible claim.
- **A counter-cut in our own data.** There's a second way to measure how
  localized the signal is, and by that measure the normal codec looks more
  concentrated than ours (it leans heavily on one electrode, "P8," worth ~12
  points; ours leans on no single electrode more than ~3). We raise this
  ourselves, because the two measurements answer different questions and a
  judge who digs will find it. Better that we hand it over.

Everything we *do* claim, we claim because we tried to break it and couldn't.

---

## Slide 7 — the close

NeuroZip makes brain datasets **roughly 144× smaller** while keeping them
**searchable by plain English.** The engineering claim is "more recognizable
meaning kept per bit than a normal compressor." The validation is that the
parts of the brain signal our method works hardest to keep are exactly the
parts vision scientists would predict. The brain science here is old and well
known. We used it as a witness that our compressor keeps the right thing.

Type a word. Find a thought.

---

# Questions a judge will (and should) ask

We wrote these out because we'd rather rehearse the hard ones than dodge them.

### "It's one person. Does this generalize?"

The dataset has ten people; we used one on purpose, to keep the experiment
clean (one variable: does the meaning-rule help?). Switching subjects is one
line of code. We expect the absolute scores to go up with more data (the
research literature sees +8–12 points), and we expect our advantage over the
baseline to survive, because that advantage comes from which parts of the
signal carry meaning, which is a fact about vision rather than about this one
person.

### "You averaged 80 repeats. Isn't that hiding the real difficulty?"

Yes, averaging makes the signal much cleaner, and single-shot reading is the
genuinely hard case we don't claim to solve. But every published result on
this dataset averages the same way, so our numbers can be compared to the
field. And our advantage over the normal codec would still be there in the
harder case: both codecs get noisier inputs together, and the gap between them
comes from what they choose to keep, not from how clean the input is.

### "Aren't you just re-finding the N170, which is 30 years old?"

Yes, and that's the point — it's our sanity check. We trained the codec to keep
object-identity information, so it keeping the object-identity brain region is
the expected result, and a good one: it means the method works. A real
discovery could live somewhere the answer *isn't* already known — memory
tasks, mental imagery, or other kinds of brain scans. That's future work, and
we won't dress it up as done.

### "Who actually needs to compress brain data?"

Three groups, roughly in order of pain. **Sharing big datasets:** this very
dataset was re-released in a half-size format just to make it movable, so the
pressure is real. **Wearable headsets:** consumer brain-sensing devices have
to stream hours of data off a small battery. **Searchable archives:** labs sit
on decades of recordings, and "find me the trials where they saw faces" is
currently a manual chore. Our method turns that into a typed query. None of
this is a five-alarm fire today, but all of it is annoying enough that people
build workarounds.

### "Your main score uses the same tool you trained against. Circular much?"

Caught. That's a real concern, and it's why we built the independent referee
(Slide 4): a separate model, a separate data slice, separate math, never
optimized against. At the hardest compression it backs us (100% vs 88%). The
honest limit: at easier settings that referee maxes out for both codecs, so
there we rely on the "meaning per bit" number instead. We always quote
whichever ruler isn't pinned at its maximum for the setting in question.

---

# For the judge who wants to go deeper

Not on the slides, but true and in the code if you ask:

- **A built-in safety check** that makes sure the learning signal actually
  reaches the codec through the frozen judge. (This is a classic way this kind
  of setup silently fails, so we check it on startup.)
- **Why our codec is deliberately simple.** We tried a fancier design and it
  hurt the comparison, because a fancier codec quietly keeps meaning even
  without our rule, which muddies the very thing we're trying to show. We went
  back to the simple one on purpose.
- **The CLIP gotcha.** The dataset's built-in CLIP features come from a
  specific version of CLIP. Use the wrong version at search time and retrieval
  silently turns to garbage. We verified this: the right version matches at
  0.98 similarity, the wrong one at basically zero. Small detail, total
  difference.
- **Everything is reproducible** from the training script, and the numbers
  here are read out of the JSON files listed at the top.

---

# An honest note on this project at this hackathon

We know this is a brain-science project at a hackathon centered on cancer and
proteomics. Our method — "compress while keeping whatever a task cares about,
and let that same judge become a search engine" — works on many kinds of data,
not just brains. Brains are simply where we proved it. We didn't build a
cancer-data version this weekend. But this is the proof-of-concept, the brain
was the cleanest place to show it, and we'd love to point it at biology that
matters to you next.

Type a word. Find a thought.
