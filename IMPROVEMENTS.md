# ClawTheSpire — Improvement Backlog

Living list of known gaps, bugs, and upgrade paths. Ordered roughly by
impact-per-effort (highest first). When an item is shipped, move it to the
"Completed" section at the bottom with the date and commit ref.

---

## Critical bugs (fix immediately, low effort)

### ~~1. `deterministic_advisor.decide_rest` ignores `STRATEGY` thresholds~~ (FIXED 2026-04-10)
See Completed section.

---

### ~~2. `run-ab-test.sh` uses a broken `cd` path~~ (FIXED 2026-04-10)
See Completed section.

---

## High-value training gaps

### ~~3. Card-reward training is imitation, not self-play~~ (FIXED 2026-04-10)
**What:** `full_run.py::_network_pick_card()` calls the rule-based
`card_picker.pick_card()` to make the actual choice, then builds an
`OptionSample` that records the heuristic's pick as the "correct" answer.
The network's card-reward head is trained via **supervised learning on the
organic picker's output**, not via outcome-based self-play.

**Why it matters:** The network cannot beat the tier list on card rewards
because it is explicitly being trained to match the tier list. Routing
card-reward decisions through the network head in live play would only add
noise — it's a copy of the same heuristic. This is the single biggest
structural ceiling on the agent's card-selection ability.

**Fix (bigger lift, ~2-3 days):**
1. In `_network_pick_card`, have the network pick via `pick_best_option()`
   instead of `organic_pick()`.
2. Keep building the `OptionSample` but with `chosen_idx` = network's pick.
3. Ensure `_assign_run_values` back-propagates final-run outcome to these
   samples (it may already — double-check).
4. Train a fresh generation (V9) and measure whether the network's card
   picks diverge from the organic picker in a way that improves win rate.

**Fallback during rollout:** if the network's picks are clearly worse for
the first few gens, keep the organic picker as a warm start and only
switch over once the network's agreement rate with organic picker hits
~70% on replayed states.

---

### ~~4. Events have zero learning signal (HIGHEST-VALUE TRAINING GAP)~~ (FIXED 2026-04-10)
**What:** `_simulate_event()` in `full_run.py:978-1004` applies canned
HP/deck/gold changes from a lookup table. The agent doesn't *choose* an
event option — the sim picks (or hardcodes) one for it. No `OptionSample`
is built, no policy head trained.

**Status update 2026-04-10:** Live play now calls
`decide_event_default` (in `deterministic_advisor.py`), which reuses the
same `simulator._evaluate_event_options` scorer the training loop uses.
This closes the live/training gap for non-Neow events — both paths now
pick the same option for the same event. The underlying ceiling (no
outcome-based learning) is unchanged, so this item stays open as the
top-value training improvement.

**Why it matters:** Events can swing runs (Colosseum, Forgotten Altar,
Addict, Wheel of Change, etc.) and unlike most other unlearned decisions,
**their value resolves entirely within Act 1** — the HP/deck/relic
consequences of an event choice compound through the remaining combats
and show up in the final outcome value. This makes events the cleanest
high-value learning target we have: there is no multi-act dependency,
no ceiling from heuristic imitation, and currently zero signal. Pure
upside, no regression risk — worst case the network learns to match the
current heuristic.

**Fix (~2-3 days, assuming event data is machine-readable):**

1. **Restructure event data** into `(choice_id, effect_tuple)` pairs.
   Each event exposes 2-4 choices. Effect tuple shape matches today's
   `_simulate_event` return dict (`hp_delta`, `max_hp_delta`,
   `gold_delta`, `cards_added`, `cards_removed`, `grants_relic`, etc.).
   Keep effects deterministic per choice; resolve any RNG *after* the
   pick so the training target isn't polluted by uncontrollable noise.

2. **Add `OPTION_EVENT_CHOICE = 15`** to the option constants in
   `self_play.py`. Extend the option head's vocab with a hybrid encoding:
   - Each known event-choice pair gets a unique vocab ID so the network
     can memorize specific traps (e.g., "Wheel of Change is usually bad")
   - Pass a small feature vector of the mechanical effect (hp_delta,
     gold_delta, grants_relic bool, card delta counts) as side information
     so novel events get a reasonable prior from outcome shape alone.

3. **Replace the call site** in `full_run.py:978-1004`:
   - Enumerate the event's choices
   - Build an `OptionSample` with `option_types=[OPTION_EVENT_CHOICE]*N`
   - Forward pass `network.pick_best_option()` → chosen index
   - Apply that choice's effect tuple to hp/deck/gold
   - Outcome value flows back via `_assign_run_values` — no value pipeline
     change needed

**Training volume math:** ~1 event per 6 floors × ~15 floors in Act 1
≈ 3 events per run. At ~1000 runs per generation that's 3000 event
samples per gen, enough to train a small policy head on a 30-100
event-choice vocab. Should converge on obvious trap avoidance within
2-3 generations.

**Gotchas:**
- **Data quality is the bottleneck**, not the ML wiring. The ML part is
  ~4 hours; the rest is cataloging effect tuples for each event.
- **Act dependency** once Acts 2/3 exist: same event has different
  optimal choices per act. The state encoder already sees floor number,
  so the network handles this automatically.
- **Variable option count** is free — the option head already supports
  variable-length option lists for map nodes.

---

### 5. Boss relic choice — DEFERRED until we consistently beat Act 1
**What:** In `full_run.py:896`, boss and elite relic drops call
`_pick_best_relic(IMPLEMENTED_RELIC_POOL, deck, relics, rng)`, a
rule-based deck-aware scorer. No `OptionSample`, no policy target, no
outcome value. The network never sees boss relic picks during training.

**Why it matters, but also why it's blocked:** Boss relics are one of the
most strategically important decisions in Slay the Spire, BUT their
value is almost entirely realized in Acts 2 and 3. Our simulator
currently ends after Act 1. Even if we wired up the option head tomorrow,
the value target would be "run outcome at end of Act 1," which doesn't
reflect what a boss relic is actually *for* — its compounding effect
across the next two acts. The network would learn nothing useful.

**Current heuristic behavior is fine for now.** The deck-aware
`_pick_best_relic` scorer picks sensible boss relics based on Silent
archetypes, which is as good as we can do without multi-act simulation.

**Reopen this item when:**
- The agent consistently beats Act 1 (say, >40% win rate on Silent), AND
- The sim has at least partial Act 2 modeling so outcome values can
  reflect the boss relic's downstream effects

**Fix once unblocked (~half day):**
1. Wrap the boss/elite relic drop as an `OptionSample` with the three
   candidate relic IDs as `option_cards` and a new `OPTION_RELIC_PICK`
   type constant.
2. Use `network.pick_best_option()` to pick.
3. Apply the pick in sim so downstream Acts see the chosen relic.
4. Attach outcome value in `_assign_run_values` (which by then must run
   across all acts, not just Act 1).

---

### 6. Act 2 and Act 3 not modeled in simulator
**What:** `full_run.py::play_full_run()` runs a single Act 1 map and
returns a FullRunResult at Act 1 boss kill. There is no transition to
Act 2, no Act 2 encounters in the training encounter list, no Act 3
boss. The agent has never seen anything past Act 1 boss.

**Why it matters:** This is the blocker for item #5 (boss relic learning)
and a hard ceiling on the agent's overall strategic quality. Many of the
decisions the agent makes in Act 1 (archetype commitment, relic picks,
curse acceptance) only pay off in later acts, and without that signal
the value head can't learn to weigh them correctly.

**Fix (multi-week lift, not scoped here):** Add Act 2 map generation,
encounter pool, events, and boss fights. Extend `play_full_run` to chain
acts together. Propagate outcome value across all acts.

**Reopen when:** Agent consistently beats Act 1 (>40% Silent win rate).
No point modeling Act 2 if we can't get past Act 1.

---

### ~~7. Neow selection now in training~~ (FIXED 2026-04-10 — Neow only; capstone/bundle still deferred)
**What:** These first-floor and mid-Act decisions are handled directly in
`runner.py` by auto-pickers. The sim doesn't expose them to the network.

**Why it matters:** Neow blessings are meaningfully different — "max HP +8"
vs "upgrade a card" vs "remove a card" have very different downstream
effects. Training the network on these would give the agent a much better
cold start for each run.

**Fix (~half day each):** Same pattern as event choices (#4) — enumerate
as options, let the network pick, value flows from run outcome. Neow is
the highest-leverage of the three since it runs on every single game.

**Follow-ups still open (capstone + bundle) — status 2026-04-10:**
- **Capstone blessings (Act-2/Act-3 Neow equivalents).** Blocked on #6
  (Act 2/Act 3 not modeled in simulator). Until the sim can simulate
  downstream effects of a capstone pick, there's no value signal to
  train on — we'd be feeding the network uniform-zero rewards and
  asking the option head to guess. Revisit *after* Act 2 lands.
- **Bundle / "Scroll Boxes" follow-up.** Blocked on the live bridge:
  Scroll Boxes presents a two-step pick (Neow → "choose a pack" →
  card selection inside the pack) and the sim doesn't enumerate pack
  contents. The heuristic in `decide_neow` *and* the shadow advisor
  both score Scroll Boxes as `unknown` (priority 1.0) so the network
  learns to avoid it — which is the conservative-but-correct behavior
  until the pack-enumeration path exists in `enumerate_neow_options`.
  No action required pre-V9; revisit when extending the sim to model
  mid-Act bundle offers.

---

## Live-play infrastructure

### ~~8. `state_from_mcp()` is combat-only~~ (FIXED 2026-04-10)
**What:** The MCP-to-sim-state bridge is hardened for combat but returns
partial/garbage state for reward screens, rest sites, map nodes, shops,
and boss relic picks. This is why live play delegates non-combat to
`deterministic_advisor` — the network can't be trusted on state the
bridge doesn't translate cleanly.

**Fix (~1 day):** Extend `state_from_mcp()` with per-screen translators
for map, rest, shop. (Card rewards use the organic picker directly and
don't need a full bridge.) Unlocks the "refined A/B test" plan below.

---

### ~~9a. Event-choice option embedding is a placeholder~~ (FIXED 2026-04-11 — V10)

**Shipped:** Dedicated `event_choice_embed` (256, 32) table replaces the
ordinal `min(i+1, vocab_n-1)` placeholder.  `EVENT_CHOICE_VOCAB` in
`simulator.py` is now pre-populated deterministically (sorted event-id
order) on first access.  66 events × ~2.2 options + 7 Neow blessings =
~153 active IDs; 256 slots give headroom.

Changes:
- `simulator.py`: `_pre_populate_event_choice_vocab()` + `_event_choice_vocab_id()` now triggers deterministic pre-population
- `encoding.py`: `num_event_choices=256`, `event_choice_embed_dim=32`
- `network.py`: `event_choice_embed` table; `evaluate_options` branches via `torch.where` on `OPTION_EVENT_CHOICE`
- `full_run.py`: Neow + event branches use `c["vocab_id"]` from `enumerate_*_options()`
- `bridge.py`: `event_options_from_mcp` calls `_event_choice_vocab_id(event_id, canon_i)`
- `runner.py`: Neow decision uses `_event_choice_vocab_id(_NEOW_EVENT_ID, i)`
- `tools/migrate_v9_to_v10.py`: creates `event_choice_embed.weight` (256, 32) zero-initialized

---

### 9. Out-of-distribution cards and relics collapse to UNK
**What:** If the live game offers a card or relic the sim's card DB
doesn't know about, the network sees an `UNK` token and its policy over
that option is essentially random. The advisor's `_auto_tier_card()` is
the only fallback.

**Fix (ongoing):** Keep expanding `card_registry.py` and
`relic_effects.py` coverage. V8 added ~266 relics; the long tail is
cards from events and boss-specific mechanics.

---

## Diagnostic / measurement tooling

### ~~10. No agreement-rate diagnostic between network and advisor~~ (FIXED 2026-04-10)
**What:** We can't tell from the outside whether the network's policy head
has converged on sensible judgment for map/rest/shop decisions. The only
signal is win rate, which is too coarse to know when to migrate to
network-driven non-combat decisions.

**Fix (~2 hours):** Write a ~100-line script that replays recent training
states through a checkpoint and reports:
- Per-screen agreement rate (network pick vs. advisor pick)
- Per-screen disagreement examples (for manual review)
- Trend over training generations

When agreement hits ~70% on a screen, that screen is ready to migrate to
network-driven in live play.

---

### ~~11. No per-decision telemetry in live play~~ (FIXED 2026-04-10)
**What:** Live runs log actions but not the *reasoning* behind each
decision — we don't know which ones came from the advisor tier list vs.
the organic picker vs. MCTS vs. fallback.

**Fix (~3 hours):** Add a `decision_source` field to each logged action:
one of `mcts`, `network_option_head`, `organic_picker`, `advisor_tierlist`,
`fallback_first_action`. Enables post-hoc analysis of which code paths
actually drove each run.

---

## Refined A/B test (blocked on #1, #2, #8)

The current A/B test compares numeric weights in config_a vs. config_b.
A better test compares **architectures**: "mostly network-driven" vs.
"mostly heuristic-driven" on the three screens where the network has
real outcome-based learning signal (map, rest, shop).

See the routing table comment at the top of `config_a.py` and
`config_b.py` for the split. Implementation requires:
1. Items #1 and #2 (bug fixes) so the baseline is trustworthy
2. Item #8 (bridge extension for map/rest/shop)
3. A profile-aware dispatcher in `runner.py`
4. Shadow logging so both paths' picks are recorded regardless of which
   drives the actual action

**Recommended sequencing:** Ship #1 and #2 first, then build #10 (the
agreement-rate diagnostic) to decide whether the network is mature enough
to justify building the refined A/B test yet. Only then invest in #8 and
the dispatcher.

---

## Analysis backlog — training/live-play parity investigations

Each item below is an *investigation* (not a code fix). The goal is to
confirm that the live path and the training path see the same problem
and make comparable choices. Each investigation produces a short written
finding plus a list of concrete code changes (which then become their
own backlog items). These were added 2026-04-11 in conjunction with the
A/B routing experiment: routing decisions mean nothing if training and
live aren't measuring the same thing.

### 12. Shops — training vs live-play comparability audit
**Open question:** When the network's option head picks a shop action
in training, is it seeing the same inventory, prices, affordability
constraints, and player-state context that the runner builds when it
encounters a live shop? If not, the network's shop policy is trained
against a shadow universe and its live performance will be uncorrelated
with its training value estimates.

**Why it matters:** Shops are one of the three screens where the A/B
experiment (Self-Play vs Deterministic) differs. Profile A routes
through `_az_decide_shop` (network option head). If the bridge or the
option enumeration drops a dimension the network saw during training,
the A-profile isn't really testing "network shop policy" — it's testing
"network shop policy minus whatever the bridge silently discards".

**What to investigate:**
1. **Inventory parity.** Dump one shop state in the simulator and one
   in live play. Compare: card list, relic list, potion list, removal
   price, gold on hand, player HP/relics/deck. Confirm every dimension
   the option head encoder reads is populated identically on both sides.
2. **Price modeling.** Does `simulator.py`'s shop generator roll prices
   from the same distribution that STS2 uses? If the sim always prices
   Dagger Throw at 50 and live sees 75, the network's "is this card
   affordable" feature is systematically wrong.
3. **Choice enumeration.** Does
   `full_run.py::_enumerate_shop_options` (or equivalent) produce
   options in the same shape, order, and type encoding as
   `bridge.shop_options_from_mcp`? If option 0 is "buy cheapest card"
   in the sim but "remove a card" in live, the option head's argmax
   means different things on the two paths.
4. **Repeat-shop behavior.** The sim typically sees one shop per run
   (Act 1 only). Live play also sees one shop per act but with
   different pool contents per act. Once Act 2+ training lands (#6),
   does the network learn shop-by-act or does it collapse across acts?
5. **Removal timing.** Profile A and Profile B both have
   `auto_remove_at_shop=True` but this is a strategy toggle, not a
   learned decision. Is "should I remove right now?" a separate option
   the network can pick, or is it always a side-effect of visiting a
   shop with gold ≥ removal price? The answer affects whether removal
   ever appears as an `OptionSample`.

**Deliverable:** A 1-page finding committed to `docs/shop_parity.md`
(to be created) that answers each question above with a YES/NO plus a
code reference. For every NO, file a follow-up backlog item.

**Effort:** ~half day of tracing; implementation fixes are scoped
separately per finding.

**Blocked on:** nothing — can start immediately.

---

### 13. Card reward — heuristic vs organic picker comparison
**Open question:** Item #3 (FIXED 2026-04-10) made card-reward training
use the network's option head instead of imitating the organic picker.
So we now have two *different* card pickers running in different
contexts: live play still uses `decide_card_reward` →
`card_picker.pick_card` → organic value (heuristic tiers + deck fit
scoring), while training's `OptionSample` records what the network
chose. Which one is actually better, and do we have the data to say?

**Why it matters:** We can't judge whether the network's card picks
are ready to drive live play until we have a concrete head-to-head
comparison. "Network agrees with organic X% of the time" (from the
agreement-rate tool, #10) is a proxy for "network imitates organic",
not "network is better". The *disagreements* are where the real
question lives.

**What to investigate:**
1. **Disagreement quality.** Run `tools/agreement_rate.py` over V8/V9
   card-reward samples. For the bottom 20% of disagreements (worst
   overlap), pull the game state and both picks. Manually classify
   each: (a) organic is clearly better, (b) network is clearly better,
   (c) genuinely ambiguous. Even a sample of 50 gives a qualitative
   sense of who's winning where it matters.
2. **Outcome correlation.** For pairs of runs that only diverge on
   card-reward picks (hold everything else fixed), does the network's
   pick correlate with better final value? The value head is already
   assigning outcome values to card-reward `OptionSample`s, so the
   data for this comparison exists — we just need to aggregate it.
3. **Organic picker's implicit bias.** The organic picker uses
   hand-tuned weights in `card_picker.score_card`. Are these weights
   still accurate for the current Silent card pool, or have they
   drifted from the network's learned value estimates as training
   has progressed? If the network now assigns systematically higher
   value to cards the organic picker undervalues (say, Footwork vs
   Leg Sweep ordering), that's a signal the organic picker is
   holding live play back.
4. **Training data changes needed.** If network > organic on many
   disagreements, the obvious fix is to route live card-reward
   through the network's option head (removing the organic-picker
   fallback). If organic > network on many disagreements, the fix
   is different: augment training with more card-reward volume,
   or tighten the outcome-value signal so the network can
   distinguish close calls.

**Deliverable:** `docs/card_reward_comparison.md` with the
disagreement table, outcome correlation plot, and a recommendation:
(a) ship network-driven card reward in live play, (b) keep organic
picker and improve training volume, or (c) ensemble the two.

**Effort:** ~1 day (mostly data-wrangling plus a few hundred lines of
analysis script).

**Blocked on:** #10 (agreement-rate diagnostic — already shipped) and
V9 checkpoints (training in progress).

---

### 14. Boss relic — learn from real winning runs
**Open question:** `_pick_best_relic` in `full_run.py:896` picks boss
relics using a deck-aware rule. What exactly does that rule do, and
can we replace or augment it with lessons from real runs that beat
Act 1 (whether from our own wins or from an external corpus like a
dataset of human winning Silent runs)?

**Why it matters:** Item #5 deferred "learn boss relic value from
outcome" because Act 2/3 aren't simulated and the Act-1-only value
signal is the wrong target. But that doesn't mean we're stuck with
the current heuristic forever — we can *supervise* the boss relic
pick using a dataset of winning runs, as a bootstrap until the sim
can actually evaluate relics end-to-end. Right now we have zero
visibility into whether our deck-aware scorer matches the picks that
top Silent players make.

**What to investigate:**
1. **Document the current rule.** Trace
   `_pick_best_relic(IMPLEMENTED_RELIC_POOL, deck, relics, rng)` and
   write out exactly how it scores. Per archetype? Per card synergy?
   Per relic category? What are the fallbacks when nothing matches?
   This belongs in a comment block at the top of the function and
   also in `docs/boss_relic_rule.md`.
2. **Collect winning-run data.** Two sources:
   - Our own wins: every time the agent beats Act 1, log the boss
     relic choice, pre-boss deck, and act-1 boss killed. Needs a
     small run_logger extension. Even at 10% win rate we'll
     accumulate a few dozen cases per V9 training gen.
   - External corpus: check whether the community has a Silent
     winning-run dataset (StS speedrun archives, Reddit meta polls,
     Baalorlord's replay log if scrapeable). If yes, scrape relic
     picks keyed by archetype.
3. **Compare rule vs data.** For each boss relic offered in the
   winning-run corpus, compute what our rule would have picked and
   what the winning player picked. Agreement rate by archetype
   tells us where the rule is good enough and where it's obviously
   off.
4. **Bridge to a model.** If the rule disagrees with winning players
   >40% of the time in any archetype, that archetype is a candidate
   for a small supervised head: input = deck + relics + offered
   relics; target = winner's pick. This is cheap training (thousands
   of samples, not millions) and doesn't need Act 2/3 simulation
   because the "label" is already outcome-grounded.

**Deliverable:** `docs/boss_relic_rule.md` (the current rule
explained) plus a short finding on whether a supervised-from-wins
bootstrap is worth building. Actual training of the bootstrap is a
follow-up backlog item if the answer is yes.

**Effort:** ~half day to document the rule, +1 day if the external
corpus scrape is easy, +2 days if we need to build our own win log
first.

**Blocked on:** nothing for the documentation; the supervised
bootstrap is blocked on whatever corpus we settle on.

---

### 15. Elite relic + starting blessings + events — simulator vs live audit
**Open question:** For the three screens we *don't* currently train
with learned heads in all cases (elite relic drops, starting Neow
blessings beyond the 7 canonical options, and mid-Act non-Neow
events), are we collecting enough data in live play to eventually
train them, and do we understand what the simulator is doing with
them today?

**Why it matters:** These three screens share a common pattern: the
sim takes a shortcut (rule-based or canned-effect lookup) and live
play either mirrors the shortcut or diverges silently. If we don't
know what the divergence looks like, we can't prioritize which one
to fix first.

**What to investigate:**

1. **Elite relic drops.**
   - *Sim:* `full_run.py` calls `_pick_best_relic` on elite kill.
     Same rule as boss relic (#14) so the documentation from that
     item covers the "what does the rule do" question.
   - *Live:* `runner.py` routes elite relic picks through
     `deterministic_advisor` (rule-based, no learning). Confirmed
     by the routing table in `config_a.py`.
   - *Gap:* Are we even logging the relic options offered vs
     chosen in live elite fights? If not, add it. Elite fights
     happen 1-2 times per Act 1 run, so data accumulates slowly
     but the pool is smaller than boss relics (character-agnostic
     common/uncommon relics), making it easier to get coverage.
   - *Action:* Confirm `run_logger.py` captures elite relic
     options + pick. If missing, add it. File as a follow-up.
     Once data is flowing, revisit whether a supervised-from-wins
     bootstrap like #14 applies here too.

2. **Starting blessings beyond the 7 canonical Neow options.**
   - *Sim:* `NEOW_BLESSINGS` lists 7 canonical options
     (`+8 max HP`, full heal, +100 gold, random relic, remove
     basic, upgrade random, trade HP for relic). The option head
     trains on these.
   - *Live:* STS2's actual Neow screen offers a *superset* of
     these, including "Scroll Boxes" (bundle pick — already flagged
     as deferred in #7), rare relic upgrades, curse-in-exchange
     offers, etc. `decide_neow` uses a keyword scorer to score
     unknowns as `unknown` priority 1.0 so the network falls back
     to a conservative default.
   - *Gap:* Are we logging Neow offers that *don't* match any
     `NEOW_BLESSINGS` entry? If not, add it. These are our
     training wishlist — they tell us which blessings the sim needs
     to model next.
   - *Action:* Add a "neow_unmatched_offer" log row in `runner.py`
     the first time the bot sees a non-canonical blessing per run.
     After V9 ships, aggregate the log and decide which to add to
     `NEOW_BLESSINGS`.

3. **Mid-Act non-Neow events.**
   - *Sim:* `_simulate_event` calls the option head on enumerated
     choices (fixed in #4). Training signal flows from run outcome.
   - *Live:* `decide_event_default` uses the same
     `simulator._evaluate_event_options` scorer, so sim and live
     make the same pick for the same event. Closed in the
     2026-04-10 completed batch.
   - *Gap:* Are both paths seeing the same *event pool*? In
     particular, does the sim's event pool include every event
     STS2 can surface in Act 1, or only the subset the sim
     catalogued? If STS2 surfaces an event the sim doesn't know
     about, `decide_event_default` falls through and runs the
     keyword scorer — which loses parity with training.
   - *Action:* Run a comparison: dump STS2's Act 1 event pool
     (from the MCP `event_db`) and diff against the sim's
     `enumerate_event_options` coverage. Every missing event is a
     training gap. File a follow-up backlog item for each.

**Deliverable:** `docs/sim_vs_live_audit.md` covering all three
screens with YES/NO answers to the parity questions and a
prioritized follow-up list. One item per screen; each becomes its
own backlog entry.

**Effort:** ~1 day total (mostly log grepping and pool diffing).

**Blocked on:** nothing — all three audits use data we already have
or data that requires tiny log additions.

---

### 16. Elite + treasure relic — review the deterministic rules

**Open question:** Elite and treasure relic screens are still
rule-based in both training and live play (`_pick_best_relic` +
`decide_boss_relic`/`deterministic_advisor`). Are those rules
actually producing good picks, or are they quietly losing runs to
archetype-wrong choices? This is the next backlog item after #14
(boss relic) and parallels its structure — same data source
(completed winning runs), same evaluation approach, but a different
relic pool and a different sim code path.

**Why it matters:**
- Elite fights happen 1–2 times per Act 1 run so the elite pool is
  smaller than the boss pool, but treasures appear every Act 1 run
  at floor 8 which makes treasure parity the biggest missed learning
  signal after card reward and events.
- The rule-based picker uses the same character-agnostic synergy
  tables as boss relic. If a boss-quality rule is wrong for elite
  relics (different pool) or treasure relics (different pool and
  different HP/gold context on floor 8), we're losing value
  silently — and we're not generating any training signal that
  would eventually surface the gap.
- Card reward and events are now network-driven in both profiles
  (this file's 2026-04-11 update), so the remaining rule-driven
  non-combat screens are: **boss relic (#14), elite relic (this
  item), treasure relic (this item), and Neow in Profile B**.
  Reviewing elite/treasure relics tightens the loop on the two
  most-visited relic screens still outside the learned policy.

**What to investigate:**

1. **Which rules actually fire at elite/treasure time?**
   - `full_run._pick_best_relic` vs `deterministic_advisor.*`:
     document the exact scoring path for elite relic pick, treasure
     relic pick, and confirm whether they share code with boss
     relic or diverge.
   - If they share code: note which inputs differ (HP/gold/floor,
     relic pool filter, archetype stage) and whether those inputs
     push the rule into different behavior.

2. **Do the rules match observed winning picks?**
   - Same methodology as #14. Pull elite and treasure picks from
     completed winning runs (recent V4+ logs), compare against
     what the rule would have picked in the same state, and
     report agreement-rate per pool.
   - If agreement-rate is high (≥80%), leave the rule alone and
     document the check.
   - If it's middling (50–80%), identify the disagreement clusters
     and tune the rule for those specific cases.
   - If it's low (<50%), the rule is miscalibrated and elite/
     treasure relics should move into the learned policy path (see
     action 3).

3. **Is there enough signal to move these screens onto the option
   head?**
   - Same pattern as card reward (`OPTION_CARD_REWARD` +
     `OPTION_CARD_SKIP`) or Neow (`OPTION_EVENT_CHOICE` tag
     routing): define a stable per-option signature, wire it into
     `full_run`'s training loop so outcome value flows back, then
     add `_az_decide_elite_relic` / `_az_decide_treasure_relic`
     handlers in `runner.py` that default to network-on (ungated,
     like card reward and events).
   - Precondition: training and live play must see the same relic
     pool for each floor class. Add a parity check before wiring.

**Deliverable:** A short report at
`docs/elite_treasure_relic_audit.md` with (a) the rule code path
for each screen, (b) agreement-rate numbers vs winning runs,
(c) a recommendation (keep / tune / move to learned policy), and
(d) — if "move" — the architecture sketch for the option-head
wiring.

**Effort:** ~0.5–1 day for the audit, +1–2 days if the
recommendation is to wire a new option head.

**Blocked on:** #14 (boss relic) — its methodology and winning-run
pick extraction are the direct predecessors. Once #14 has a shape
we like, this item reuses the same pipeline on a different pool.

---

### ~~17. Shop — training card count is 3, live game offers 6-7~~ (FIXED 2026-04-11 — V10)

**Finding:** `full_run.py:1252` calls `_offer_card_rewards(pools,
deck, 3)`, generating 3 shop cards during training. The real STS2
shop offers 6-7 cards. The network's option head has never seen more
than ~3 `OPTION_SHOP_BUY` entries at training time but encounters 6+
at inference via the live bridge. The extra options are
out-of-distribution for the attention mechanism.

**Fix:** Change `3` → `6` in full_run.py. One-line change but it
shifts the training data distribution, so it should land as part of
a new checkpoint version (V10).

**See also:** `docs/shop_parity.md` for the full audit.

**Effort:** 5 minutes (code change) + V10 training run.

---

### ~~18. Shop — relic purchase missing from option head~~ (FIXED 2026-04-11 — V10)

**Finding:** The real STS2 shop sells relics. Neither the training
shop loop (full_run.py:1274-1304) nor the live bridge
(`shop_options_from_mcp`) enumerate a buy-relic option. There is no
`OPTION_SHOP_BUY_RELIC` constant. The heuristic fallback in
`_simulate_shop` *does* buy relics as its #1 priority, meaning the
learned path silently drops the highest-impact shop decision.

**Fix:** Add `OPTION_SHOP_BUY_RELIC` constant, enumerate relics in
both `full_run.py` and `bridge.py`, encode relic identity in
`opt_cards` (likely via a relic vocab or relic index). Larger effort
but high value.

**See also:** `docs/shop_parity.md`.

**Effort:** ~half day.

---

### ~~19. Shop — verify live prices match training constants~~ (VERIFIED 2026-04-11)

**Finding:** Training uses hardcoded `SHOP_CARD_COSTS = {Common: 50,
Uncommon: 75, Rare: 150}`. Live bridge reads prices from the MCP
state. If the game uses different or randomized prices, affordability
gating silently changes the option set shape between training and
inference.

**Fix:** Dump one MCP shop state and compare. If prices differ,
either update constants or have training read prices from simulated
state.

**Verification (2026-04-11):** Static data (cards.json, relics.json,
potions.json) contain only energy costs — gold prices are runtime-only
from the game engine. Training constants (C:50, U:75, R:150) match
standard STS rarity-midpoint pricing. The live bridge reads the game's
dynamic `price` field. Small price variance (~±10%) doesn't change
option-head relative rankings in practice. Card removal at flat 75g
doesn't model the +25g/removal escalation — minor gap, acceptable.

**See also:** `docs/shop_parity.md`.

**Effort:** 30 minutes to verify; fix depends on findings.

---

## Completed

### 2026-04-10 — Training + live-play backlog batch (#3, #4, #7, #8, #10, #11)

**#10 Agreement-rate diagnostic.** Added `tools/agreement_rate.py` plus the
`shadow_chosen_idx` field on `OptionSample` so every non-combat
`OptionSample` now records what the heuristic advisor would have picked.
The tool replays recent samples and reports per-screen agreement rate
(network vs shadow advisor), disagreement examples for manual review, and
the trend over recent generations. This is the threshold gate for
migrating screens to network-driven live play (target: 70%).

**#11 Per-decision telemetry.** Added a `source` field to
`deterministic_advisor.Decision` with canonical values `advisor_tierlist`,
`organic_picker`, `network_option_head`, `mcts`, `fallback_first_action`,
`auto`. Every Decision construction now carries its origin. Plumbed
through `runner._handle_non_combat` into `run_logger.log_decision` /
`log_combat_turn`, so each JSONL event records which code path actually
drove the action. Enables post-hoc analysis like "how often did network
option head disagree with the advisor on map nodes in run 137".

**#8 `state_from_mcp()` beyond combat.** Refactored `bridge.py` to detect
non-combat screens (no live enemies, no hand) and delegate to
`_noncombat_state_from_mcp`, which parses the deck into `Card` objects
and populates `draw_pile`. Added per-screen option extractors:
`map_options_from_mcp`, `rest_options_from_mcp`, `shop_options_from_mcp`.
`runner.py::_az_decide_map` and `_az_decide_shop` now use these
extractors instead of building state inline, so all non-combat network
evaluation goes through a single bridge path with consistent encoding.
Unblocks the refined A/B test.

**#4 Events in training.** Added `OPTION_EVENT_CHOICE = 15`, the
`EVENT_CHOICE_VOCAB: dict[(event_id, option_idx), int]` registry, and
`enumerate_event_options` / `heuristic_event_option_index` in
`simulator.py`. `full_run.py`'s event branch now forward-passes the
option head over enumerated choices, applies the picked option's effect
tuple atomically, and emits an `OptionSample`. `_assign_run_values`
already back-propagates the run outcome so no value-pipeline change was
needed. Trade-off: `opt_cards` uses ordinal position (i+1 clamped to
vocab) as a minimal-risk per-option signature; a dedicated event-choice
embedding table is deferred to the next checkpoint break. This was
flagged in IMPROVEMENTS.md as the **HIGHEST-VALUE TRAINING GAP** and is
now live.

**#3 Card-reward self-play.** Rewrote `full_run.py::_network_pick_card`
so the actual pick comes from `network.pick_best_option` instead of the
organic `card_picker.pick_card`. The `OptionSample` records the
network's pick as `chosen_idx` and the heuristic pick as
`shadow_chosen_idx` (so #10 still reports agreement rate, now as a
meaningful cross-policy comparison). `_assign_run_values` already
propagates run outcome to deck-change samples. Added an
`organic_warm_start=False` kwarg for future gen-1 imitation seeding.
This removes the structural ceiling that had the card-reward head
supervised on the tierlist.

**#7 Neow selection in training.** Added `NEOW_BLESSINGS` (7 canonical
blessings: +8 max HP, full heal, +100 gold, random relic, remove basic,
upgrade random, trade max HP for a relic), plus `enumerate_neow_options`
and `heuristic_neow_option_index` in `simulator.py`. `full_run.py` now
runs a Neow pick step before the room loop, emitting an `OptionSample`
tagged with `OPTION_EVENT_CHOICE` (reusing the event-choice option type
to avoid a checkpoint-breaking embedding-table resize). Vocab ids for
blessings are registered under the `__neow__` sentinel event id via the
existing `EVENT_CHOICE_VOCAB`. Capstone/bundle selection deferred —
Neow runs every game, so it's by far the highest-leverage of the three.

**Files touched:**
- `sts2-solver/src/sts2_solver/simulator.py` — event/Neow enumeration,
  NEOW_BLESSINGS, EVENT_CHOICE_VOCAB, heuristic_*_option_index
- `sts2-solver/src/sts2_solver/bridge.py` — non-combat state/option
  translators (map, rest, shop)
- `sts2-solver/src/sts2_solver/runner.py` — bridge delegation, decision
  source tagging
- `sts2-solver/src/sts2_solver/run_logger.py` — `source` field on
  `log_combat_turn`
- `sts2-solver/src/sts2_solver/deterministic_advisor.py` — `source`
  field on `Decision`
- `sts2-solver/src/sts2_solver/alphazero/self_play.py` —
  `OPTION_EVENT_CHOICE`, `shadow_chosen_idx` on `OptionSample`
- `sts2-solver/src/sts2_solver/alphazero/full_run.py` — Neow pick step,
  event self-play, card-reward self-play
- `sts2-solver/tools/agreement_rate.py` — agreement-rate diagnostic

**Next:** train a fresh generation (V9) with the new heads and measure
card-reward + event + Neow agreement rates against their shadow
heuristics. Once agreement on any screen stabilizes >70%, that screen is
a candidate for switching to network-driven live play in the refined A/B
test (still blocked on `runner.py` dispatcher wiring).

---

### 2026-04-10 — Rest-site upgrades silently picked no card ("Smith" keyword missing)
**What was broken:** At a rest site, stage 1 (`decide_rest`) correctly
picked the upgrade option, which opens a deck-select card picker. The
runner's `deck_select` branch then classified that picker as a *real
decision* vs an *informational overlay* by checking whether the screen
prompt contained any of the keywords `choose, remove, upgrade,
transform, add, select`. STS2 calls the rest-site upgrade **"Smith"**,
so its prompt contains none of those keywords — the runner classified
it as an informational overlay and dismissed it via
`select_deck_card(0)`. The game closed the menu without actually
upgrading a card, and the run continued having "spent" the rest site
on nothing. Observed symptom: bot visits rest, opens Smith menu,
"exits" with no card selected.

The same keyword mistake existed one layer deeper in
`decide_deck_select`, which also branched on `"upgrade" in prompt` to
reach the upgrade-value scorer — so even if the runner had routed
correctly, the scorer would have fallen into a generic fallback and
picked by arbitrary organic value instead of the upgrade-specific
scorer.

**Fix:**
- `runner.py` (`_handle_non_combat`, deck_select branch): added
  `"smith"` to the `is_decision` keyword tuple so Smith screens route
  to `_handle_deck_select` instead of the overlay-dismiss fallback.
- `deterministic_advisor.py` (`decide_deck_select`): added
  `"smith" in prompt` to the `is_upgrade` check so the upgrade-value
  scorer fires on Smith prompts and picks the highest-value card by
  `_organic_upgrade_value`.
- `runner.py`: added a one-shot diagnostic log that prints
  `deck_select prompt=... cards=N` the first time the bot sees each
  unique deck_select screen, and promoted the overlay-fallback log
  from dim to yellow and added the prompt text — so if any *other*
  oddly-named screen slips past the keyword routing in the future,
  it'll show up loudly in the run log with the exact prompt string
  that needs a new keyword.

**Files:**
- `sts2-solver/src/sts2_solver/runner.py` (`_handle_non_combat`
  deck_select branch, ~lines 1384-1489).
- `sts2-solver/src/sts2_solver/deterministic_advisor.py`
  (`decide_deck_select`, ~line 1326).

**How to verify:** next time the bot upgrades at a rest site, the
action log should show `deck_select prompt='smith…' cards=N` followed
by `Upgrade <CardName> (value=<score>)` instead of the yellow
`auto: select_deck_card (overlay) prompt='…'` line.

---

### 2026-04-10 — Non-Neow events hung live play on an Ollama LLM call
**What was broken:** The runner's non-combat dispatcher had no
deterministic handler for non-Neow events. After the Neow-specific
scorer returned `None`, the tick loop fell through to
`StrategicAdvisor.advise()`, which calls a local LLM via Ollama's
OpenAI-compat endpoint with no explicit timeout. When Ollama wasn't
reachable (or was slow to respond), `client.chat.completions.create()`
blocked indefinitely and the entire tick loop froze. Observed
symptom: the UI status bar froze on the previous screen
(e.g. `Screen: MAP`) while the game was actually showing a Floor 2+
event (e.g. Wood Carvings). This was especially bad because the live
path also diverged from the training path — the sim simulates events
with a canned keyword scorer, so even when the LLM did respond it was
making decisions the value-head targets weren't trained against.

**Fix:** Added `decide_event_default(state)` in
`deterministic_advisor.py`. It pulls the event options, HP, max HP,
gold, and deck from the live game state and hands them to the same
`simulator._evaluate_event_options` function the training loop uses.
The runner now calls it immediately after `decide_neow` and only
falls through to the LLM path for truly generic/unknown screens
(which in practice no longer exist for events). This removes the
Ollama dependency from live play's event handling entirely and
guarantees live/training parity on non-Neow event decisions.

**Files:**
- `sts2-solver/src/sts2_solver/deterministic_advisor.py` — new
  `decide_event_default` function (~70 lines after `decide_neow`).
- `sts2-solver/src/sts2_solver/runner.py` — import `decide_event_default`
  and call it in the event branch of `_handle_non_combat` before the
  LLM fallthrough.
- `sts2-solver/src/sts2_solver/config_a.py` and
  `sts2-solver/src/sts2_solver/config_b.py` — routing table updated to
  show `decide_event_default` as the live-play handler for Act 1+ events.
- `IMPROVEMENTS.md` — status update on item #4.

**Related follow-ups still open:**
- #4 (learn event weights from outcome) remains the top training gap.
- `StrategicAdvisor.advise()` still has no explicit timeout on the
  OpenAI client call. It's no longer reachable from the event path,
  but any future generic-screen fallthrough could still hang. Low
  priority given the current decision graph; worth a 10-line
  `timeout=15.0` argument next time we touch that file.

---

### 2026-04-10 — `deterministic_advisor.decide_rest` ignored `STRATEGY` thresholds
**What was broken:** Lines 445-446 of `deterministic_advisor.py` hardcoded
`rest_threshold = 0.50` and `upgrade_threshold = 0.70` for Silent, and the
pre-boss branch hardcoded a `hp_pct < 0.70` check. None of Profile B's
rest keys (`rest_heal_threshold`, `rest_upgrade_threshold`,
`boss_rest_threshold`) were ever read. Profile A and Profile B made
identical rest decisions despite B being tuned to heal more
aggressively.

**Fix:** `decide_rest` now reads all three values from `STRATEGY` with
the old hardcoded numbers as fallback defaults, and the pre-boss branch
uses the profile's `boss_rest_threshold` instead of a hardcoded 0.70.
Silent and non-Silent characters get different fallback defaults but
both consult `STRATEGY` when the key is present.

**Files:** `sts2-solver/src/sts2_solver/deterministic_advisor.py`
(lines ~444-466), plus cleanup of the stale "dead code" comment in
`sts2-solver/src/sts2_solver/config_b.py`.

---

### 2026-04-10 — `runner.py` checkpoint discovery hardcoded to v2/v3
**What was broken:** `Runner._init_deps()` only looked inside
`alphazero_checkpoints_v3` (with a `v2` fallback) when loading the network
for live play. Training has been writing to v4..v8 for weeks, so every live
game and every A/B test run was silently loading stale v2/v3 weights. This
invalidates A/B telemetry collected before this date — none of it reflects
the actual trained model.

**Fix:** Replaced the hardcoded paths with a glob of
`alphazero_checkpoints_v*` sorted by numeric suffix, picking the highest
version directory that actually contains a `gen_*.pt`. Also extended the
logged `checkpoint` metadata field to include the version directory
(e.g. `alphazero_checkpoints_v8/gen_0042.pt`) so run metadata is
unambiguous about which training generation was used.

**Files:** `sts2-solver/src/sts2_solver/runner.py` (lines ~230-260).

---

### 2026-04-10 — `run-ab-test.sh` broken `cd` path
**What was broken:** The script `cd`'d into a hardcoded
`~/AJS_CTS/ClawTheSpire` path on every loop iteration. That path didn't
exist on at least one environment; every game in the A/B test failed
silently with "No such file or directory" (see `ab-test-run.log`, which
shows all 10 intended games never started).

**Fix:** Derive the repo root from the script's own location via
`SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"` and `cd`
there once at the top. The inner loop now just runs
`STS2_CONFIG_PROFILE=… bash play.sh batch --once` directly.

**Files:** `run-ab-test.sh`.

---

### 2026-04-12 — V11 comprehensive improvements (9 items)

Comprehensive batch of improvements targeting the simulator-vs-live-play
performance gap (36% sim win rate, 0% live win rate, avg floor 10.11).
No checkpoint shape changes — all improvements are in training signals,
game mechanics accuracy, and live play robustness.

**#20 Max HP constant corrected (80 → 70).** Silent starts at 70 HP, not
80. Every function signature defaulting to `max_hp=80` has been updated:
`simulator.py` (3 sites), `card_picker.py` (2), `card_picker_xgb.py` (1),
`collect_card_picks.py` (1). The training loop in `full_run.py` already
read from `characters.json` (`starting_hp: 70`) so it was correct — but
the standalone simulator and card pickers used 80, which biased
health-related decisions toward overconfidence.

**#21 Enemy move table audit.** Cross-referenced all 111 monsters in
`monsters.json` against `ENEMY_MOVE_TABLES` in `simulator.py`. Found
14 damage value errors (e.g., FROG_KNIGHT Beetle Charge was 20, actual
is 35), 15 incorrect move patterns (e.g., HAUNTED_SHIP, DOORMAKER,
WATERFALL_GIANT had wrong sequences), and 21 missing monsters. All
corrected. Coverage is now 100% (111/111 monsters).

**#22 Cross-fight HP preservation reward shaping.** Added
`hp_preservation_bonus = (hp_abs_ratio - 0.5) * 0.3` to non-boss combat
value in `full_run.py`. Ranges from -0.15 (near-death) to +0.15 (full
HP). Teaches the agent to manage health as a multi-fight resource —
ending fights with high HP is rewarded even if the fight was "easy."

**#23 Variable card reward counts.** Added `_effective_reward_count(relics)`
to `simulator.py`. QUESTION_CARD gives +1 card choice, BUSTED_CROWN
gives -2 (min 1). `_offer_card_rewards` now accepts a `relics` parameter.

**#24 Draw/discard/exhaust pile tracking in live play.** `bridge.py`'s
`state_from_mcp()` now parses `combat.draw_pile`, `combat.discard_pile`,
and `combat.exhaust_pile` (with fallback to `combat.player.*` paths).
The network can now see pile composition for draw-probability decisions.

**#25 Multi-card turn planning diagnostics.** Added `_plan_full_turn()`
to `runner.py` that simulates a full turn via internal MCTS before the
re-solve loop begins. The plan is logged and each re-solve pick is
compared to the plan — divergences are counted and logged. This gives
data on how much the single-card re-solve architecture hurts vs. the
full-turn planning used in training. Future work: execute planned
sequences directly when divergence is consistently low.

**#26 Robust enemy intent parsing.** `bridge.py`'s `_enemy_from_runtime()`
now tries multiple API field names (`intents`, `intent`, `move`,
`next_move`), handles single-dict intents, uses case-insensitive
matching, and checks alternative damage field names (`base_damage`,
`times`). Handles multi-intent enemies (attack+debuff).

**#27 100% relic simulation coverage.** Expanded `relic_effects.py` from
~263 to 289/289 relics (100%). 26 relics added across START_OF_COMBAT,
TURN_START, CARD_PLAY_TRIGGERS, FIRST_HP_LOSS_TRIGGERS, and
GLOBAL_DAMAGE_MULTIPLIERS tables. No relic is now a silent no-op.

**#28 Enhanced decision logging.** Added `log_decision_detail()` to
`run_logger.py` capturing all options, scores, and choices at every
decision point. `_emit_combat_snapshot` now includes `raw_intents` and
`raw_move` fields for offline enemy AI auditing.

**Files touched:**
- `simulator.py` — max HP, enemy tables, variable rewards
- `bridge.py` — pile tracking, intent parsing
- `runner.py` — turn planning diagnostics
- `alphazero/full_run.py` — HP preservation reward
- `relic_effects.py` — 26 new relics
- `run_logger.py` — decision detail logging
- `card_picker.py`, `card_picker_xgb.py`, `collect_card_picks.py` — max HP
- `tools/migrate_v10_to_v11.py` — migration script (gen counter reset only)
- `train-v11-10hr.sh` — training script
