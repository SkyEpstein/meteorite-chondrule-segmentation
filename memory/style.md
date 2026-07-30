# Writing style guide

The project's standard for prose. Governed by principle 13 of
[constitution.md](constitution.md).

Applies to every word written in this repository: README, ROADMAP, CHANGELOG,
specs, the constitution, code comments and docstrings, human-readable message and
assert strings, matplotlib figure titles, commit messages, and pull request
descriptions.

## The hard rule: style edits never change facts

A style pass may only change how something reads. It may never change what the
repository asserts. Do not change, round, recompute, soften, strengthen, add, or
delete any of:

- a number, threshold, metric, tolerance, count, date, or version
- a file path, filename, or code symbol
- a claim, retraction, withdrawal, caveat, warning, known issue, or bug note
- a status marker: done, not done, incomplete, in progress, SUPERSEDED,
  RETRACTED, WITHDRAWN, CLAIM WITHDRAWN, DO NOT QUOTE, blocked
- a table row, a table column, or the order of rows
- a link, an attribution, or a date

Never delete a caveat to improve flow. Never make a result read better than the
original text made it read. If a sentence depends on a precise technical
distinction, keep the distinction even when that makes the sentence longer.
Clarity is the goal, not compression.

After any style pass, verify mechanically that the multiset of numeric tokens in
the file is unchanged.

## No em dashes

Never use an em dash. Recast the sentence instead, choosing whichever fits:

- a comma, for a light aside
- a colon, when what follows explains or lists
- a full stop and a new sentence, when both halves stand alone
- parentheses, for a true aside
- a connective word: because, so, since, which, that

Do not substitute a spaced hyphen or a double hyphen. That is the same habit in a
different costume.

## Ranges, minus signs, and other punctuation

- A numeric range in prose reads "0.85 to 0.92", not with an en dash.
- A numeric range inside a table cell, where space is tight, uses a plain hyphen.
- Use an ASCII hyphen for a minus sign, not the Unicode minus.
- No arrows. Write "to", "gives", "then", or "which means".
- No middot separators. Use a comma, or split into list items.

## No emoji

Emoji in headings, tables, or status markers read as machine-generated. Delete
them and let the words carry the status. A completed item reads "(done)", not a
check mark. A warning reads as a sentence, not a warning sign. Keep load-bearing
capitalized labels such as KNOWN ISSUES or RETRACTED in caps.

## Habits to avoid

These are the tells that make writing read as machine-generated:

- **Bold spam.** At most one bolded span per paragraph, and only where the
  emphasis does real work. Genuine warnings and table winners may keep it.
- **Italic spam.** Italics for a genuinely contrasted word, not for texture.
- **Rule-of-three lists** where the third item carries no information.
- **"not X, but Y" as a flourish.** State it directly. Keep the construction only
  where the contrast is the actual technical point, for example "this is
  selection bias, not threshold leakage".
- **Restating the paragraph in its final sentence.** Cut the restatement.
- **Portentous fragments** used for drama, such as "Pure rework." Fold them into
  the sentence before.
- **Filler vocabulary:** delve, leverage, seamless, crucial, pivotal, testament,
  landscape, realm, underscore, comprehensive, meticulous, cutting-edge, unlock,
  elevate, tapestry, "it is worth noting", "it is important to note".
- **Throat-clearing** before the point, and hedging that softens a measured
  result.

## Voice to keep

This project's documents are blunt, specific, and self-critical, and they
disclose their own failures. That is the good part. Short declarative sentences.
Contractions are fine. Do not flatten the writing into corporate neutral, and do
not add hedging to make a negative result sound gentler.

## Acceptance checks

A file passes when all of these hold:

- no em dash, en dash, or Unicode minus
- no arrow or middot
- no emoji
- no spaced hyphen or double hyphen introduced as an em-dash substitute
- the multiset of numeric tokens is unchanged from before the edit
- every status label, retraction, and caveat still present
