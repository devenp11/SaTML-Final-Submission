# Superseded: log-probability scoring runs

These two files are **not** the outputs behind the Section 7 results. They come from
the first run of the experiment, which scored the judge by
`Pr[YES] / (Pr[YES] + Pr[NO])` from first-token log-probabilities and thresholded at
0.5. That quantity is exactly temperature-invariant, because temperature scaling
raises both logits to the power 1/T and cannot change which is larger, so the
"sampled" arm was not in fact sampled and the comparison was uninformative.

They are retained only so the pitfall recorded in Section 7 can be checked directly:
their `score` field is a continuous probability, whereas every file in the parent
directory carries a discrete 0/1 sampled verdict.

Do not use these to reproduce any reported number.
