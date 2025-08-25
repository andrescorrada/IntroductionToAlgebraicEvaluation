"""@author: Andrés Corrada-Emmanuel."""

import ntqr.evaluations

# This is the most important test module for the NTQR package.
# The current tests use experimental data for binary classifiers.
# A better test suite should include at least R=3 tests since that
# is when two classifiers can disagree and both be wrong.

# This test sketch comes from experiments following Bin Yu's Chapter
# 13 Jupyter notebook experiments with the online consumer dataset.
vote_counts = {
    ("a", "a", "a"): 31,
    ("a", "a", "b"): 8,
    ("a", "b", "a"): 27,
    ("a", "b", "b"): 52,
    ("b", "a", "a"): 1,
    ("b", "a", "b"): 1,
    ("b", "b", "a"): 8,
    ("b", "b", "b"): 72,
}

qa = 195
mVarieties = ntqr.evaluations.MAxiomsVarieties(
    ("a", "b"), ("i", "j", "k"), vote_counts, (qa, 200 - qa), 2
)
ijVariety = mVarieties.mvariety(
    (
        "i",
        "j",
    )
)
ikVariety = mVarieties.mvariety(
    (
        "i",
        "k",
    )
)
jkVariety = mVarieties.mvariety(
    (
        "j",
        "k",
    )
)
iVariety = mVarieties.mvariety(("i",))
jVariety = mVarieties.mvariety(("j",))
kVariety = mVarieties.mvariety(("k",))
