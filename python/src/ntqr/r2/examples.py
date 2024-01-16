"""@author: Andrés Corrada-Emmanuel."""
from ntqr.r2.datasketches import LabelVoteCounts

uciadult_label_counts: LabelVoteCounts = {
    "a": {
        ("a", "a", "a"): 715,
        ("a", "a", "b"): 161,
        ("a", "b", "a"): 2406,
        ("a", "b", "b"): 455,
        ("b", "a", "a"): 290,
        ("b", "a", "b"): 94,
        ("b", "b", "a"): 1335,
        ("b", "b", "b"): 231,
    },
    "b": {
        ("a", "a", "a"): 271,
        ("a", "a", "b"): 469,
        ("a", "b", "a"): 3395,
        ("a", "b", "b"): 7517,
        ("b", "a", "a"): 272,
        ("b", "a", "b"): 399,
        ("b", "b", "a"): 6377,
        ("b", "b", "b"): 12455,
    },
}
