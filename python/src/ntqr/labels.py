"""@author: Andrés Corrada-Emmanuel."""

from typing import Iterable


class Label(str):
    """
    Label object to guarantee a label is stringifiable.

    """

    def __init__(self, label):
        """


        Parameters
        ----------
        label : str
            Label to appear in variable subscripts.

        Returns
        -------
        None.

        """
        self._label = label

    def __str__(self):
        return self._label


class Labels(tuple):
    """
    Labels used in test question responses. The NTQR package assumes that
    all test questions have the same, fixed, label set. These are a
    tuple-like object so the user can specify the canonical order of
    the labels.

    The number of labels defines the integer R in the acronym NTQR.
    Binary classification: R=2
    Three label classification: R=3, etc.
    """

    def __init__(self, labels: Iterable[str]):
        """


        Parameters
        ----------
        labels : Iterable[str]
           Labels for question responses.

        Returns
        -------
        None.

        """
        self._labels = tuple([Label(label) for label in labels])
