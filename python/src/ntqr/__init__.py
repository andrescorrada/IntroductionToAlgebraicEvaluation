"""
Evaluate noisy decision makers using logic and algebra.

Classes:

Functions:

Misc variables:

    __version__
    uci_adult_test_example
"""

import shutil
import os
from importlib import resources

__version__ = "0.7.7"


from ntqr.r2.examples import uciadult_label_counts

from ntqr.r2.datasketches import TrioLabelVoteCounts, TrioVoteCounts

from ntqr.r2.evaluators import (
    ErrorIndependentEvaluation,
    MajorityVotingEvaluation,
    SupervisedEvaluation,
)

from ntqr.labels import Label, Labels


def copy_notebooks():
    # 1. Get the path to the notebooks folder inside the package
    # 'ntqr' is your package name, 'notebooks' is the subfolder
    try:
        source_path = resources.files("ntqr").joinpath("notebooks")

        destination = os.path.join(os.getcwd(), "ntqr_notebooks")

        if not os.path.exists(destination):
            # shutil.copytree needs a string path, so we cast the PosixPath
            shutil.copytree(str(source_path), destination)
            print(f"✅ Notebooks copied to: {destination}")
        else:
            print(f"❌ Folder '{destination}' already exists!")

    except Exception as e:
        print(f"Error finding notebooks: {e}")
