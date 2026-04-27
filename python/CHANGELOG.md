# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- v0.5 Added generic classes for arbitrary number of labels and classifiers.
       These are ntqr.raxioms.MAxiomsIdeal and
       ntqr.evaluations.MLabelResponseSimplexes

- v0.6 Added ntqr.evaluations classes to handle generic number of classifiers
       and labels. Varieties up to m=2 are now computable.


### Changed

### Removed

## [0.5.1] - 2025-04-25
## [0.6]   - 2025-05-09
## [0.6.1] - 2025-05-15
## [0.6.2] - 2025-05-19

### Added

### Fixed

- Incorrect processing of labels in ntqr.raxioms.MAxiomsIdeal._m_two_ideal.

- v0.6.1 Fixed bugs for the M=1,2 axioms expressed in terms of 'errors'
         variables. Updated the Jupyter notebook documentation accordingly.
- v0.6.2 Fixed bug in the M=2 axioms as implemented in
         ntqr.raxioms.MAxiomsIdeal.

### Changed

### Removed

## [0.7] - 2025-08-25

### Added

- Classes in ntqr.statistics and ntqr.raxioms to support a simpler formulation
  of the software.
    - ntqr.statistics.AnswerKeyVariables takes care of the "q vars", the
      variables that denote the count of each label in the answer key.
    - ntqr.statistics.ResponseVariables simplifies the creation of the
      variables associated with the response counts associated with a
      test count or the label response counts that define a group evaluation.
    - ntqr.raxioms.SimplexAxioms creates the axioms associated with each
      by-label decision event space.
    - ntqr.raxioms.MarginalizationAxioms creates the axioms that enforce
      the correct marginalization of decision events by true label.
    - ntqr.raxioms.ObservableAxioms creates the axioms that tie observed
      decision event counts by the classifers to the same events by true
      label.

### Fixed

- Finally fixed longstanding bug for documentation Jupyter notebooks! The
difficulty in finding it is that the Jupyter notebook has no formatting
problems. The problem arose when using nbsphinx to produce documentation
HTML pages for readthedocs.org. This implied that the problem was in how
nbsphinx was configured. This has lead to many fruitless attempts to fix
it. The problem was in how sympy.init_print was being called in each notebook.
Online documentation now looks much better.

### Changed

- The overworked class ntqr.statistics.MClassifiersVariables was moved
  to use the new, simpler classes ntqr.statistics.AnswerKeyVariables and
  ntqr.statistics.ResponseVariables to simplify its own code.
- Jupyter notebooks are now using ntqr.statistics.{AnswerkKeyVariables,
  ResponseVariables}

### Removed

## [0.7.5] - 2026-05-20

### Added

- new dataclass MVariety to contain the set of group evaluations given up
  to M=m statistics. Mostly important, it allows the __and__ operation of
  equal or lesser order.
- new dataclass MVarietyTupleDict to save space while carrying out exact
  computations.

### Fixed

### Changed

### Removed

## [0.7.6] - 2026-04-27

### Added

### Fixed

- Fixed bug in ntqr.statistics.ResponseVariables.errors returning None.
- Sundry typos in docstrings.

### Changed

- Changed the documentation Jupyter notebooks to reflect simpler formulation
  of the axioms.

### Deprecated

- Started deprecation of classes in ntqr.statistics and ntqr.raxioms that
  used the 'ground-up' axioms that required marginalized variables to be
  computed before full ones.




