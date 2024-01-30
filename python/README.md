# Logic tools to make your AI safer

![NTQR](./NTQRpt24.png)

```console
~$: pip install ntqr
```

Evaluation of noisy decision makers in unsupervised settings is a fundamental
safety engineering problem. This library contains the algebraic postulates that
govern **any** evaluation/grading of noisy binary classifiers/responders.
"Noisy" means that the decision makers (humans, robots, algorithms, etc.) are
not always correct. Using the counts of how often a group/ensemble of them
agreed and disagreed while responding to a finite test, we can infer their
average statistics of correctness.

For a high level, conceptual understanding of what you can do with the
evaluation algorithms and postulates in this package, check out the conceptual
guide. The formalism of NTQR logic is what makes it invaluable for safety
applications. In unsupervised settings, your AI is flying blind when it comes
to assessing itself on unlabeled data. The algorithms in this package allow
you to use a group of classifiers to grade themselves.

All the complicated algebraic geometry computations here are meant to
accomplish only one thing - give you a logically consistent framework for
validating **any** algorithm that evaluates classifiers on a test that used
unlabeled data. This logical framework has three properties that make it
useful in AI safety applications -

1. It is **universal**. The algorithms here apply to any domain. There is no
   Out of Distribution (OOD) problem when you use algebraic evaluation because
   it does not use any probability theory. By only using summary statistics of
   how a group of classifiers labeled a test set, we can treat all classifiers,
   whether human or robotic, as black boxes. There are no hyperparameters
   to tune or set in NTQR algorithms. If they where, these algorithms could
   not claim to be universal.

2. It is **complete**. The finite nature of any test given to a group of
   binary classifiers means we can guarantee the existence of a complete
   set of postulates that must be obeyed during any evaluation. Completeness
   is a logical safety shield. It allows us to create theorem provers that
   can unequivocably detect violations of the logical consistency of **any**
   grading algorithm.

3. Allows you to create **self-alarming** evaluation algorithms.
   Algebraic evaluation algorithms warn when their assumptions are wrong. This
   is the single most important safety feature of algebraic evaluation.
   No method that uses representation of the domain or probability theory
   can do this. Charles Perrow, the author of "Normal Accidents", said

     > Unfortunately, most warning systems do not warn us that
     > they can no longer warn us.

   This package aleviates that problem. Algebraic evaluation can detect many
   cases where its evaluation assumptions are wrong.

>**Warning**
This library is under heavy development and is presently meant only
for research and educational purposes. AI or any safety engineering is
not solvable by any one tool. These tools are meant to be part of a broader
safety monitoring system and are not meant as standalone solutions.
NTQR algorithms are meant to complement, not supplant, other safety tools.

