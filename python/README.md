# Tools for the logic of evaluation using unlabeled data

![NTQR log](./NTQRpt24.png)

```console
~$: pip install ntqr
```

Evaluation of noisy decision makers in unsupervised settings is a fundamental
safety engineering problem. This library contains the algebraic postulates that
govern **any** evaluation/grading of noisy binary classifiers/responders.
"Noisy" means that the decision makers (humans, robots, algorithms, etc.) are
not always correct. Using the counts of how often a group/ensemble of them
agreed and disagreed while responding to a finite test, we can infer their
the average statistics of correctness.

For those philosophically inclined, this is the mathematics of logically
consistent evaluations for noisy decision makers. In unsupervised settings,
there is no "answer key" for tests that use unlabeled data. Therefore,
the scientific question is how far we can progress when all we have is
logical consistency of evaluations. This package shows we can get a lot
done.

## Design features

1.  Uses no probability theory. All the algorithms in this package are
    purely algebraic polynomials of statistics of how the group votes on each
    item/question in the evaluation/test. This makes this algebra **universal**
    and is the justification for its postulaic and logic nature. The postulates
    presented in ntqr.r2.postulates are applicable to **any** domain in which
    **any** group of classifiers have been tested.

2.  It treats all groups of decision makers as black boxes. The only arguments
    to the algorithms are sample statistics of how they voted in agreement and
    disagreement during the test. This is also another reason the algorithms are
    domain independent. Algebraic evaluation has no knowledge of the domain in
    which an evaluation was carried out. This lack of representational knowledge
   is crucial to its universal character.

3. Algebraic evaluation algorithms warn when their assumptions are wrong. This
   is the single most important safety feature of algebraic evaluation. No method
   that uses representation of the domain or probability theory can do this.
   Charles Perrow, the author of "Normal Accidents", said

   > Unfortunately, most warning systems do not warn us that
   > they can no longer warn us.

   This package aleviates that problem. Algebraic evaluation can detect many
   cases where its evaluation assumptions are wrong. This is demonstrated here
   with the apperance of irrational numbers during evaluation using the
   ntqr.r2.ErrorIndependentEvaluation class. By construction, finite tests can
   only have rational numbers as possible evaluations. The appearance of
   irrational numbers as estimates signals that an algebraic evaluator's
   assumptions are wrong.

4. Implements the exact solution for the case of error independent binary
   classifiers. This is a much better estimator than majority voting evaluation
   (also implemented in this package).

>**Warning**
This library is under heavy development and is presently meant only
for research and educational purposes. AI or any safety engineering is
not solvable by any one tool. These tools are meant to be part of a broader
safety monitoring system and are not meant as standalone solutions.
NTQR algorithms are meant to complement, not supplement, other safety tools.

