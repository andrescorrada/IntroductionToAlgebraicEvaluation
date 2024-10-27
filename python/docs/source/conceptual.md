# A high-level, conceptual explanation of NTQR logic

All the complicated algebraic geometry computations here are meant to
accomplish only one thing - give you a logically consistent framework for
validating **any** algorithm that evaluates classifiers on a test that used
unlabeled data. This logical framework has three properties that make it
useful in AI safety applications:

1. It is **universal**. The algorithms here apply to any domain. There is no
   Out of Distribution (OOD) problem when you use algebraic evaluation because
   it does not use any probability theory. By only using summary statistics of
   how a group of classifiers labeled a test set, we can treat all classifiers,
   whether human or robotic, as black boxes. There are no hyperparameters
   to tune or set in NTQR algorithms. If they were, these algorithms could
   not claim to be universal.

2. It is **complete**. The finite nature of any test given to a group of
   binary classifiers means we can guarantee the existence of a complete
   set of postulates that must be obeyed during any evaluation. Completeness
   is a logical safety shield. It allows us to create theorem provers that
   can unequivocably detect violations of the logical consistency of **any**
   grading algorithm. This is demonstrated here by the error-independent
   evaluator outputting an irrational number for test ratios that can only
   be rationals.

3. It allows you to create **self-alarming** evaluation algorithms.
   Algebraic evaluation algorithms warn when their assumptions are wrong. This
   is the single most important safety feature of algebraic evaluation.
   No method that uses representation of the domain or probability theory
   can do this. Charles Perrow, the author of "Normal Accidents", said

     > Unfortunately, most warning systems do not warn us that
     > they can no longer warn us.

   This package aleviates that problem. Algebraic evaluation can detect many
   cases where its evaluation assumptions are wrong.

Another advantage of the axiomatic formalism is that it naturally incorporates
measures of error correlation between the classifiers. These error correlation
statistics are also sample statistics specific to a given test just like the
individual label accuracies of the classifiers. By observing how pairs of
classifiers agree and disagree we can limit not just their individual
performances but also the possible error correlations between them.

In the case of error independent classifiers, we can solve these axioms for
ensembles of three or more of them to obtain point estimates of their
performance that allow us to improve upon majority voting. Instead of
deciding on each question by going with the majority vote of a mixture
of experts, we can collect their answers, evaluate them using these
**jury evaluation theorems**, and then decide if we want to go with majority
voting or not depending the results of the evaluation.

NTQR operates outside the box of conventional AI practice. This
mathematical and logical formalism is new and will become the foundation of
all future work on evaluation when you use unlabeled data. NTQR logic can
help you make your AI safer. It is the only logical choice when it comes to
AI safety.
