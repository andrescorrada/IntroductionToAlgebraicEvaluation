# A high-level, conceptual explanation of NTQR logic

No one knows the ground truth in unsupervised settings. Nonetheless,
how experts disagree can help us exclude possible evaluations for
them. NTQR approaches this problem as a logical question - What are
the group evaluations that are logically consistent with how we
observe experts disagreeing in their decisions?

For example, if two classifiers disagree in their decisions, they
cannot **both** be 100% correct. This is a purely logical argument of
excluding a possible group evaluation (both are 100% correct) based on
the fact that they disagreed. Their disagreement is not logically
consistent with assigning them a 100% correct evaluation.

This simple example exhibits all the traits of how we can create a
logic of unsupervised evaluation that is universal and useful. It is
universal because it has no semantics about the classification task or
internal knowledge about how the classifiers operate. The two experts
disagreeing in the example above could be human or robots, it does not
matter.

The only input used for the algorithms in NTQR are the observed counts
of how classifiers agreed and disagreed when labeling items.  There
are $R^N$ ways that N classifiers can agree/disagree between R
labels. A classification test can thus be summarized by the observed
counts of these events. In a test of size $Q$ these counts would sum
to $Q$.

By talking about counts of events that we label arbitrarily, we have
stripped the test of any semantic information. All we are left with is
the count of their agreement/disagreements on a finite set of
responses. This makes this counting logic universal - it applies to
any classification test.

If we knew how these event counts were partitioned across the true
labels, we would have enough information to calculate average correct
and incorrect decisions. For example, we could calculate average label
accuracy for any classifier by marginalizing out the other ones.

This logic is much easier to formalize because it is guaranteed to be
complete in any domain.  Consider the unknown answer key to a
classification test. We can represent any such key as a tuple of the
number of times a label appears in it. This maps any possible answer
key to an integer point in an R-dimensional space. The finite integer
points in this space are complete -- they are guaranteed to trap any
possible answer key.

Because of this completeness, we can trade uncertainty about world
models that experts may have into uncertainty about how to evaluate
their decisions.  The answer key simplex is the same for all tests of
size Q and R labels.

This is useful but one must be careful to not ascribe magical powers
to purely logical arguments. We are trading domain verification for
test verification.  Logic alone cannot do this validation. Even
something so simple as the size of the test needs to be validated as
appropriate in any given monitoring application.

The NTQR package contains the linear algebra algorithms that allow you
to calculate the logically consistent set. This is a subset of all
possible grades.  These sets can be represented by the equations that
define them. All possible grades obey the simplex and marginalization
equations. The logical set are those that obey the observable
equations.
