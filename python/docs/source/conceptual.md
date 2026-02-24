# A high-level, conceptual explanation of NTQR logic

No one knows the ground truth in unsupervised settings. Evaluation of
experts seems impossible when the answer keys to the tests they take
are unknown. All is not lost. We can still ask -- what group grades
are logically consistent with the counts of how experts agreed and
disagreed on a test?

There are $R^N$ ways that N classifiers can agree/disagree between R
labels. A classification test can thus be summarized by the observed
counts of these events. In a test of size $Q$ these counts would sum
to $Q$.

If we knew how these event counts were partitioned across the true
labels, we would have enough information to calculate average correct
and incorrect decisions. For example, we could calculate average label
accuracy for any classifier by marginalizing out the other ones.

These event counts do restrict the possible evaluations for the group.
And when classifiers disagree "enough", this can alert you that one of
them is violating your specified label accuracy.

By talking about counts of events that we label arbitrarily, we have
stripped the test of any semantic information. All we are left with is
the count of their agreement/disagreements on a finite set of
responses. This makes this counting logic universal - it applies to any
classification test.

The answer key is represented by the count of labels in it. This can be
represented as an integer point in an R-dimensional simplex. One nice
property of this representation of the key is that it is complete -- it
is guaranteed to trap any possible answer key for the test.

Because of this completeness, we can trade uncertainty about world models
that experts may have into uncertainty about how to evaluate their decisions.
The answer key simplex is the same for all tests of size Q and R labels.

This is useful but we are trading domain verification for test
verification.  Lgic alone cannot do this validation. Even something so
simple as the size of the test needs to be validated as appropriate in
any given monitoring application.

The NTQR package contains the linear algebra algorithms that allow you
to calculate the logically consistent set. This is a subset of all
possible grades.  These sets can be represented by the equations that
define them. All possible grades obey the simplex and marginalization
equations. The logical set are those that obey the observable equations.
