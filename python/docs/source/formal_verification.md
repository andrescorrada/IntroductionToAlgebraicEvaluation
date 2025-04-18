# Formal Verification of Unsupervised Evaluations

The NTQR package serves a dual purpose. It is both a logic for unsupervised
evaluations of classifiers and a formalism for constructing unsupervised
evaluators. This page discusses the logic aspect of the package by discussing
the idea of formally verifying unsupervised evaluations.

The ideas of formal verification discussed here borrow heavily from the
field of 
[formal software verification](https://en.wikipedia.org/wiki/Formal_verification)
as well recent calls for guaranteed safe AI systems such as 
[Toward Guaranteed Safe AI](https://arxiv.org/html/2405.06624v2) . But note
that unlike the title "Toward ..." there is nothing unfinished about
formal verification of unsupervised evaluation of binary classifiers,
the topic of this page.

## Tests and their evaluation models

By treating the responses to an evaluation as equivalent to filling out
a multiple choice exam bubble sheet we strip any evaluation of its
semantic interpretations. Grading a bubble sheet response sheet only
requires an answer key telling us what correct label ('a', 'b', 'c', ...)
is attached to question, say 10, on the test.

This semantic stripping of a concrete binary evaluation has the
effect of allowing us to treat two different types of interpretations
of a binary test with the same algebraic logic.

The task of binary classification is well known in the ML/AI community
it is an abstract representation of the task of putting one of two labels
on items in a dataset based on their features. The 'task' is abstract in
the sense that the same tools that we use for generating algorithms that
classifying something as a 'cat' or 'dog' could also be used for any
other binary classification task, say between 'correct' or 'incorrect',
or between 'up' or 'down'.

But the concept of binary classification in the ML/AI community is concrete
in the sense that it assumes that there is a semantic equality between
two possible responses in a bubble sheet. But this is not so for multiple
choice exams. The bubble sheet tells you nothing about what 'a' or 'b'
mean in any given question. Thus, we cannot know if the 'a' response in
any question corresponds to same semantic concept when we see an 'a' response
in another one.

This distinction is crucial if one is to understand that the NTQR package is
not just about evaluating binary classifiers. It is also about evaluating
graders of any other task that is being evaluated. This distinction will
become clearer when we discuss safety specifications and their relation to
these two different interpretations of a binary response test. Generically,
I call them R-class tests.

In addition, an R-class test can have many models. The simplest one is where
we just count the number of label responses for a given classifier. This is
the one being used for the current alarm implementations. But we could look
at an evaluation model for how pairs aligned in answering the questions. For
{math}`R` classes, a pair would have {math}`R^2` possible decision patterns.
In general, we could construct models for any number of classifiers,
{math}`R^N`. NTQR code is starting to support arbitrary number of labels.

The above does not exhaust the number of models possible for a finite test.
We could start considering evaluations by aligning the classifier decisions
across question pairs, and so on.

## Safety specifications for evaluation models

Logic cannot determine the costs and benefits of classifier decisions.
A level of performance that is acceptable in one application, may be
intolerable in another. One could be as severe as requiring unanimity
from all test takers on all questions. Any disagreement would immediately
tell us our safety specification was violated.

In general, we want to be less severe. The class
`ntqr.alarms.LabelsSafetySpecification` allows you to set a minimum level of
performance for each label as a ratio. This corresponds to the case where
we binary bubble sheet corresponds to a classification task. Then it makes
sense to require performance on each label since that can be related to
fixed costs when the safety specification is violated.

But in the case of a multiple choice exam for graders of other noisy AI
algorithms, it does not make sense to require per-label performance. In
that case we would want to specify a total grade and the class to use is
`ntqr.alarms.GradeSafetySpecification`.

## Verifying group evaluations

The verification power of unsupervised evaluations is expressed by the
question - what group evaluations are logically consistent with the observed
agreements and disagreements between the classifiers? This is accomplished
by using the axioms as filters - no group evaluation that violates a given
set of axioms is allowed.

This verification is under construction right now. What helps is that
ensembles are going to satisfy a nested set of axioms. All the members of an
ensemble must obey the axioms for single classifiers. Thus, if we eliminate
single classifer evaluations on the basis of these axioms, they cannot be
made consistent by observing how a classifier agreed or not with another.

Thus verification can be viewed as a geometrical 'slicing' operation. Single
axioms act as planes in response space that create cuboids for each label.
Pair aligned responses can only eliminate corner pairs. Aligned trios will
eliminate cube corners, etc.