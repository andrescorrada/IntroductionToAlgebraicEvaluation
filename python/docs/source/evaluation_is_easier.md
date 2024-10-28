# Evaluation is Easier than Decision

The class `ntqr.r2.evaluators.ErrorIndependentEvaluation` is an example of
a jury **evaluation** theorem. Use the agreements and disagreements between
test takers to evaluate them, not to decide what each question's answer should
be.

Majority voting is know, by Condorcet's celebrated 1785 jury **decision**
theorem to be optimal when you are forced to make a per-item decision. On
the spot, if you will. But holding off on deciding and evaluating for
error independent classifiers is much better as a way to decide on all questions
at once. It allows you to use the prevalence of the labels and their accuracy
to opt for majority or minority voting depending on the decision pattern.
This is not a universal prescription but one that depends on carrying out
evaluation **before** decision.