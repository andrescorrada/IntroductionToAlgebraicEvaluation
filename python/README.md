# Logic tools to make your AI safer

![NTQR](./img/NTQRpt24.png)

```console
~$: pip install ntqr
```

:::{figure-md}
![Prevalence estimates](./img/threeLLMsBIGBenchMistakeMultistepArithmetic.png)

**The only evaluations possible for three LLMs (Claude, Mistral, ChatGPT) that
graded a fourth one (PaLM2) doing the multistep-arithmetic test in the
BIG-Bench-Mistake dataset. Using the axiom for the single binary classifier, we
can reduce each LLMs possible evaluations as grader of the PaLM2 to the
circumscribed planes inside the space of all possible evaluations.** :::

Evaluation of noisy decision makers in unsupervised settings is a fundamental
safety engineering problem. This library contains algorithms that treat this
problem from a logical point of view. What are the evaluations that are
logically consistent with how we observe classifiers disagreeing in their
decisions?

Logical consistency between disagreeing experts provides a simple and universal
way to reduce the set of all possible evaluations on a test of size $Q$ to the
set of logically consistent observations. The "one-bit" demonstration of this
filtering out of possible evaluations is given by the example of two experts
that we have observed to disagree on a test. We can immediately deduce that it
is impossible for **both** to be 100% correct.

Notice the power of this elimination argument. We do not need to know anything
about the test or its correct answers. By just looking at how they agree and
disagree, we can immediately deduce what group evaluations are impossible. The
NTQR package carries this out by formulating algebraic logical axioms that must
be obeyed by all evaluations of classifier ensembles.

An algebra of evaluation logic for classifiers can be constructed by
considering three sets of equations.

1. Simplex equations that express the integer partition of discrete event
   counts for any subset of the classifiers.
1. Marginalization equations for how counts of ensembles marginalize to
   subsets.
1. Observation equations that relate ensemble decision event counts to
   marginalization sums over unknown true labels.

The first two sets define the set of all possible evaluations for any finite
test of size $Q$. This set can be quite large as it is polynomial in $Q$ and
exponential in the number of labels, $R$, and the size of the subset, $m$ :
($Q^{R^(m+1)}$) The last set, then filters out the evaluations not logically
consistent with how we observed the classifiers disagreeing on the test.

This is not a logic about how to make decisions. That is what the symbolic
systems try to do. They do this using a **world model**. These are hard define
except in simulated worlds. Evaluations are not like that. A binary test is a
binary test in any domain. The logic of unsupervised evaluation requires
**evaluation models**, not world models. And these are trivial to specify and
construct.

The upcoming version (0.8) has two major changes.

1. The role of the three sets of equations and their use for constructing the
   possible and consistent sets will become central. These are both linear
   Diophantine systems of equations for which there are many good open-source
   and commercial solvers.
1. Calculating the solutions to the Diophantine system is tractable but
   impractical. The no-knowledge alarms, however, can be formulated as a Linear
   Programming (LP) problem that can carry out computations in less than a
   second. [PuLP](https://coin-or.github.io/pulp/) is going to be used to do
   this.

Brief guide:

1. Formal verification of unsupervised evaluations: The NTQR package is working
   out the logic for verifying unsupervised evaluations - what are the group
   evaluations that are logically consistent with how the test takers agree and
   disagree on multiple-choice exams? The page "Formal verification of
   evaluations" explains this further.
1. A way to stop infinite monitoring chains: Who grades the graders? Montioring
   unsupervised test takers raises the specter of endless graders or monitors.
   By having a logic of unsupervised evaluation, we can stop those infinite
   chains. We can verify that pairs of classifiers are misaligned, for example.
   Take a look at the "Logical Alarms" Jupyter notebook.
1. Jury evaluation theorems: Jury decision theorems - when does the crowd
   decide wisely? - go as far back as Condorcet's 1785 theorem proving that
   majority voting makes the crowd wiser for better than average jury members.
   The NTQR package contains jury evaluation theorems - when does the crowd
   evaluate itself wisely? It turns out it does this better than majority
   voting can decide. This has important consequences for how we shoud design
   safer Ai systems. Check out the "Evaluation is easier than decision" page.

> **Warning** This library is under heavy development and is presently meant
> only for research and educational purposes. AI or any safety engineering is
> not solvable by any one tool. These tools are meant to be part of a broader
> safety monitoring system and are not meant as standalone solutions. NTQR
> algorithms are meant to complement, not supplant, other safety tools.
