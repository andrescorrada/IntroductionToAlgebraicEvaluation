# Logic tools to make your AI safer

![NTQR](./img/NTQRpt24.png)

```console
~$: pip install ntqr
```

:::{figure-md}
![Prevalence estimates](./img/threeLLMsBIGBenchMistakeMultistepArithmetic.png)

**The only evaluations possible for three LLMs (Claude, Mistral, ChatGPT) that
graded a fourth one (PaLM2) doing the multistep-arithmetic test in the BIG-Bench-Mistake
dataset. Using the axiom for the single binary classifier, we can reduce each LLMs possible
evaluations as grader of the PaLM2 to the circumscribed planes inside the space of
all possible evaluations.**
:::


Evaluation of noisy decision makers in unsupervised settings is a fundamental
safety engineering problem. This library contains algorithms that treat this
problem from a logical point of view by considering the axiomatic relationships
that must govern how groups may be performing given how they agree and disagree
in their answers to a common test.

A simple demonstration of the power of logic to clarify possible group evaluations
for noisy experts is given by the example of two of them taking a common test
and disagreeing on at least one answer. We can immediately deduce that it is
impossible for **both** to be 100% correct. Notice the power of this elimination
argument. We do not need to know anything about the test or its correct answers.
By just looking at how they agree and disagree, we can immediately deduce what
group evaluations are impossible. The NTQR package carries this out by formulating
algebraic logical axioms that must be obeyed by all evaluations of a given type
or model.

Using logic as an aid for safety is well known in other industries. Formal
software verification is one requirement for certifying the software that
carries out emergency shutdowns of nuclear plants. Autonomous vehicles are
field tested using linear temporal logic.

But logic seems out of place with how we engineer AI agents nowadays,
mostly as statistical algorithms that depend and rely on a well developed set
of ideas from probability theory. In addition, for some readers it may remind
them of the symbolic AI systems that were developed before probability theory
became the basic tool for building algorithms. These used knowledge bases and
tried to reason about the world and their actions. The logic of unsupervised
evaluation we are talking about is none of those.

It is not a logic about how to make decisions. That is what the symbolic
systems try to do. They do withs using a **world model**. These are hard
define except in simulated worlds. Evaluations are not like that. A binary
test is a binary test in any domain. The logic of unsupervised evaluation
requires **evaluation models**. And these are trivial to specify and construct.


The use of logic to keep us safer is
well known in many engineering contexts. Software that is used to safely shutdown
nuclear plants is certified by using methods from formal software verification
to logically prove they comply with their specified use. The NTQR package brings
the framework of formal verification to unsupervised evaluations.

The current version is building out the axioms and logic for the case of binary
classifiers and responders. Future versions will consider 3 or more classes.
At a high level, this package contains a logic of formally verifying binary
evaluations and evaluators and logical tools that use the axioms of that
logic with additional assumptions.
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

>**Warning**
This library is under heavy development and is presently meant only
for research and educational purposes. AI or any safety engineering is
not solvable by any one tool. These tools are meant to be part of a broader
safety monitoring system and are not meant as standalone solutions.
NTQR algorithms are meant to complement, not supplant, other safety tools.

