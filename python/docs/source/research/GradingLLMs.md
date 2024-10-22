# Grading LLMs with NTQR Logic

## Introduction

There are two ways to look at the mathematics used in algebraic evaluation.
The first and most obvious one is that it is about evaluating classifiers.
This is the **semantic** interpretation of variables like {math}`P_a`,
the prevalence of the "a" label in the test, where the "a" label points
to the same object in every test question. If you are testing binary
classifiers, you know that the response from them will mean the same thing
for each item(question) in your dataset(test).

But there is an alternative way to view evaluation that is **semantic free**
in the following sense - you are answering a multiple choice exam and filling
out a bubble sheet where you have to translate the response choices you have
been given to the arbitrary symbols - "a", "b", "c", etc. What connects this
semantic free interpretation to the semantic one is the math. It is identical.

But in the semantic free interpretation, there is no meaning to statistics of
the test like {math}`P_{\text{label}}` outside the test. The number of, say,
"b" correct questions in an exam is an incidental fact about the test. It says
nothing about a concept in the domain that the responders were being tested
on.

The same algebraic evaluation algorithms apply to a multiple choice exam about
geology or philosophy. And it is this interpretation that makes it useful
for evaluating noisy LLMs.

## NTQR logic as a digitizing format for logical consistency

The other mental step one needs to take to understand the utility of NTQR
evaluation logic is that it can be applied widely if you separate the
digitization from the actual task the noisy LLMs may be doing.

Consider a philosophy essay exam. All the questions require essay responses.
The teacher would normally grade the test and perhaps assign a number scale
to each response as a way to arrive at a final exam grade. This is
digitization.

Digitization is what any engineering system ultimately does to fulfill
engineering spec and safety concerns. At some point it is economically
inefficient to worry about too many digits of precision with your instruments.
So if you had the theory of how to grade {math}`R=10` exams, you could
do a pretty good job evaluating the responses of an LLM to philosophy queries.
This is very much in line with demonstrations that LLMs can pass bar
exams, etc.

## Terminating endless monitoring chains with multiple-choice exams

The confusion between the literal/semantic interpretation of binary tests and
their logical nature is rampant in the AI/ML research community. This takes many
forms,
1. Confusing the task with the nature of the agent. If an AI agent or LLM takes a binary test or does binary classification that does not mean the agent **is** or can only be used as a binary classifier. An agent may be evaluated in their performance as a binary classifier/responder. But this does not make the agent **only** a binary classifier.
2. Confusing the task of binary classification with the task of responding to a binary response test. During a binary classification evaluation, we are assured of the semantic equality between responses to different questions. The label 'a' means, say, 'dog' in all questions. But binary response tests also can have their responses mapped to two labels for each questions. But the meaning of answering 'a' in one question need not have semantic relation to also responding 'a' in other questions in the test. In either the literal or logical interpretation of the test, we can still talk about the number of correct responses to 'a' or 'b' questions.

Another confusion by AI/ML researchers is not understanding the nature of unsupervised evaluation and how this affects AI safety. A recent paper 
by Meta and KAUST researchers, [Agent-as-Judge: Evaluate Agents with Agents](https://arxiv.org/abs/2410.10934), has the following dismissal of 
the utility of multiple choice exams -

> And thus agentic systems should be evaluated like a human, with
> rich evaluative feedback which looks at the full thought and action trajectory; evaluating an agentic system
> in the traditional way is like evaluating a student using multiple-choice testing—a comparatively unreliable
> estimator (Park, 2010).

Never mind that multiple-choice exams are extensively used to help us evaluate humans as they learn. They are a workhorse of evaluations.
In addition, it is somewhat absurd to complain about unreliable tests when this whole paper shows that agents are **also** unreliable judges
of other agents! All tests have their flaws and their proper use should reflect that understanding.

But most importantly, the much maligned multiple-choice exams have a very important role to play in AI safety. They can terminate endless
monitoring chains - a fundamental problem in unsupervised evaluation. Who checks that our monitoring systems is working correctly? Who grades
the graders? For that you need to have a formal verification framework. Only multiple-choice exams can terminate endless monitoring chains
since we can formalize its axioms and verification. That is a central purpose of the algorithms in the `ntqr` package.

## How to grade LLMs on binary response tests

This version (0.3) of the Python package can be used to grade {math}`R=2`
exams. The conceptual-algorithm for doing so goes as follows

1. Collect a bunch of binary response questions, they could be on one subject
   or many. It does not matter. Randomly assign the two possible responses
   for each question to one of two labels.
2. Test an ensemble of LLMs on those questions.
3. Apply the algorithms here to evaluate them on the test.

The random assignment of binary labels to each question in the test means
that for a large test the prevalence of correct "a" and "b" questions will
approach each other. Absent a theory about the nature of the test, one
cannot say how this convergence will occur.

## LLMs as detectors of logically correct Chain-of-Thought responses

Recent work by Gladys Tyen, Hassan Mansoor, Victor Cărbune, Peter Chen,
and Tony Mak 
[*LLMs cannot find reasoning errors, but can correct them!*](https://arxiv.org/abs/2311.08516)
is one example where we can see the blurring of the semantic and semantic-free
interpretations. Logically correct is semantic free. But the LLMs are being
trained to detect a consistent label of being logically correct across
different Chain-of-Thought (CoT) prompts.

Most interesting in their work is the creation of a dataset to explore the
accuracy of LLMs in two different tasks - finding the logical errors (the
use-case for which you can use algebraic evaluation) and correcting them.

Algebraic evaluation is not a magical technology. While it could help you
evaluate the accuracy of your LLMs in detecting the logical consistency of
a CoT, it cannot help you diagnose or correct it.
