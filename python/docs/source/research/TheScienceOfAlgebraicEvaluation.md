# The Science and Engineering of Algebraic Evaluation for Safety Applications

The mathematics of algebraic evaluation makes clear that absent the true
labels for your data, all you can prove is logical validity of the evaluation
of an ensemble of noisy oracles. The theorems should be viewed as exploring
the outer limits of what knowledge is achievable when the **only** information
we have for evaluation are their noisy decisions.

For the philosophical readers, the uitility and limitations of algebraic
evaluation can be summarized by saying that it is an algebraic framework
for establishing the logical consistency of evaluations of tests taken
by noisy responders using only statistics of their observed decisions.

Typography can be exploited to illustrate the domain of this logic of
evaluation. Our mise-en-scene is as follows, we have three actors on
the evaluation stage.

- The N noisy, disagreeing oracles making decisions on what the correct
  responses are to a multiple-choice test you have given them.
- The algorithm you are using to evaluate them given the results of the test.
  This could be the Bayesian approaches pioneered by Raykar or the spectral
  ones associated with the work of Parisi and associates. Or it could be,
  as illustrated in the package with two different algebraic evaluators:
  
     **Majority voting:** You use the majority decisions of the ensemble to
     reconstruct the unknown/missing/spoofed answer key. Note that majority
     voting can be restated as a theorem of evaluation that has as its
     assumptions that the minority vote being right never occured during
     the test. The theoretical flaw of majority voting for safe evaluations
     is that it is unable to signal the failure of these assumptions. This
     makes it sub-optimal for safety applications.
  
     **Error Independent Algebraic Evaluator:** You use the exact algebraic
     that is possible when you have error-independent ensembles. This is
     actually an easier condition to satisfy than the one used by majority
     voting and has the enormous safety virtue that it **signals** when its
     error-independence assumption is violated.
  
- The validator of the evaluation. This is the crucial role that must always
  exist to be truly safe. How do we know an evaluation is valid? In a setting
  where we have used unlabeled data to see how the noisy oracles agree and
  disagree? If we want provably correct or incorrect evaluations in
  unsupervised settings, the only way to do that is by having **complete**
  sets of postulates. This immediately eliminates all probabilistic models,
  or any construct based on representations of what the noisy oracles are
  deciding upon. The representation, itself, would be an untestable assumption.
  Algebraic evaluation is the only way to do it. Only sample statistics can
  have complete postulates that will allow us to unequivocally say if an
  evaluation is inconsistent with the observed decisions.

  So provable safety requires that we have a finite chain of validation:
  (validator of evaluation (evaluator(noisy oracles))).

## Degree of validity is a scientific and engineering problem

But there is not universal validity. Not only are you always just computing
sample statistics, you have to consider how to engineer your ensembles to
be near the error-independent point that is going to make them useful for
unsupervised evaluation.

And if you think of algebraic evaluation using ensembles as a form of
error-correcting code (which it is!), then you realize that
just like we have many error-correcting codes depending on the application,
we will need different ensemble construction methods for evaluation.

## Error analysis as an intersection between the science and mathematics

The great theoretical strength of algebraic evaluation is that it has the
complete set of evaluation postulates for arbitrarily correlated classifiers.
This allows you to perform a theoretical jiu-jitsu move as follows.

- We have the exact algebraic solution for error independent binary classifiers
  as algebraic polynomials of their agreement and disagreement statistics.
- But it is almost certain (a probability claim!) that they will not be. How
  do we prepare for this?
- Since we have the **generating set** for the observed voting statistics that
  tells us exactly how the frequency of any voting event is related to their
  average label accuracies and correlations, we can fold that back into the
  independent solution to, in turn,
- Get another algebraic polynomial that gives the estimated label accuracies
  as functions of the true label accuracies **plus** the unknown error
  correlations.

This is very beautiful and computationally clear because of the algebra, but
here is where the math is silent on what that means in an actual deployment.
It is a significant advance that we can derive such a clear expression of what
we estimate and the unknown test error correlations, but each application has
to then measure what the variation in these estimates mean for its safety
framework.

This slippage between the mathematics and safety application is common to all
scientific modelling. The practical utility of algebraic evaluation is that
since it focuses on sample statistics, it can provide a universal framework for
studying this theory/safety coupling.
