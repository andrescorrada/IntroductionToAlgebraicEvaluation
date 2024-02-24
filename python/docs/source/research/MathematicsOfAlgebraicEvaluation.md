# The Mathematics of Algebraic Evaluation

The algorithms in this package were developed using the mathematics of
algebraic geometry. For technical details, take a look at the following
papers -

1. [Algebraic Ground Truth Inference: Non-Parametric Estimation of 
    Sample Errors by AI Algorithms](https://arxiv.org/abs/2006.08312): The
    problem of unsupervised monitoring is not just about safety. It also
    occurs as a principal/agent monitoring paradox that has economic
    consequences. We deploy AI in an industrial process presumably to make
    it better, more efficient, etc. How can we monitor deployed systems to
    make sure they are working at peak efficiency? This paper discusses
    the error independent, exact algebraic model for evaluation as a possible
    way to do this unsupervised QA monitoring of your industrial process.

2. [Independence Tests Without Ground Truth for Noisy 
    Learners](https://arxiv.org/abs/2010.15662): This paper tries to resolve
    the monitoring paradox in unsupervised settings - how do we know the
    assumptions of the **evaluation algorithm**, the grader of the noisy AI
    agents, is itself correct when we use it in the field? The paper gives
    an incomplete resolution based on comparing multiple trios of classifiers.
    This has been superseded by a theorem proving that irrational number
    outputs from the error-independent algebraic evaluator signal the
    classifiers were not error-independent on the evaluation test.

3. [Streaming algorithms for evaluating noisy judges on 
    unlabeled data](https://arxiv.org/abs/2306.01726): This paper was
    submitted to NeurIPS 2023 and rejected. The reviews can be found in
    [OpenReview.net](https://openreview.net/forum?id=8S6ZeKB8tu). The
    reviews led to the realization that the ML community has published
    and established as authoritative the work of Platanios and Mitchell.
    Unfortunately, their work is flawed. The proof of their mistake is
    detailed in the technical appendix to this paper.

4. [The logic of NTQR evaluations of noisy AI agents: 
    Complete postulates and logically consistent error 
    correlations](https://arxiv.org/abs/2312.05392): This paper is the
    additional step that was needed to understand why algebraic evaluation
    is not just an algebra, it is a logic for determing the validity of
    **any** algorithm that evaluates noisy agents.

These papers are stronger in the mathematical side than in the scientific
side. Theorems are theorems and they are either true or false. The reader
can decide by themselves if they are valid. The science side, however,
needs to be developed further. This means showing how these algorithms can
be safely implemented in deployed systems. See the Science of Algebraic
Evaluation page for some of the initial steps being taken to fix this
imbalance.
