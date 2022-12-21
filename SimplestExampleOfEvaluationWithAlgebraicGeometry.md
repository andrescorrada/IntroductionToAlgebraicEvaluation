# The simplest introduction to algebraic geometry for evaluation

The machinery of algebraic geometry is complicated and mostly unfamiliar
to ML experts. This explanation is the simplest way to construct an evaluation
that can then illustrate all the relevant algebraic concepts and how they allow
us to grade noisy judges on unlabeled data.

The work that will be accomplished here can be summarized in this single image,

<p>
<figure>
    <img src="img/evaluation-variety-single-binary-classifier-adult-penn-ml.png"
         alt="The evaluation variety for a single binary classifier tested on
         an UCI Adult dataset."
         height="400">
    <figcaption>
    Figure Caption: <b>The algebraic variety of evaluation for a single binary
    classifier tested on an UCI Adult subset.</b>
    </figcaption>
</figure>
</p>

As the term itself implies, algebraic geometry connects two branches of
mathematics - algebra and geometry. More precisely - the geometry of the zeros
of polynomial ideals. The connections of this mathematics to the evaluation of
noisy judges are as follows:
1. The decisions of an ensemble allow us to collect frequency statistics on
their errorful decisions. The trivial case we will consider here (n=1!) is
already complicated on the possible statistics of the sample we may be
interested in. Let us start simple with the "point statistics" - what is the
frequency of vote patterns when we just see their decisions for a single item
on the test set. There are many more statistics beyond this we may care to probe
most of them remain unsolved problems in the field. My work is at the very door
of what algebraic evaluation will become. For example, we could be interested
in the error patterns when we look at the decisions of the ensemble over two
items in the test set. Algebraically it does not matter what the selection
function for a pair is. That is what makes algebra so useful for epistemic
uncertainty. Algebra has no brain. No probability distribution that models how
errorful the experiment or observation will be like. Algebra is perfect for
intelligence thermometers.
