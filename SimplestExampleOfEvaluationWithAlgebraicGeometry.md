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
on the test set. There are many more statistics beyond this we may care to probe.
Most of them remain unsolved problems in the field. My work is at the very door
of what algebraic evaluation will become. For example, we could be interested
in the error patterns when we look at the decisions of the ensemble over two
items in the test set. Algebraically it does not matter what the selection
function for a pair is. That is what makes algebra so useful for epistemic
uncertainty. Algebra has no brain. No probability distribution that models how
errorful the experiment or observation will be like. Algebra is perfect for
intelligence thermometers.
One can think of these observables as the data sketch of the decisions that
will be used to evaluate the noisy algorithms. Once you fix the decision
events you will collect from the test set, you will be able to define the
unknown statistics that explain your "surface" frequency counts.
2. The immediate consequence of just estimating sample statistics is that these are
finite once you have settled on the decision events. If you are just observing
events related to single items, as we will consider here, then there is a finite
polynomial of the unobserved evaluation statistics that will be equivalent to
the observed decision event.

## Algebraic evaluation of the trivial ensemble (n=1)

The trivial mathematical object exists in many mathematical fields. Algebraic evaluation
is the same. The ensemble consisting of one noisy algorithm is the trivial evaluation
ensemble.

1. Observable frequency of decision events: Our choice will be the simplest. We
collect the number of times the ensemble voted ("a") versus ("b"). Let us denote
that by $f_\alpha$ and $f_\beta.$
2. The number of times that the trivial ensemble voted $\alpha$ is, by definition,
equal to the number of times it was correct (the item being labeled was $\alpha$)
plus the number of times it was incorrect (the item was $\beta$). That can be
written as the exact polynomial[^1],

$$ f_\alpha = P_\alpha  P_{1,\alpha} + P_\beta  (1 - P_{1, \beta}). $$

3. We want to hammer an important point now. The polynomial above is exact and
universal. Any noisy algorithm that you test will satisfy this evaluation
polynomial exactly. No ifs, ands, or buts about it. Evaluation polynomials, because
they just estimate sample statistics, are universal. The lack of any domain
knowledge or information in this equation is precisely its utility. Algebra is
dumb and thus universal. No epistemic ignorance on our part will ever make
this algebraic equation be wrong. There are no wrong evaluation polynomials.
This is not at all what happens in Machine Training Land. Neither I nor anyone
else will ever be able to write the universal model that explains all the
phenomena we see. Being an expert is hard. Evaluation is not. That is the
core of our insight of why algebras of evaluation could be useful. They are
dumb but universal. This is a good property to have when you must evaluate
those smarter than yourself.
4. Since there are two decision event frequencies in this simple case, there is
another exact polynomial we can write for the trivial ensemble,

$$ f_\beta = P_\alpha (1 - P_{1,\alpha}) + P_\beta P_{1, \beta}. $$

5. These two equations, actually three, define the polynomial ideal for our
evaluation task of this single binary classifier. A third equation expressing
the dependency between $P_\alpha$ and $P_\beta$ has been added.

$$\begin{align*}
f_\alpha &= P_\alpha  P_{1,\alpha} + P_\beta  (1 - P_{1, \beta}) \\
f_\beta  &= P_\alpha (1 - P_{1,\alpha}) + P_\beta P_{1, \beta}, \\
1 &= P_\alpha + P_\beta
\end{align*}$$

The equations have been arranged so that the things we can observe without
knowledge of the correct answers are on the left. We can see how many times
a given noisy classifier decides $\alpha$ and $\beta$. We do not need
to know the correct answers to count these sample statistics of the test this
classifier took.
On the right side are all the sample statistics of the test that we want to know.
The test needs to be graded on unlabeled data. To grade this binary classifier we
would want to know some environmental sample statistics that have nothing to do
with the classifier. We are talking about either $P_\alpha$ or $P_\beta$. The
number of questions on the test that have correct answer $\alpha$ exists independent
of the classifier used to perform the evaluation.
The other set of sample statistics we need are the label accuracies of the
classifier, $P_{1,\alpha}$ and $P_{1,\beta}.$ We implore the reader to remember
our notational warning. These "P"s are not distributions. They are variables that
stand for the unknown statistics of the sample we seek.

6. We can simplify the algebraic work by eliminating, say, the $P_\beta$ variable.
This leaves us with the final polynomial set,

$$\begin{align*}
f_\alpha &= P_\alpha  P_{1,\alpha} + (1 - P_\alpha)  (1 - P_{1, \beta}) \\
f_\beta  &= P_\alpha (1 - P_{1,\alpha}) + (1 - P_\alpha) P_{1, \beta}
\end{align*}$$

Algebraic geometry started as the study of the zeros of polynomial systems as
the one above. The difference between surface and form, a common topic in the
sciences, arises here too. As it does in linear algebra too. We have equations
and their solutions. Equations are the algebraic objects. Solutions are points
in the space defined by the variables in the polynomial.

What is the relation between the two? To clarify this we can start by rewriting
the equations so we eliminate the equal sign. We write the polynomials as,

$$\begin{align*}
-f_\alpha + P_\alpha  P_{1,\alpha} + (1 - P_\alpha)  (1 - P_{1, \beta}) \\
-f_\beta  + P_\alpha (1 - P_{1,\alpha}) + (1 - P_\alpha) P_{1, \beta}
\end{align*}$$

We can evaluate these equations for any point in the 3-d space of the variables,

$$(P_\alpha, P_{1,\alpha}, P_{1,\beta})$$

The points in the space where all these polynomials are zero is called the
algebraic variety. By rewriting the equation is this way, we can also see
that any linear combination or multiplication of them would still yield
zero for any point that made the original equations zero.

How varied is this space of possible polynomials? Viewed geometrically we
can restate this as follows - our original polynomials define a geometric
object where all the polynomials evaluate to zero. How many "distinct"
polynomials are always zero on exactly the same geometric object? Given
some finite polynomial set that defines the surface where the polynomials
are zero, how many other polynomias are zero there also?

This is the celebrated Hilbert Basis Theorem - every polynomial ideal can
be finitely generated. One way to find such a basis is to calculate the
Groebner basis. The result for our evaluation of a single binary classifiers
is,

$$ P_\alpha (P_{1,\alpha} - f_\alpha) - (1 - P_\alpha) (P_{1,\beta} - f_\beta).$$

We can conclude that the evaluation variety for a single binary classifier must
be a two-dimensional surface in the three dimensional space of the sample
statistics. This is the two-dimensional surface shown in the figure at the top
of this document.

## Conclusions and next steps

What have we accomplished? What are our next steps in applying algebraic
geometry to the task of evaluating binary classifiers on unlabeled data?

1. We set out to evaluate a single binary classifier on unlabeled data. We
found out that we could not get a point estimate for the performance of
the classifier. The best we could do is to define a 2-d surface in the sample
statistics 3-d space. Algebraic evaluation reduced our uncertainty of the
classifiers performance from a 3-d cube to a 2-d surface.

2. Note that at no point did we need to specify any distribution or make
any assumptions about how the classifier made the decisions or how difficult
the items were in the dataset. Purely algebraic considerations lead us to
the discovery of a 2-d surface in the 3-d unknown sample statistics space
that must contain the true evaluation values.

3. Once we go beyond the trivial ensemble, we must consider the topic of error
independence for ensembles of noisy binary classifiers. How do we define
error independence when we are not using measure theory? How does error
correlation on the sample affect our ability to carry out the evaluation?

4. Handling non-zero error correlation is the most important practical problem
when one considers using algebraic evaluation. The current state of the field
is as follows:
 - Three error independent binary classifiers allow us to obtain an exact
 algebraic solution that consists of just two zero-dimensional surfaces (points)
 in the evaluation space.
 - The Groebner basis for three, arbitrarily correlated classifiers is possible.
 - The combination of the Groebner basis assuming error independence along with
 the arbitrarily correlated basis allows us to do Taylor expansions around
 the error independent model. This allows us to consider engineering evaluation
 ensembles that hover around error independence.

[^1]: Working with sample statistics such as, $P_{1,\alpha}$, the frequency of
times the classifier labeled $\alpha$ items correctly, raises a possible source
of notational confusion for the reader - "'Ps'? I thought you did not use
probability theory in algebraic evaluation." We do not. This is merely a
notational convention. Any "P" variable seen here is a variable for a sample
statistic, not a pointer to some probability distribution. We could use $\phi$
instead of $P$ to denote these sample statistics if that helps you realize that we
are pointing to variables that denote the value of a sample statistic, not an
unknown probability distribution.
