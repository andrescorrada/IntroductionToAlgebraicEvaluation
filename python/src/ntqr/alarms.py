"""Algorithms for logical alarms based on the axioms.

Formal verification of unsupervised evaluations is carried out by
using the agreements and disagreements between classifiers to detect
if they are misaligned given a safety specification.

The 'atomic' logical test for the alarms is a look at the group
evaluations that are logically consistent with how the classifiers
aligned in their decisions and an assumed number of corrects for
each label in the true, but unknown, answer key for the exam.

For example, in a test with three possible responses or classes
for each question, we need to specify,

    qs = (q_label_1, q_label_2, q_label_3)

where

    sum(qs) = Q,

with Q the size of the test. So a test with Q=10, could have a qs setting of, 
(5,3,2) since sum(5,3,2) = 10.

This atomic misalignment test at fixed qs value then allows you
to create custom alarms depending on your application domain.
Some examples,

1. The prevalence of classes in your tests is biased toward
small amounts of one label or the other. In that case, you can construct
an alarm as,

        all([alarm.misaligned_at_qs(qs, responses) for qs my_range()])

2. The method `ntqr.SingleClassifierAxiomsAlarm.are_misaligned`
is a test for fully unsupervised settings and is equivalent to,

        all([alarm.misaligned_at_qs((qa,Q-qa), rs) for qa in range(0,Q+1)])

That is, the only thing you have are the classifiers' responses and the
size of the test, Q.

3. You believe that your classifiers are high performing and therefore
will only accept (Q_label_1, Q_label_2, ...) settings for which
all your classifiers are better than x% at detecting all the labels.
This turns the atomic logical test into a measuring instrument for
the prevalence of the labels in the tested dataset. The method

    SingleClassifierAxiomsAlarm.are_misaligned( responses )

is the fully unsupervised version of what logical alarms can do.
It detects (imperfectly!) if at least one member in an ensemble is
violating a user provided safety specification when doing classification
with R classes.

The name 'are_misaligned' should make clear that this detects when
classifiers are misaligned **and** this is not the same thing as being
correct. If a pair of classifiers are being tested, if both are wrong
in the same way, `.are_misaligned` will return False.

The user is encouraged to think of these alarms as building blocks for
algorithms that use the philosophy of error-detecting codes. For example,
by having three classifiers, as long as one of them is behaving correctly,
`.are_misaligned` will return True.
"""

import itertools


class SingleClassifierAxiomsAlarm:
    """Alarm based on the single classifier axioms for the ensemble members

    Initialize with 'classifiers_axioms', list of
    ntqr.r3.raxioms.SingleClassifierAxioms, one per classifier we want
    to compare.

    Also needs class that does SingleClassifierEvaluations given the
    axioms as 'cls_single_evals'
    """

    def __init__(self, Q, classifiers_axioms, cls_single_evals):
        """"""

        self.Q = Q

        self.classifiers_axioms = classifiers_axioms
        self.labels = classifiers_axioms[0].labels

        self.evals = [
            cls_single_evals(self.Q, axioms)
            for axioms in self.classifiers_axioms
        ]

    def set_safety_specification(self, factors):
        """Set alarm's safetySpecification given factors"""
        assert len(factors) == len(self.labels)
        self.safety_specification = LabelsSafetySpecification(factors)

    def misaligned_at_qs(self, qs, responses):
        """Boolean test to see if the classifiers violate the safety
        specification at given questions correct numbers, qs.
        """

        assert self.check_responses(qs, responses)

        max_corrects = [
            sca_eval.max_correct_at_qs(qs, cresponses)
            for sca_eval, cresponses in zip(self.evals, responses)
        ]

        return any(
            (not self.safety_specification.is_satisfied(qs, max_correct))
            for max_correct in max_corrects
        )

    def misalignment_trace(self, responses):
        """Boolean trace of the classifiers test alignment over all
        possible qs values for a test of size Q.

        responses: Iterator over classifier ordered label responses.
        Generically of the form (num_label_1_responses, num_label_2_res...)
        """
        all_qs = self.evals[0].all_qs()
        trace = set(
            (qs, self.misaligned_at_qs(qs, responses)) for qs in all_qs
        )
        return trace

    def are_misaligned(self, responses):
        "Boolean AND of the misalignment trace given responses."
        trace = self.misalignment_trace(responses)
        return all(list(zip(*trace))[1])

    def check_responses(self, qs, responses):
        """Checks logical constraints on responses.
        1. The sum of label correct questions equals the size of the test.

            sum(qs) = Q

        2. All classifiers label responses also sum to the test size.

            all( (sum(classifer_rsps) == Q) for classifier_rsps in responses)

        """
        q = sum(qs)
        if q == self.Q:
            qs_check = True
        else:
            qs_check = False
        responses_check = all(
            [
                (q == sum(classifier_responses))
                for classifier_responses in responses
            ]
        )

        consistency_check = all([qs_check, responses_check])
        if consistency_check == False:
            print("q ", q, "Q ", self.Q),
            print("qs ", qs)
            print("responses ", responses)
            print("checks ", [qs_check, responses_check])

        return all([qs_check, responses_check])


class LabelsSafetySpecification:
    """Simple example of safety specification for each label."""

    def __init__(self, factors):
        """


        Parameters
        ----------
        factors : List[int]
            List of factors to be used in safety specification tests,
            one per label.

        Returns
        -------
        None.

        """

        self.factors = factors

    def is_satisfied(self, qs, correct_responses):
        """Checks that list of label accuracies satisfy the
        safety specification.

        Returns True if all labels satisfy the equation
          factor*correct_response - ql > 0
        given each label's factor and assumed number in
        the unknown test answer key, False otherwise.
        """
        tests = [
            (factor * correct_response - ql) > 0 if ql > 0 else True
            for factor, correct_response, ql in zip(
                self.factors, correct_responses, qs
            )
        ]
        return all(tests)

    def pair_safe_evaluations_at_qs(self, qs):
        """All pair evaluations satisfying safety spec at given qs."""
        correct_ranges = [
            [
                q_correct
                for q_correct in range(0, ql + 1)
                if ((factor * q_correct) - ql > 0)
            ]
            for factor, ql in zip(self.factors, qs)
        ]

        return [
            itertools.product(correct_range, repeat=2)
            for correct_range in correct_ranges
        ]


class GradeSafetySpecification:
    """Simple example of a grade safety specification."""

    def __init__(self, factors):
        """


        Parameters
        ----------
        factors: list of int
            Label accuracy is required to be better than Q_label/factor.

        Returns
        -------
        None.

        """

        self.factors = factors

    def is_satisfied(self, qs, correct_responses):
        """Checks that list of label accuracies satisfy the
        safety specification.

        Parameters
        ----------
        qs : list of int
            Number of label questions in the test.
        correct_responses : iterable of list of ints
            Number of label correct responses, one per label.

        Returns
        -------
        True if the max correct answers for labels satisfy
        factor*sum(correct_label) - Q > 0 given questions
        numbers qs, False otherwise.
        """
        Q = sum(qs)
        grade_test = self.factor * sum(correct_responses) - Q > 0
        return grade_test

    def pair_safe_evaluations_at_qs(self, qs):
        """All pair evaluations satisfying safety spec at given qs."""
        ql_ranges = (
            [q_correct for q_correct in range(0, ql + 1)] for ql in qs
        )
        pair_safe_evaluations = filter(
            lambda rs: self.is_satisfied(qs, rs), itertools.product(*ql_ranges)
        )
        return [
            itertools.product(correct_range, repeat=2)
            for correct_range in zip(*pair_safe_evaluations)
        ]
