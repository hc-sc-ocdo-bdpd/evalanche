# Statistical uncertainty and model comparisons

Evalanche reports point estimates, uncertainty intervals, and paired model
comparisons. These answer different questions and should be read together.

## Pass-rate intervals

Each model's pass rate includes a 95% Wilson score interval. The Wilson method
is used instead of the simple normal, or Wald, interval because the Wald
interval can behave poorly with small samples and pass rates near zero or one.

The interval describes uncertainty caused by evaluating a finite sample of
cases. A wide interval means the benchmark does not yet estimate the model's
pass rate precisely.

## Paired comparisons

All candidate models must be evaluated on the same case IDs. Evalanche uses
the two-sided exact McNemar test to compare each pair's pass/fail outcomes. The
test focuses on the useful disagreements: cases passed by one model and failed
by the other.

This paired analysis is more appropriate than comparing two separate pass-rate
intervals because the models answered the same cases.

When more than two models are evaluated, Evalanche applies Holm correction to
the full set of pairwise p-values. This controls the chance of declaring at
least one false difference across the set of comparisons.

An observed rank is not automatically a recommendation. Evalanche reports a
clear leader only when it has a higher pass rate and the corrected paired test
distinguishes it from every other evaluated model. Otherwise, the report says
that there is no clear winner yet.

## What these calculations do not cover

The current calculations assume that benchmark cases are representative and
independent. They quantify case-sampling uncertainty only. They do not include:

- variation from repeated candidate-model generations;
- variation from repeated LLM-judge calls;
- uncertainty caused by prompt changes;
- dependence among related cases, such as two languages from one product;
- practical differences in cost, latency, privacy, or availability.

Grouped benchmarks will need cluster-aware analysis before related cases are
treated as independent evidence.

## References

- Brown, Cai, and DasGupta (2001), [Interval Estimation for a Binomial
  Proportion](https://www.acsu.buffalo.edu/~cxma/STA517/Interval%20Estimation%20for%20Binomial%20Proportion-StatSci.pdf).
- Dror et al. (2018), [The Hitchhiker's Guide to Testing Statistical
  Significance in Natural Language
  Processing](https://aclanthology.org/P18-1128/).
- Card et al. (2020), [With Little Power Comes Great
  Responsibility](https://aclanthology.org/2020.emnlp-main.745/).
- Holm (1979), [A Simple Sequentially Rejective Multiple Test
  Procedure](https://www.jstor.org/stable/4615733).