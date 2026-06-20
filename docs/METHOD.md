# Method Notes

FAR-SQL separates SQL equivalence judgment into two regions:

1. Formal determined region: if VeriEQL returns `equivalent` or
   `non_equivalent`, FAR-SQL keeps that label as the final decision.
2. Non-determined completion region: if VeriEQL returns `timeout`,
   `unsupported`, `runtime_error`, `conversion_error`, or `unknown`, the case is
   normalized and completed by a local SQL-equivalence model.

The pipeline has three main pieces:

- Verification-yield profiles learned from historical VeriEQL logs.
- Dynamic budget routing for the online VeriEQL call.
- State-aware semantic completion with Self-Consistency@3.

The strict profile sends every non-determined case to model completion. The
paper-reproduction profile keeps an additional timeout-positive-prior heuristic
because it was part of the submitted-result artifacts. The audit script reports
that heuristic separately so it is not confused with LLM completion.

