# Experiment Report Template

## Question and hypothesis

- Research question:
- Expected debate effect:
- Primary metric:
- Stopping rule and sample size chosen before running:

## Configuration

| Field | Value |
|---|---|
| Dataset, split, fingerprint | |
| Model/provider version | |
| Judge model | |
| Agents × rounds | |
| Temperature / seed | |
| Date | |

## Results

| Strategy | N | Accuracy/F1 | Parse failures | Tokens | Mean latency | Cost |
|---|---:|---:|---:|---:|---:|---:|
| Direct | | | | | | |
| Debate | | | | | | |

Report an uncertainty interval for the paired metric difference on serious runs. Include at least five
qualitative cases: two improvements, two regressions, and one disagreement/tie.

## Threats to validity

- Did majority voting add a self-consistency advantage independent of debate?
- Could the judge favor the generator model family?
- Were parse failures counted, inspected, and reported?
- Were dataset revisions and prompt changes controlled?
- Does added accuracy justify added latency and cost?

## Conclusion

State what the data supports, what it does not support, and the next falsifiable experiment.

