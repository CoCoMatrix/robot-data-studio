from __future__ import annotations

from collections.abc import Iterable

from .models import Disposition, Finding, WindowMask


def disposition_from_findings(
    findings: Iterable[Finding],
    *,
    hard_gates: Iterable[str] = (),
    review_gates: Iterable[str] = (),
) -> Disposition:
    hard_gate_set = set(hard_gates)
    review_gate_set = set(review_gates)
    disposition = Disposition()
    for finding in findings:
        if finding.outcome == "hard_reject":
            disposition.hard_reject = True
            disposition.hard_reasons.append(finding.code)
        elif finding.outcome == "review":
            disposition.review = True
            disposition.review_reasons.append(finding.code)
        elif finding.outcome == "window_mask":
            if finding.start_frame is None or finding.end_frame is None:
                raise ValueError(f"window mask finding {finding.code!r} needs frame bounds")
            disposition.window_masks.append(
                WindowMask(
                    start_frame=finding.start_frame,
                    end_frame=finding.end_frame,
                    reason=finding.code,
                )
            )
        elif finding.outcome == "soft_weight":
            disposition.soft_weight *= finding.weight_multiplier
        if finding.code in hard_gate_set:
            disposition.hard_reject = True
            disposition.hard_reasons.append(finding.code)
        if finding.code in review_gate_set:
            disposition.review = True
            disposition.review_reasons.append(finding.code)
    disposition.hard_reasons = sorted(set(disposition.hard_reasons))
    disposition.review_reasons = sorted(set(disposition.review_reasons))
    return disposition
