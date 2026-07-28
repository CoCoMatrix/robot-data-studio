# Pose and validity contract

Store absolute per-frame EEF/hand pose and Camera0 pose with transform direction, frame IDs, units, Rot6D convention, and independent left/right validity.

- Robot rows retain source-native state/joints and recorded action targets plus absolute observation/action EEF poses. Derive action EEF from the same row's recorded target through authoritative FK; use only explicitly allowed same-row anchor joints. Never use future observation state or apply a fixed lag shift.
- Human rows contain measured absolute hand poses and validity only. Do not fabricate Robot state/action fields. A later fixed-width training adapter may add a missing-joint token and false mask after loading.
- Do not interpolate invalid gaps. Within a chunk, when one hand first becomes invalid at step `j`, mask that hand for `j..end`; keep the other hand independent. A later valid anchor may start a new window.

Construct action views online after loading. Freeze Camera0 at chunk `t0`; interpret `T_rel(k)=inv(T_EE,0)@T_EE,k` in that frozen coordinate contract and serialize the established AHA FINAL_V2 convention exactly: translation `p_target - p_anchor`, rotation `R_target @ R_anchor.T`. Keep the non-commuting rotation equivalence test. Never cross episodes or pre-materialize horizon-dependent relative actions in V3.
