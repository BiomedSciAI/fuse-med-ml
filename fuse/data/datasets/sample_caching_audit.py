from typing import Optional
from time import time
from deepdiff import DeepDiff
from fuse.data import get_sample_id

"""
By auditing the samples, "stale" caches can be found, which is very important to detect.
A stale cache of a sample is a cached sample which contains different information then the same sample as it is being freshly created.
There are several reasons that it can happen, for example, a change in some code dependency in some operation in the sample processing pipeline.
Note - setting a too high audit frequency will slow your training.
audit example usage:
# a minimalistic approach, testing only the first sample. Almost no slow down of entire train session, but not periodic audit so higher chance to miss a stale cached sample.
SampleCachingAudit(audit_first_sample=True,audit_rate=None)
)

#another audit usage example - in this case the first sample will be audited, and also one sample every 20 minutes
SampleCachingAudit(audit_first_sample=True, audit_rate=20, audit_units='minutes')
)
"""


class SampleCachingAudit:
    def __init__(
        self,
        audit_first_sample: bool = True,
        audit_rate: Optional[int] = 30,
        audit_units: str = "minutes",
        **audit_diff_kwargs: Optional[dict],
    ):
        """
        :param audit_rate: how frequently, a sample will be both loaded from cache AND loaded fully without using cache.
        Pass 0 or None to disable.
        The purpose of this is to detect cases in which the cached samples no longer match the sample loading sequence of Ops,
        and a cache reset is required.
        Will be ignored if no cacher is provided.
        :param audit_units: the units in which audit_rate will be used. Supported options are ['minutes', 'samples']
        Will be ignored if no cacher is provided.
        :param **audit_diff_kwargs: optionally, pass custom kwargs to DeepDiff comparison.
        This is useful if, for example, you want small epsilon differences to be ignored.
        In such case, you can provide math_epsilon=1e-9 to avoid throwing exception for small differences

        See their documentation here to learn about additional possible flags: https://zepworks.com/deepdiff/current/
        Important - as a default, we pass ignore_nan_inequality=True, as we think it's the most reasonable comparison strategy suitable for caches.
        You can use ignore_nan_inequality=False if you prefer the original default behavior (which considers NaN to not be equal NaN)

        """

        if "ignore_nan_inequality" not in audit_diff_kwargs:
            audit_diff_kwargs["ignore_nan_inequality"] = True
        _audit_unit_options = ["minutes", "samples", None]
        if audit_units not in _audit_unit_options:
            raise Exception(f"audit_units must be one of {_audit_unit_options}")
        self._audit_rate = audit_rate
        self._audit_first_sample = audit_first_sample
        self._audited_so_far = 0
        if self._audit_rate == 0:
            self._audit_rate = None
        self._audit_units = audit_units
        self._audit_units_passed_since_last_audit = 0.0
        if self._audit_units == "minutes":
            self._prev_time = time()
        self._audit_diff_kwargs = audit_diff_kwargs

    def update(self) -> bool:
        """
        Updates internal state related to the audit features (comparison of a sample loaded from cache with a fully loaded/processed sample)
        returns whether an audit should occur now or not.
        """
        if (self._audit_first_sample) and (self._audited_so_far == 0):
            return True
        if self._audit_rate is not None:
            # progress audit units passed so far
            if self._audit_units == "minutes":
                self._audit_units_passed_since_last_audit += (time() - self._prev_time) / 60.0
                self._prev_time = time()
            elif self._audit_units == "samples":
                self._audit_units_passed_since_last_audit += 1
            else:
                assert False

            # check if we need an audit now
            if self._audit_units_passed_since_last_audit >= self._audit_rate:
                # reset it
                if self._audit_units == "minutes":
                    self._audit_units_passed_since_last_audit %= self._audit_rate
                else:
                    self._audit_units_passed_since_last_audit = 0.0
                return True
        return False

    def audit(self, cached_sample, fresh_sample):
        diff = DeepDiff(cached_sample, fresh_sample, **self._audit_diff_kwargs)
        self._audited_so_far += 1
        if len(diff) > 0:
            raise Exception(
                f"Error! During AUDIT found a mismatch between cached_sample and loaded sample.\n"
                "Please reset your cache.\n"
                "Note - this can happen if a change in your (static) pipeline Ops is not expressed in the calculated hash function.\n"
                "There are several reasons that can cause this, for example, you are calling, from within your op external code.\n"
                "This is perfectly fine to do, just make sure you reset your cache after such change.\n"
                "Gladly, the Audit feature caught this stale cache state! :)\n"
                f"sample id in which this staleness was caught: {get_sample_id(fresh_sample)}\n"
                'NOTE: if small changes between the saved cached and the live-loaded/processed sample are ok for your use case, you can set a tolerance epsilon like this: audit_diff_kwargs={"math_epsilon":1e-9}'
                f"diff = {diff}"
            )
