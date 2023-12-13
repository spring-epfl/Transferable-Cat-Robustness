"""
Expansion counter.
"""
import attr
from defaultcontext import with_default_context


class CounterLimitExceededError(Exception):
    pass


@with_default_context(use_empty_init=True)
@attr.s(auto_attribs=True)
class ExpansionCounter:
    """Context-local counter of expanded nodes."""

    count: int = 0
    counter_lim: int = None
    debug_freq: int = None

    def increment(self):
        if self.counter_lim is not None and self.count > self.counter_lim:
            raise CounterLimitExceededError(
                "Expansion counter limit {} reached.".format(self.counter_lim)
            )
        if self.debug_freq is not None and self.count % self.debug_freq == 0:
            print("Counter is: {}.".format(self.count))
        self.count += 1
