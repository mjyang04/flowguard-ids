"""SSL loss functions implemented in this project.

Only CLAN is implemented for the current thesis scope. The broader SSL
baseline family is discussed in the literature review and left as future work.
"""

from .clan import CLANLoss, clan_loss

__all__ = ["CLANLoss", "clan_loss"]
