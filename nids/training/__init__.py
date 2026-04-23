"""Training loop + SSL loss implementations.

Populated in a follow-up session with:
- ``CLANLoss`` (the CLAN novel contribution)
- Re-implementations of SimCLR / Barlow Twins / BYOL / VICReg / SimSiam /
  ConFlow / SSCL-IDS losses for head-to-head comparison
- A generic ``Trainer`` orchestrating pretrain + fine-tune phases.
"""

__all__: list[str] = []
