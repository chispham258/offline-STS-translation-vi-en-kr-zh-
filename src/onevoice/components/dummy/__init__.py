"""Test doubles. Importing this module registers all four (D17 forbids twice)."""

from onevoice.components.dummy.echo_asr import EchoASR
from onevoice.components.dummy.energy_vad import EnergyVAD
from onevoice.components.dummy.identity_mt import IdentityMT
from onevoice.components.dummy.silence_tts import SilenceTTS

__all__ = ["EchoASR", "EnergyVAD", "IdentityMT", "SilenceTTS"]
