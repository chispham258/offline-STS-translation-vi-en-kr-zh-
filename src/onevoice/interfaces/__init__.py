"""Component contracts. Every implementation satisfies one of these (D6)."""

from onevoice.interfaces.base import StreamingComponent
from onevoice.interfaces.base_asr import BaseASR
from onevoice.interfaces.base_mt import BaseMT
from onevoice.interfaces.base_tts import BaseTTS
from onevoice.interfaces.base_vad import BaseVAD

__all__ = ["BaseASR", "BaseMT", "BaseTTS", "BaseVAD", "StreamingComponent"]
