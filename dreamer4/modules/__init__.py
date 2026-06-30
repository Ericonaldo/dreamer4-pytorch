from dreamer4.modules.base import BaseModule
from dreamer4.modules.bc_dynamics import BCDynamicsModule
from dreamer4.modules.rl import RLModule
from dreamer4.modules.tokenizer import TokenizerModule

STAGES = {
    "tokenizer": TokenizerModule,
    "bc_dynamics": BCDynamicsModule,
    "rl": RLModule,
}

__all__ = [
    "BaseModule",
    "BCDynamicsModule",
    "RLModule",
    "STAGES",
    "TokenizerModule",
]
