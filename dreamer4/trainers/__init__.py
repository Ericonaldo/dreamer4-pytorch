from dreamer4.trainers.base import BaseModule
from dreamer4.trainers.bc_dynamics import BCDynamicsModule
from dreamer4.trainers.rl import RLModule
from dreamer4.trainers.tokenizer import TokenizerModule

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
