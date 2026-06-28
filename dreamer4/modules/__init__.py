from dreamer4.modules.base import BaseModule
from dreamer4.modules.bc import BCModule
from dreamer4.modules.bc_dynamics import BCDynamicsModule
from dreamer4.modules.dynamics import DynamicsModule
from dreamer4.modules.rl import RLModule
from dreamer4.modules.tokenizer import TokenizerModule

STAGES = {
    "tokenizer": TokenizerModule,
    "dynamics": DynamicsModule,
    "bc": BCModule,
    "bc_dynamics": BCDynamicsModule,
    "rl": RLModule,
}

__all__ = [
    "BaseModule",
    "BCModule",
    "BCDynamicsModule",
    "DynamicsModule",
    "RLModule",
    "STAGES",
    "TokenizerModule",
]
