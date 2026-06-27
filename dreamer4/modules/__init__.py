from dreamer4.modules.base import BaseModule
from dreamer4.modules.bc import BCModule
from dreamer4.modules.bc_dynamics import BCDynamicsModule
from dreamer4.modules.dynamics import DynamicsModule
from dreamer4.modules.policy import PolicyModule
from dreamer4.modules.tokenizer import TokenizerModule

STAGES = {
    "tokenizer": TokenizerModule,
    "dynamics": DynamicsModule,
    "bc": BCModule,
    "bc_dynamics": BCDynamicsModule,
    "policy": PolicyModule,
}

__all__ = [
    "BaseModule",
    "BCModule",
    "BCDynamicsModule",
    "DynamicsModule",
    "PolicyModule",
    "STAGES",
    "TokenizerModule",
]
