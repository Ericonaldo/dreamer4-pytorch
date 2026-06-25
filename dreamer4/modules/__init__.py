from dreamer4.modules.base import BaseModule
from dreamer4.modules.bc import BCModule
from dreamer4.modules.dynamics import DynamicsModule
from dreamer4.modules.policy import PolicyModule
from dreamer4.modules.tokenizer import TokenizerModule

STAGES = {
    "tokenizer": TokenizerModule,
    "dynamics": DynamicsModule,
    "bc": BCModule,
    "policy": PolicyModule,
}

__all__ = [
    "BaseModule",
    "BCModule",
    "DynamicsModule",
    "PolicyModule",
    "STAGES",
    "TokenizerModule",
]
