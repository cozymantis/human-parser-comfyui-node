from .HumanParserLIPCustomNode import HumanParserLIPCustomNode
from .HumanParserATRCustomNode import HumanParserATRCustomNode
from .HumanParserPascalCustomNode import HumanParserPascalCustomNode

NODE_CLASS_MAPPINGS = {
  "Cozy Human Parser LIP" : HumanParserLIPCustomNode,
  "Cozy Human Parser ATR" : HumanParserATRCustomNode,
  "Cozy Human Parser Pascal" : HumanParserPascalCustomNode,
}

__all__ = ['NODE_CLASS_MAPPINGS']