"""The documents the shipped types of team keep, and what each does.

Each is a YAML file here. One that does more than hold a text its team
rewrites has a module beside it with its protocol.
"""

from openhands.agent_server.director.orgs.documents.contracts import ContractsProtocol
from openhands.agent_server.director.orgs.documents.spec import SpecProtocol


__all__ = ["ContractsProtocol", "SpecProtocol"]
