from pydantic import BaseModel
from typing import List

class Startup(BaseModel):
    startupName: str
    description: str
    industry: str

class ClusterRequest(BaseModel):
    startups: List[Startup]

class ClusteredStartup(Startup):
    cluster_id: int
    label: str

class ClusterResponse(BaseModel):
    clustered: List[ClusteredStartup]
