from sqlalchemy import Index

from neurondb.schemas.tables import SQLANeuronDescription

DB_INDICES = [
    Index(
        f"{SQLANeuronDescription.__tablename__}_hnsw_cosine_index",
        SQLANeuronDescription.description_embedding,
        postgresql_using="hnsw",
        postgresql_with={"m": 32, "ef_construction": 200},
        postgresql_ops={SQLANeuronDescription.description_embedding.name: "vector_cosine_ops"},
    ),
]
