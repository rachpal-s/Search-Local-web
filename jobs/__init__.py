"""jobs — batch ingestion of folder trees into named collections.

Layout mirrors docstore/: models are dataclasses, persistence is SQLite in the
same database, and the pipeline reuses docstore's stage functions rather than
reimplementing them.

    models.py    Job / FolderSpec / JobOptions / Phase
    store.py     queue persistence and the claim protocol
    pipeline.py  the multicore per-document pipeline
    runner.py    phase orchestration for one job
    worker.py    the daemon; run as a separate process from uvicorn
"""
