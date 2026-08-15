"""agents — specialist worker nodes and the supervisor/critic pair.

Nothing is re-exported here on purpose. workflow/registry.py imports each
worker by module path, and eager imports at package level would drag the whole
scraping stack (playwright, goose3, newspaper) into any process that only
wanted one agent.
"""
