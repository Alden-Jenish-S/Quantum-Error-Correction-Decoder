# Scope incident

During file creation, the agent accidentally added a root file named
`SCALING_DO_NOT_CREATE`, then an empty `SCOPE_ERROR` file while removing it.
Both newly created files were immediately removed. No existing root file was
edited or deleted. This was a scope violation; final deliverables are under D.
