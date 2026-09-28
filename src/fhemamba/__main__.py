"""Support the same installed CLI through python -m fhemamba."""

from .cli import main

raise SystemExit(main())
