"""`python -m figrip ...` is the same as the `figrip` command."""
import sys

from .cli import main

sys.exit(main())
