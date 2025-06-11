# Todo Folder

This directory holds open work items and prototypes.

## Stub Tracking

`AGENTS/tools/stubfinder.py` scans the codebase for `STUB:` blocks and writes
each one here as an individual `.stub.md` file. The filename is based on the
source path and starting line number.

Running the tool first deletes existing `.stub.md` files, ensuring results stay
current rather than piling up. The developer setup scripts invoke this utility
automatically so stubs are captured at startup.

Agents may implement a stub by creating a new file with the same basename but a
different extension (e.g., `.py`, `.txt`). Please call out any tasks that need
explicit human judgment or access.

## Prototyping Welcome

Autonomous agents are encouraged to drop experimental drafts or small scripts in
this folder. Keep filenames expressive and mirror the stub names when
appropriate so human collaborators can easily cross-reference.

Initialization logs are not automatically ingested by language models. Review
these files as needed when exploring the repository.
