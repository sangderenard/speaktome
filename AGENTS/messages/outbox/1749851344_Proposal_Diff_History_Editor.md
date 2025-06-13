# Proposal: Diff History Editor

A conceptual flag titled **Diff History Editor** has been added describing a
lightweight tool that replays or rewinds archived `.diff` patches.

Agents should generate patches from local commits, delete the commit, and place
the diff along with an explanatory memo here for review. These patches then move
to an archive after review, forming a reversible changelog.

See `AGENTS/conceptual_flags/Diff_History_Editor.md` for details.
