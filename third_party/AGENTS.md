# Third-Party Dependencies

This directory vendors all external C libraries used by the low-level
implementations. Each subfolder corresponds to a project pinned to a
specific commit or release.

Whenever possible, keep the sources here so that builds are entirely
offline and reproducible. If a dependency is missing, the Zig build
script will attempt to fetch the required source code on demand.
