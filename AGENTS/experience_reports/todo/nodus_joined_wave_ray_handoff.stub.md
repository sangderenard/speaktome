# Nodus joined wave/ray handoff

The two-stage 512×512 angular-propagation → aperture-material Nodus schedule
passes on all three Vulkan devices. The ray schedules are still split at
sensor→T1→T2→T3 and T3→T5, with T4 arena continuation separate. Carry the
authored T3 branch and arena identity through the state-library contract,
execute the existing fftfree/Jones/power-accounting sequence around the
compiled arithmetic, and return its continuation to the ray schedule.
See `../1794589010_DOC_Nodus_Joined_Wave_Kernel_Schedule.md`.
