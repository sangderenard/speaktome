# Nodus Radio direct FLAC deployment

The Turing upload route requires the existing host-declared Nodus codec in
`run/nodus_radio.json`. That file was absent in the inspected live state.

After both coordinated reviews land, the operator should install the existing
Radio file-station configuration, confirm native executable and codec paths,
resume the station, and verify HLS. Then barnacleboy4 should upload the
Lantern and Barnacles FLAC masters through the Radio panel. Compare each
returned SHA-256 with the creator's ledger. Upload success alone does not
establish audible playback.
