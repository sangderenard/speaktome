# Investigate Apple oil-toy SDF contact path (parked)

Public iPhone ticket `1794100682-1467606889` and server log are preserved.
See `../1794101000_LOG_Apple_Oil_Toy_Contact_Investigation_Parked.md`.
The failure is oil-toy specific; compare Apple compiled SDF distance/normal,
contact impulse and finite checks at the first divergent step with a desktop
run. Do not treat the separate Marble-to-Fountain gravity leak as this
regression. The issue is parked while marble camera and saved-mesh work finishes.
