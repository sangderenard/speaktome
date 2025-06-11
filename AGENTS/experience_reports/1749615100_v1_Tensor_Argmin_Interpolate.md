# Implemented argmin and interpolation

## Prompt History
- User: "Implement argmin and interpolate for the abstract tensor suite of backends, for now just do dimension-wise linear interpolation with indices as fixed width, unless you know offhand a better resizing algorithm for dimensional floats"

## Notes
Implemented `argmin` and `interpolate` across tensor backends and updated tests. The ASCII kernel classifier now relies on these ops for final classification.
