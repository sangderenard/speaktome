# Nodus shader artifact packaging handoff

Have Turing's artifact publishing path place emitted shader source beside its
`turing-compiled-program-api-v1` YAML, or emit a
`nodus-turing-shader-artifact-v1` request manifest referencing both. The API
schema itself already has the required language, stage, entrypoint, and
component ABI fields.
