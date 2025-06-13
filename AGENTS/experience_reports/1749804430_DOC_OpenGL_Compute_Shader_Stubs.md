# OpenGL Compute Shader Stubs

**Date/Version:** 1749804430 v1
**Title:** OpenGL_Compute_Shader_Stubs

## Overview
Added a new directory `glsl_kernels` containing stub compute shader files with
clear sentinel markers. These stubs will allow future composition of shader
segments into batch operations. An accompanying `AGENTS.md` explains the folder
purpose.

## Prompts
"In accelerated backends i want you to make c source and glsl or whatever it's called source folder, i think we already have a c source folder or file of some sorts. Create some opengl script stubs for compute shaders that have comment sentinels isolating their initialization code from their operation code from their output code. They will be able to be run alone but also composited and compiled to form a cache of operation chains derived by placing the middles from all the necessary steps for a batch operation. The opengl backend will decide how long it can keep making nested changes before it must pull redistribute the data to a new compute shader batch configuration"
