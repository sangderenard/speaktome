# Parametric L-System Render Function

**Date:** 2026-08-01
**Title:** Added the source-level RGB render function for the L-system demo

## Activity

Added one module-level `render` function after `ParametricLSystem`. It has no
`main` wrapper and uses the class directly: the frame counter selects one of
the six built-in presets, `preset()` constructs it, and `trace()` produces the
turtle segments. The function draws one complete RGB frame and returns its
red, green, and blue byte planes. Repeated site calls advance `t`, cycling the
presets.

The palette follows the existing Mandelbrot renderer's three cosine-offset
channels. No alternate renderer class or separate runtime adapter remains in
the source.

## Verification

No execution or compilation probe was retained for this change. The requested
work was limited to adding the render function rather than evaluating it.

## Prompt History

> last thing we need to do is make a main for the file that produces a demo compatible with the site's system such that it will draw a bunch of patterns super fast in loops when the method is run, not necessarily a real main, but something the site can run that will pump out image data in color of patterns from different presets or randomized parameterizations

> just look at the mandlebrot render function

> you aren't coding math you're just using the class in a function to make images

> just put the function after the class

> don't use main

> do it the way the mandlebrot does

> you aren't being asked to evaluate that

> you are not being asked to make creative choices; you were told what to do

## Next Steps

None recorded as a guestbook task.
