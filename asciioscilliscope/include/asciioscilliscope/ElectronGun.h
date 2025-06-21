#pragma once
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace asciioscilliscope {

/**
 * High-Definition Electron Gun & Interior Geometry
 * -----------------------------------------------
 * Simulates a CRT electron beam interacting with an imperfect semi-parabolic,
 * semi-spherical interior surface and phosphor screen. Operates on a full-resolution
 * HD tensor until sampling, producing time-sliced diff masks via PixelFrameBuffer.
 *
 * Pipeline:
 *  1. Beam Emission: generate an HD diff mask tensor [channels, HD_rows, HD_cols]
 *     based on confinement, gain, beam angle, and field curvature functions.
 *  2. Deflection Circuits: compute per-pixel (x,y) offsets by integrating input
 *     waveforms (sine, saw, triangle, custom) and field strength over time.
 *  3. Interior Reflection: apply a single-reflection ray-tracing step using
 *     geometry parameters and reflectivity coefficients to modulate intensity.
 *  4. Diffusion in Transit: model beam broadening via Gaussian convolution stub.
 *  5. Buffer Output: write incremental diff into a PixelFrameBuffer<DataType>
 *     for downstream phosphor sampling.
 *
 * Geometry & Beam Settings:
 *   struct InteriorGeometry { double radiusX, radiusY, curvature; double reflectivity; };
 *   struct DeflectionSettings { double gainRise, gainFall; double maxAngle; };
 *   enum class Waveform { Sine, Saw, Triangle, Square, Custom };
 *
 * Extended API Method Stubs:
 *   - loadInteriorGeometry(const InteriorGeometry& geom)
 *   - setDeflectionParameters(const DeflectionSettings& def)
 *   - attachSignalGenerator(Waveform mode, double frequency)
 *   - emitTimeSlice(double timestamp)
 *   - simulateReflection(int steps)
 *
 * This class integrates with PixelFrameBuffer for double-buffered diffs,
 * and can operate in full vs differential modes as part of the CRT simulation.
 */

/**
 * ElectronGun<DataType>
 * ----------------------
 * Simulates a high-definition CRT electron gun beam delivering energy diffs
 * across a display plane. Produces time-sliced diff masks on the HD tensor.
 *
 * Template Parameters:
 *   - DataType: Numeric type for beam intensity (e.g., float)
 *
 * Responsibilities:
 *   - Configure beam focusing parameters: ConfinementRadius, FieldCurvature,
 *     BeamAngle, Gain
 *   - Accept normalized HD tensor input or procedural patterns
 *   - Generate incremental diff mask tensors for each time slice
 *   - Support broad vs. sharp beam emission modes
 *
 * API Methods:
 *   - setFocusParameters(...) : adjust beam shape
 *   - emitDiffMask(timeStep)  : Eigen::Tensor<DataType,3> [HD_rows, HD_cols, channels]
 *   - reset()                 : reset internal state
 */
template<typename DataType = float>
class ElectronGun {
public:
    /**
     * Constructor
     *
     * @param hdRows      High-definition plane height (pixels)
     * @param hdCols      High-definition plane width (pixels)
     * @param channels    Number of beam channels (e.g., color planes)
     */
    ElectronGun(int hdRows, int hdCols, int channels);

    /**
     * setFocusParameters
     * ------------------
     * Configure mathematical parameters for beam confinement and steering.
     *
     * @param confinementRadius  Effective beam radius (pixels)
     * @param fieldCurvature     Curvature factor for beam spread
     * @param beamAngle          Angle of beam deflection (degrees)
     * @param gain               Multiplicative intensity gain
     */
    void setFocusParameters(DataType confinementRadius,
                            DataType fieldCurvature,
                            DataType beamAngle,
                            DataType gain);

    /**
     * emitDiffMask
     * ------------
     * Generate a diff mask for the given time slice, representing incremental
     * energy delivered by the beam to the HD plane.
     *
     * @param timeStep  Discrete time index or timestamp
     * @return Tensor<DataType,3> of shape [channels, hdRows, hdCols]
     */
    Eigen::Tensor<DataType,3> emitDiffMask(int timeStep);

    /**
     * reset
     * -----
     * Reset internal time-step counters and beam state.
     */
    void reset();

private:
    int hdRows_, hdCols_, channels_;
    DataType confinementRadius_, fieldCurvature_, beamAngle_, gain_;
    int currentStep_;
};

} // namespace asciioscilliscope
