#pragma once
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace asciioscilliscope {

/**
 * CharClassifier
 * --------------
 * Converts aggregated color values into ASCII characters using a brightness
 * ramp. A single call to `classify` represents an averaged measurement from a
 * console-character sized region rather than a one-to-one pixel mapping.
 * Supports future extensions for per-channel phosphor timing offsets and
 * batched tensor classifications.
 *
 * Responsibilities:
 *   - Maintain a character ramp lookup (configurable)
 *   - Map normalized or raw color values to ASCII symbols
 *   - Optionally apply per-channel delays for realistic phosphor simulation
 *
 * Note: a console character represents many HD pixels. Classification never
 * maps a single HD sample directly to ASCII. Inputs should be averaged or
 * otherwise reduced to a representative brightness value. The number of color
 * channels is not fixed; the classifier works with any channel count as long as
 * the data is aggregated into a brightness value per console site. Use the
 * vector-based API for fully general input.
 *
 * API:
 *   - classify(r, g, b): maps an aggregated RGB sample to a char
 *   - classifyBatch(tensor): maps entire color tensor to char tensor
 *
 * TODO:
 *   - Expose custom ramp strings via constructor or setter
 *   - Add classifyBatch(const Eigen::Tensor<float,4>&) for tensors
 *   - Implement per-channel temporal offsets for R/G/B timing shifts
 */
class CharClassifier {
public:
    /**
     * Constructor
     * -----------
     * Optionally supply a custom ramp string used for brightness mapping.
     * If not provided, a default 70-character ramp is used.
     */
    explicit CharClassifier(const std::string& ramp = "");
    /**
     * classify
     * --------
     * Maps an RGB brightness sample to an ASCII character. The RGB values may
     * represent the average over many pixels. Extra color channels, if present,
     * should be blended before calling this method. This is strictly a
     * convenience wrapper around the generic vector-based `classify` method.
     *
     * @param r  Aggregated red component [0,255]
     * @param g  Aggregated green component [0,255]
     * @param b  Aggregated blue component [0,255]
     * @return ASCII symbol representing brightness
     */
    char classify(uint8_t r, uint8_t g, uint8_t b) const;

    /**
     * classify
     * --------
     * Variant that accepts an arbitrary number of channel intensities.
     * The brightness is computed by averaging all provided values.
     *
     * @param channels Vector of intensity samples [0,255]
     * @return ASCII symbol representing brightness
     */
    char classify(const std::vector<uint8_t>& channels) const;

    /**
     * setRamp
     * -------
     * Replace the brightness ramp used for classification.
     * The string should be ordered from darkest to brightest.
     */
    void setRamp(const std::string& ramp);

    /**
     * classifyBatch
     * -------------
     * (Future) Process a 4D tensor [batch, rows, cols, channels]
     * and return a 3D tensor of chars [batch, rows, cols].
     *
     * @param colorTensor  Eigen::Tensor<float,4> normalized [0.0f,1.0f]
     * @return Eigen::Tensor<char,3> of ASCII codes
     */
    // Eigen::Tensor<char,3> classifyBatch(const Eigen::Tensor<float,4>& colorTensor) const;

    /**
     * ChannelMode
     * -----------
     * Defines how multiple color channels are handled during ASCII conversion:
     *   - Blend   : mix channels into a single luminance stream before mapping
     *   - Discrete: classify each channel separately and combine symbols/colors
     */
    enum class ChannelMode { Blend, Discrete };

    /**
     * SiteMetadata
     * ------------
     * Describes a sampling site in the reduced grid:
     *   - siteRow, siteCol : location in low-res sampling grid
     *   - radius            : HD radius for averaging intensity
     *   - hdRowStart, hdColStart : HD plane coordinates of the subregion
     */
    struct SiteMetadata {
        int siteRow;
        int siteCol;
        int radius;
        int hdRowStart;
        int hdColStart;
    };

    /**
     * classifyHD
     * ----------
     * Process a high-definition tensor of shape [channels, HD_rows, HD_cols]
     * and produce a low-res ASCII tensor [HD_rows, HD_cols], mapping each
     * pixel via the brightness ramp and ChannelMode.
     *
     * @param hdTensor  Eigen::Tensor<float,3> normalized intensities
     * @param mode      ChannelMode::Blend or Discrete
     * @return Eigen::Tensor<char,2> ASCII codes
     */
    // Eigen::Tensor<char,2> classifyHD(const Eigen::Tensor<float,3>& hdTensor,
    //                                  ChannelMode mode = ChannelMode::Blend) const;

    /**
     * classifySampleSites
     * -------------------
     * Convert a reduced SampleTensor [channels, numSites] to ASCII symbols
     * using provided site metadata. Applies ChannelMode to mix or separate.
     *
     * @param sampleTensor  Eigen::Tensor<float,2> [channels, numSites]
     * @param sites         Vector of SiteMetadata for each site
     * @param mode          ChannelMode
     * @return Vector of (siteRow, siteCol, char, r, g, b)
     */
    // std::vector<std::tuple<int,int,char,uint8_t,uint8_t,uint8_t>>
    // classifySampleSites(const Eigen::Tensor<float,2>& sampleTensor,
    //                     const std::vector<SiteMetadata>& sites,
    //                     ChannelMode mode = ChannelMode::Blend) const;

private:
    std::string ramp_;
};

} // namespace asciioscilliscope
