import mitsuba as mi
import matplotlib.pyplot as plt

# Set the Mitsuba variant to scalar spectral polarized
mi.set_variant('scalar_spectral')

# Load the scene file
scene = mi.load_file('.\\dining-room\\dining-room\\scene.xml')

# Render the scene with high sample count
spp = 24  # High quality rendering
image = mi.render(scene, spp=spp)

# Save the output as a multi-channel EXR file
output_file = 'polarized_render_660nm.exr'
bitmap = mi.Bitmap(image, channel_names=['R', 'G', 'B'] + scene.integrator().aov_names())
bitmap.write(output_file)

# Visualization of S0 (Intensity channel)
channels = dict(bitmap.split())

plt.figure(figsize=(5, 5))
plt.imshow(channels['S0'].convert(srgb_gamma=True), cmap='gray')
plt.colorbar()
plt.xticks([]); plt.yticks([])
plt.xlabel("S0: Intensity", size=14, weight='bold')
plt.title("Polarized Intensity at 660nm")
plt.show()

# Visualization of Stokes components S1, S2, and S3
def plot_stokes_component(ax, image, title):
    data = mi.TensorXf(image)[:, :, 0]
    plot_minmax = 0.05 * max(mi.dr.max(data), mi.dr.max(-data))
    img = ax.imshow(data, cmap='coolwarm', vmin=-plot_minmax, vmax=+plot_minmax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, size=14, weight='bold')
    return img

fig, ax = plt.subplots(ncols=3, figsize=(18, 5))
img1 = plot_stokes_component(ax[0], channels['S1'], "S1: Horizontal vs. Vertical")
img2 = plot_stokes_component(ax[1], channels['S2'], "S2: Diagonal Polarization")
img3 = plot_stokes_component(ax[2], channels['S3'], "S3: Circular Polarization")

plt.colorbar(img1, ax=ax[0])
plt.colorbar(img2, ax=ax[1])
plt.colorbar(img3, ax=ax[2])

plt.show()

print(f"Render saved to {output_file}")
