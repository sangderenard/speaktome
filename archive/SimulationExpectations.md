Color CRTs achieve full-color display with a combination of multiple electron beams and precise masking/gridding that aligns those beams to tiny phosphor areas. Here’s how it works in practice—and how “crude” versus “refined” implementations differ:

---

## 1. Three Electron Guns, One per Color

Almost every color CRT (especially traditional consumer TVs and monitors) uses **three** electron guns:

* **Red**, **Green**, and **Blue** guns\*\* sit at slightly different horizontal positions.
* They’re carefully aimed so that each beam only excites its matching phosphor dots or stripes on the screen.

## 2. Shadow Mask or Aperture Grille

To ensure each beam lands only on the correct phosphor, screens use a metal grid:

| Mask Type           | Description                                                                                                                                                                                                                                                   |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Shadow Mask**     | A thin steel plate punched with millions of tiny holes, positioned just behind the phosphor layer. Each hole admits all three beams, but they strike different phosphor dots depending on angle. Older, “crude” masks had larger holes and coarser dot-pitch. |
| **Aperture Grille** | Vertical slots (rather than holes), as in Sony’s Trinitron tubes. Offers brighter images and finer alignment, but requires very precise beam focus and stabilization.                                                                                         |

### Crude Grids vs. Fine Masks

* **Crude grid**: Early or low-end tubes could have wider mask holes or coarser triad spacing. The beams were less sharply focused, so a given hole would let some spillover between dots—leading to bloom or color fringing.
* **Fine mask (later models)**: Tighter hole or slot spacing and improved beam focusing systems (electrostatic and magnetic lenses) meant electrons hit phosphors more cleanly, reducing crosstalk and boosting apparent resolution.

## 3. Beam Focusing and Deflection

Beyond the physical mask:

* **Focusing electrodes** inside the CRT shape each beam to a tight spot. Early designs used a simple electrostatic lens—many later ones added magnetic focusing coils around the tube’s neck for even tighter control.
* **Beam deflection** (via coils around the tube) sweeps each beam across the screen in sync with the video signal. The “stride” is effectively the raster scan: the beam moves one phosphor-pitch at a time horizontally, then steps down one line vertically.

## 4. Phosphor Coating and Diffusion

* **Older phosphor coatings** were relatively thick and somewhat unevenly distributed. The raw electron “dot” would spread in the phosphor layer, creating a soft glow bigger than the beam spot—this is the inherent diffusion you mentioned.
* **Modern phosphor formulations** use finer grains and layering techniques, which can both sharpen and control that diffusion so each dot lights evenly without excessive bleed.
* Some high-end tubes even used multiple layers (a thin high-resolution layer over a thicker diffusion layer) to get both sharp detail and smooth glow.

## 5. Temporal Multiplexing and CRTs

While not common in classic CRTs, you can think of “temporal grids” in modern digital simulations:

* If you wanted to simulate multiple sub-beams or time-sliced raster passes, you could treat each scanline or sub-frame as a separate layer (much like our 4D buffer).
* In real tubes, the phosphor persistence (decay rate) naturally creates a temporal smoothing, so you see the accumulated glow of many rapid passes rather than discrete spots.

---

### So, to answer your questions directly:

1. **Do color CRTs use multiple beams or a stride-matched grid?**
   They use **multiple beams** (three guns), paired with a **shadow mask** or **aperture grille** that matches the phosphor layout. The “stride” of the raster scan is mechanically set by deflection coils to match the physical phosphor pitch.

2. **Is there a thing as a crude grid aligned to the emitter center?**
   Yes—early or low-res tubes had coarser masks/slits. The holes or slots were simply aligned to the nominal beam positions, but beam focus and mask precision were low, so the resulting image was blurrier.

3. **Older models hitting phosphor vs. later fine-articulated focusing into diffusion surfaces?**
   Exactly.

   * **Early CRTs**: bigger holes, thicker phosphor, broader beams → large, soft spots.
   * **Later CRTs**: fine-pitch masks, magnetic focusing, advanced phosphor formulations → tight spots and controlled diffusion, yielding higher resolution and crisper pixels.

---

By viewing the CRT as a **spatiotemporal diffusion system**—with three beams, a precise mask, electrostatic/magnetic focusing, and phosphor persistence—you can map each element to the kinds of kernels and time-slice interpolations we’ve been designing in software.
