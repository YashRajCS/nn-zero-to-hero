# L4: Reading the Diagnostics (6-layer tanh MLP, embed_dim=20, hidden=200)

## Activation distribution

- The histogram is a gentle U-shape with slight horns near -1 and +1, mean ~0, and all 5 layers sit on top of each other (consistent depth-to-depth).
- Why the U: the pre-activations are roughly Gaussian (a bell). tanh is steep near 0 and flat near ±1, so it spreads the middle values out and compresses the tails into a narrow band near ±1 — that pile-up is the horns. Nothing is clipped; this is just the natural shape of tanh(Gaussian).
- My net reads ~2% saturated (|a| > 0.97), meaning only ~2% of neurons sit in the flat dead zone where the gradient ≈ 0. So ~98% are in the responsive region and can still learn. A gentle U like this = healthy.
- The bad version: a sharp U with tall horns and high saturation %, where most neurons are stuck at ±1, gradients there ≈ 0, and those neurons barely train.

## Gradient distribution

- Roughly equal spread across all 5 layers, no shrinking or widening with depth → gradients flow cleanly to every layer → init is good.
- The failure mode to contrast against: gradients that shrink layer-by-layer with depth (vanishing) or blow up (exploding).

## grad:data ratio (static, per weight)

- Definition: gradient magnitude / weight magnitude - how big the proposed nudge is relative to the weight itself.
- My read: layer 0 (embedding-side) and the last layer (output weights) show higher ratios than the middle layers; the middle hidden weights are lower and tightly clustered.

## update:data ratio (over time)

- Definition: (learning_rate × gradient) / weight magnitude — the size of the step I actually took, relative to the weight. This indicates how healthy our learning rate is.
- The black line at -3 (1e-3) is the healthy reference. Below it = steps too small (learning too slow); above it = steps too big (too fast).

## Bad-init A/B

- With BatchNorm on, even a bad init stays fairly tame, as the BN re-centers/re-scales each layer's pre-activations, so the histogram doesn't fully collapse. This is BN's main job: keep layers trainable regardless of init quality.
- With BatchNorm removed AND a bad init, the activations go to a sharp U: middle near-empty, almost all mass piled in tall horns at ±1 (high saturation). Those saturated neurons have gradient ≈ 0, so they barely learn — the failure mode the good init avoids.
