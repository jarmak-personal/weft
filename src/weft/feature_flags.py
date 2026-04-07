"""Feature flag schema for WEFT research pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class FeatureFlags:
    rd_model_selection: bool = True
    candidate_bank: str = "linear5"
    multi_rounds: int = 1
    adaptive_tile_budget: bool = False
    edge_weighted_objective: bool = False
    target_bpp: float | None = None
    enable_res2: bool = False
    search_mode: str = "greedy"
    beam_width: int = 4
    mcmc_steps: int = 24
    stochastic_restarts: int = 0
    early_exit_patience: int = 2
    maxcompute_fit_passes: int = 1
    hierarchical_tiling_level: str = "off"
    edge_budget_boost_level: str = "off"
    primitive_dictionary_level: str = "off"
    stroke_objective_level: str = "off"
    subpixel_primitives_level: str = "off"
    res2_basis_blocks_level: str = "off"
    mixed_action_beam_level: str = "off"
    residual_patch_borrow_level: str = "off"
    decode_refinement_level: str = "off"
    entropy_context_model_level: str = "off"
    split_entropy_streams: bool = False
    neighbor_delta_coding: bool = False
    container_v2_blocks: bool = False
    # Phase 2 #17: split image into albedo*lighting via Retinex; encode
    # albedo with the primitive pipeline and store lighting as a small
    # low-res grid (BLOCK_LITE) that the decoder multiplies back at
    # output time. Off by default until we measure on a wider corpus.
    decompose_lighting: bool = False
    # Resolution of the lighting grid stored in BLOCK_LITE. 32x32 is a
    # reasonable starting point — about 3 KB at int8 per channel and
    # smooth enough that bilinear upsample to source dims looks correct.
    lighting_grid_size: int = 32
    # Brainstorm #11: replace per-tile primitive search with a closed-form
    # bicubic Bézier patch fit (4×4 RGB control grid per tile). The
    # encoder is one linear-algebra pass — orders of magnitude faster
    # than greedy primitive search — and the decoder evaluates the
    # bicubic on the CPU instead of going through the OptiX BVH path.
    # Off by default; enabled via the "bicubic" sweep variant.
    bicubic_patch_tiles: bool = False
    # Brainstorm #20: K-color palette + per-pixel index grid. No tiles,
    # no primitives, no bicubic basis — the entire image is encoded as
    # (palette[K], labels[H][W]). Designed for hard-edged content
    # (screenshots, vector art, pixel art, text) where every other
    # basis fights step discontinuities. The labels grid compresses
    # heavily via the bitstream-level zstd pass since adjacent pixels
    # share labels. ``palette_planes_k`` selects the palette size; 0
    # (default) means the palette path is off.
    palette_planes_k: int = 0
    # Per-tile hybrid: after the primitive-stack greedy fit, also fit
    # a 4×4 bicubic Bézier patch per tile and replace the primitive
    # list with a single PRIM_BICUBIC primitive on tiles where the
    # bicubic has a better R-D score. The bitstream is still a normal
    # primitive-stack adaptive encode (just with a new primitive type
    # PRIM_BICUBIC = 5 mixed in), so the decoder dispatches correctly
    # via the existing render_tile primitive walk. Designed to win on
    # mixed-content images where some tiles are smooth (bicubic) and
    # others are detail-heavy (primitives).
    hybrid_bicubic_per_tile: bool = False
    # Brainstorm #16: per-tile DCT residual layer. After the primitive
    # greedy fit, compute the residual (source - reconstruction),
    # DCT-II it per channel per tile, quantize the coefficients, and
    # store them in a new BLOCK_DCT. Decoder adds the inverse-DCT
    # residual back on top of the primitive reconstruction. Designed
    # to close the natural-photo / dense-frequency gap vs JPEG/WebP
    # by giving the encoder a frequency-domain basis for the residual
    # that the primitive bases can't capture.
    dct_residual: bool = False
    # Quality knob for the DCT residual quantization step. Lower step
    # = finer quantization = higher PSNR / more bytes. ``None`` (the
    # default) derives the step from the encoder ``quality`` field via
    # ``dct_residual.quant_step_for_quality``; setting an explicit
    # value overrides that.
    dct_residual_step: float | None = None
    # Per-frequency quantization weight ramp for the DCT residual.
    # 0 = uniform quantization (legacy, simpler); positive values
    # quantize high-frequency coefficients more aggressively. ~4.0
    # gives JPEG-like ~5× variation between DC and the highest
    # frequency, which closes the low-BPP gap on natural-photo content
    # without affecting high-quality reconstruction.
    dct_residual_freq_alpha: float = 4.0
    # Colorspace + chroma layout for the DCT residual:
    #   0 = RGB (3 full-size channels, no colorspace conversion)
    #   1 = YCbCr 4:4:4 (BT.601, chroma at full tile size)
    #   2 = YCbCr 4:2:0 (BT.601, chroma 2× box-averaged → ~75% chroma
    #       byte savings vs RGB at the same Y quality)
    # 4:2:0 is the JPEG/WebP/MPEG default and exploits the eye's much
    # lower chroma acuity. Default 2 because it's nearly free visually
    # and is what made the residual layer cost-effective on natural
    # photos in the iso-bytes sweep.
    dct_residual_chroma_mode: int = 2
    # Per-tile RMS skip threshold for the DCT residual layer. Tiles
    # whose post-primitive residual has root-mean-square (RMS) below
    # this value (in [0, 1] linear units) are marked absent in the
    # BLOCK_DCT presence bitmask and contribute zero coefficients,
    # saving the per-tile entropy overhead. 0.0 disables skipping
    # (every tile gets coefficients). 0.005 ≈ 1.3 × 8-bit step, which
    # is well below the visual noise floor for natural-photo content.
    dct_residual_skip_threshold: float = 0.005
    # Per-tile mode selection: when True, each tile is R-D scored as
    # either "primitive stack + DCT residual" (current mode) or
    # "empty primitive list + DCT-only residual" (new mode). The
    # encoder picks whichever is cheaper per tile. On natural-photo
    # content the primitive layer is often a net cost — it spends
    # ~100 bytes per tile to capture structure that the DCT's
    # low-frequency coefficients would capture anyway. Requires
    # dct_residual=True (there's no layer to absorb the signal
    # without primitives otherwise). No format change: empty
    # primitive lists + DCT residual already decode correctly.
    dct_residual_per_tile_mode: bool = False
    # BLOCK_DCT v5: per-tile adaptive quantization (experimental).
    # When True, the encoder picks a per-tile multiplicative scale on
    # ``quant_step`` — currently finer quant on the busiest 25% of
    # tiles, base step everywhere else — and stores the scales as a
    # u8-per-tile side stream in BLOCK_DCT. Off by default because
    # the simple energy- and activity-based heuristics tested so far
    # don't beat uniform-finer at iso-bytes on the natural-photo
    # corpus (uniform is approximately the R-D optimum for PSNR on
    # roughly-IID-Gaussian residuals). The format change exists so
    # future heuristics — perceptual masking, R-D-greedy water-
    # filling, learned masks — can drop in without a wire-format
    # break.
    dct_residual_adaptive_quant: bool = False
    # Brainstorm #1: gradient-field encoder. Stores (∂I/∂x, ∂I/∂y)
    # quantized to int8 per channel + per-channel means. Decoder solves
    # the Poisson equation ∇²I = div(grad) via DCT (closed-form, fast).
    # Wins on hard-edge / large-flat-region content where the gradient
    # field is ~99% sparse and zstd crushes the dense int8 maps. Loses
    # on smooth-varying content where every pixel has a small but
    # nonzero gradient that quantizes below the noise floor.
    gradient_field: bool = False
    # Quantization scale for gradient_field. ``int8 unit = 1/scale`` so
    # the representable gradient range is ±127/scale. scale=128 gives
    # range ±0.99 (~1% clipping on pure black-to-white edges) at
    # resolution 1/128 — the right balance for hard-edge content.
    gradient_field_scale: int = 128
    # Soft deadzone threshold for the gradient field — gradients with
    # absolute value below this (in [0, 1] linear units) are zeroed
    # before quantization. Sparsifies the field for hard-edge content.
    gradient_field_threshold: float = 0.005
    # Auto-select: encode the image with each candidate basis
    # (bicubic, palette-16, palette-64) and keep the one with the best
    # rate-distortion score (PSNR - lambda * BPP). The encoder pays
    # the cost of running every candidate but writes only the winner
    # to disk; the bitstream is identical to whichever variant won, so
    # the standard decoder dispatches correctly without changes.
    auto_select: bool = False
    # Rate-distortion lambda for ``auto_select``. Higher = prioritize
    # smaller bitstreams; lower = prioritize PSNR. 4.0 is a reasonable
    # mid-point that picks bicubic on natural photos and palette-64 on
    # hard-edged content for the current corpus.
    auto_select_lambda: float = 4.0
    # Auto-select restriction: when True, only scalable variants
    # compete — the ones that render crisply at arbitrary output
    # resolutions via analytic primitives or nearest-neighbor
    # palette upsampling. Filters out raster variants (hybrid-dct,
    # hybrid-dct-tight, gradient) whose DCT-residual or gradient-
    # field blocks are defined per source pixel and produce
    # bilinear-blurry output when upscaled. Use this when the target
    # use case is resolution-independent output (charts, diagrams,
    # UI mockups, text) — the SVG / icon replacement pitch.
    prefer_scalable: bool = False

    @staticmethod
    def from_dict(data: dict[str, Any] | None) -> "FeatureFlags":
        d = data or {}
        ff = FeatureFlags(
            rd_model_selection=bool(d.get("rd_model_selection", True)),
            candidate_bank=str(d.get("candidate_bank", "linear5")),
            multi_rounds=int(d.get("multi_rounds", 1)),
            adaptive_tile_budget=bool(d.get("adaptive_tile_budget", False)),
            edge_weighted_objective=bool(d.get("edge_weighted_objective", False)),
            target_bpp=(float(d["target_bpp"]) if d.get("target_bpp") is not None else None),
            enable_res2=bool(d.get("enable_res2", False)),
            search_mode=str(d.get("search_mode", "greedy")),
            beam_width=int(d.get("beam_width", 4)),
            mcmc_steps=int(d.get("mcmc_steps", 24)),
            stochastic_restarts=int(d.get("stochastic_restarts", 0)),
            early_exit_patience=int(d.get("early_exit_patience", 2)),
            maxcompute_fit_passes=int(d.get("maxcompute_fit_passes", 1)),
            hierarchical_tiling_level=str(d.get("hierarchical_tiling_level", "off")),
            edge_budget_boost_level=str(d.get("edge_budget_boost_level", "off")),
            primitive_dictionary_level=str(d.get("primitive_dictionary_level", "off")),
            stroke_objective_level=str(d.get("stroke_objective_level", "off")),
            subpixel_primitives_level=str(d.get("subpixel_primitives_level", "off")),
            res2_basis_blocks_level=str(d.get("res2_basis_blocks_level", "off")),
            mixed_action_beam_level=str(d.get("mixed_action_beam_level", "off")),
            residual_patch_borrow_level=str(d.get("residual_patch_borrow_level", "off")),
            decode_refinement_level=str(d.get("decode_refinement_level", "off")),
            entropy_context_model_level=str(d.get("entropy_context_model_level", "off")),
            split_entropy_streams=bool(d.get("split_entropy_streams", False)),
            neighbor_delta_coding=bool(d.get("neighbor_delta_coding", False)),
            container_v2_blocks=bool(d.get("container_v2_blocks", False)),
            decompose_lighting=bool(d.get("decompose_lighting", False)),
            lighting_grid_size=int(d.get("lighting_grid_size", 32)),
            bicubic_patch_tiles=bool(d.get("bicubic_patch_tiles", False)),
            palette_planes_k=int(d.get("palette_planes_k", 0)),
            hybrid_bicubic_per_tile=bool(d.get("hybrid_bicubic_per_tile", False)),
            gradient_field=bool(d.get("gradient_field", False)),
            gradient_field_scale=int(d.get("gradient_field_scale", 128)),
            gradient_field_threshold=float(d.get("gradient_field_threshold", 0.005)),
            dct_residual=bool(d.get("dct_residual", False)),
            dct_residual_step=(
                float(d["dct_residual_step"])
                if d.get("dct_residual_step") is not None
                else None
            ),
            dct_residual_freq_alpha=float(d.get("dct_residual_freq_alpha", 4.0)),
            dct_residual_chroma_mode=int(d.get("dct_residual_chroma_mode", 2)),
            dct_residual_skip_threshold=float(d.get("dct_residual_skip_threshold", 0.005)),
            dct_residual_adaptive_quant=bool(d.get("dct_residual_adaptive_quant", False)),
            dct_residual_per_tile_mode=bool(d.get("dct_residual_per_tile_mode", False)),
            auto_select=bool(d.get("auto_select", False)),
            auto_select_lambda=float(d.get("auto_select_lambda", 4.0)),
            prefer_scalable=bool(d.get("prefer_scalable", False)),
        )
        if ff.multi_rounds < 1:
            raise ValueError("feature_flags.multi_rounds must be >= 1")
        if ff.candidate_bank not in {"linear5", "rich18"}:
            raise ValueError(f"unsupported feature_flags.candidate_bank: {ff.candidate_bank}")
        if ff.search_mode not in {"greedy", "beam", "mcmc"}:
            raise ValueError(f"unsupported feature_flags.search_mode: {ff.search_mode}")
        if ff.beam_width < 1:
            raise ValueError("feature_flags.beam_width must be >= 1")
        if ff.mcmc_steps < 0:
            raise ValueError("feature_flags.mcmc_steps must be >= 0")
        if ff.stochastic_restarts < 0:
            raise ValueError("feature_flags.stochastic_restarts must be >= 0")
        if ff.early_exit_patience < 0:
            raise ValueError("feature_flags.early_exit_patience must be >= 0")
        if ff.maxcompute_fit_passes < 1:
            raise ValueError("feature_flags.maxcompute_fit_passes must be >= 1")
        if ff.lighting_grid_size < 4 or ff.lighting_grid_size > 256:
            raise ValueError("feature_flags.lighting_grid_size must be in [4, 256]")
        if ff.palette_planes_k < 0 or ff.palette_planes_k > 256:
            raise ValueError("feature_flags.palette_planes_k must be in [0, 256] (0 = off)")
        if ff.gradient_field_scale < 1 or ff.gradient_field_scale > 65535:
            raise ValueError("feature_flags.gradient_field_scale must be in [1, 65535]")
        if ff.gradient_field_threshold < 0:
            raise ValueError("feature_flags.gradient_field_threshold must be >= 0")
        if ff.dct_residual_step is not None and ff.dct_residual_step <= 0:
            raise ValueError("feature_flags.dct_residual_step must be > 0 (or None)")
        if ff.dct_residual_freq_alpha < 0:
            raise ValueError("feature_flags.dct_residual_freq_alpha must be >= 0")
        if ff.dct_residual_chroma_mode not in (0, 1, 2):
            raise ValueError(
                "feature_flags.dct_residual_chroma_mode must be 0 (RGB), "
                "1 (YCbCr 4:4:4) or 2 (YCbCr 4:2:0)"
            )
        if ff.dct_residual_skip_threshold < 0:
            raise ValueError("feature_flags.dct_residual_skip_threshold must be >= 0")
        for level in (
            ff.hierarchical_tiling_level,
            ff.edge_budget_boost_level,
            ff.primitive_dictionary_level,
            ff.stroke_objective_level,
            ff.subpixel_primitives_level,
            ff.res2_basis_blocks_level,
            ff.mixed_action_beam_level,
            ff.residual_patch_borrow_level,
            ff.decode_refinement_level,
            ff.entropy_context_model_level,
        ):
            if level not in {"off", "mid", "max"}:
                raise ValueError(f"unsupported technique level: {level}")
        return ff
