// Weft site script. No framework, no build step.
//
// Responsibilities:
//  1. Fetch assets/bench_data.json produced by generate_site_assets.py
//  2. Populate hero stats + benchmark table from the JSON
//  3. Render the gallery grid (one card per fixture) with interactive
//     image-comparison sliders
//  4. Wire the gallery toggle (JPEG / WebP / Source) so it swaps the
//     B-side of every slider, plus the delta badge and stat caption
//  5. Wire the slider drag + touch + keyboard interaction
//
// The JSON schema is defined by scripts/generate_site_assets.py.

(function () {
  "use strict";

  const DATA_URL = "assets/bench_data.json";

  // Describes how to render each comparison mode. Each entry maps to the
  // shape of bench_data.json fields and the labels shown in the UI.
  const MODES = {
    jpeg: {
      assetKey: "jpeg",
      isoKey: "jpeg_iso",
      deltaKey: "delta_db",
      sliderLabel: "JPEG ▶",
      statLabel: (f) => `JPEG (iso-bytes, q${f.jpeg_iso.quality})`,
      altText: (f) => `JPEG at iso-bytes: ${f.title}`,
      // When true, the badge shows Δ = weft − other. When false (e.g. Source
      // mode), the badge shows the Weft absolute PSNR since comparing
      // against ground truth doesn't produce a meaningful delta.
      hasDelta: true,
    },
    webp: {
      assetKey: "webp",
      isoKey: "webp_iso",
      deltaKey: "delta_db_vs_webp",
      sliderLabel: "WebP ▶",
      statLabel: (f) => `WebP (iso-bytes, q${f.webp_iso.quality})`,
      altText: (f) => `WebP at iso-bytes: ${f.title}`,
      hasDelta: true,
    },
    source: {
      assetKey: "source",
      // No iso key — source is the uncompressed ground truth. Stats show
      // the source's PNG-lossless byte count (a reference point) and the
      // "lossless" indicator instead of a PSNR number.
      isoKey: null,
      deltaKey: null,
      sliderLabel: "Source ▶",
      statLabel: () => "Source (ground truth, lossless)",
      altText: (f) => `Source: ${f.title}`,
      hasDelta: false,
    },
  };

  const DEFAULT_MODE = "jpeg";

  // Closure state used by the toggle handler.
  let allFixtures = [];
  let currentMode = DEFAULT_MODE;

  /* ─────────────────────────── fetch + bootstrap ─────────────────── */

  async function boot() {
    let data;
    try {
      const res = await fetch(DATA_URL);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      data = await res.json();
    } catch (err) {
      console.error("Failed to load bench_data.json:", err);
      renderError(
        "Could not load the comparison data. If you're viewing this file " +
          "directly from disk, fetch() is blocked by browser security — " +
          "run `python -m http.server` from the docs/ directory and visit " +
          "http://localhost:8000/ instead."
      );
      return;
    }

    allFixtures = data.fixtures;

    populateHeroStats(data);
    renderGallery(allFixtures);
    renderBenchTable(data);
    initSliders();
    initToggle();
  }

  function renderError(msg) {
    const g = document.getElementById("gallery-grid");
    if (g) {
      g.innerHTML = `<div class="gallery-loading">${escapeHtml(msg)}</div>`;
    }
    const tb = document.getElementById("bench-tbody");
    if (tb) {
      tb.innerHTML = `<tr><td colspan="6" class="tbody-loading">${escapeHtml(
        msg
      )}</td></tr>`;
    }
  }

  /* ─────────────────────────── hero stats ─────────────────────────── */

  function populateHeroStats(data) {
    const winsEl = document.getElementById("stat-wins");
    if (winsEl) {
      winsEl.innerHTML = `${data.wins_jpeg}<span class="stat-div">/</span>${data.fixture_count}`;
    }
    const deltaEl = document.getElementById("stat-avg-delta");
    if (deltaEl) {
      const sign = data.avg_delta_db_vs_jpeg >= 0 ? "+" : "";
      deltaEl.textContent = `${sign}${data.avg_delta_db_vs_jpeg.toFixed(2)} dB`;
    }
    const weftTotalEl = document.getElementById("stat-weft-total");
    if (weftTotalEl) {
      weftTotalEl.textContent = `${Math.round(data.totals_kb.weft)} KB`;
    }
    const ratioEl = document.getElementById("stat-png-ratio");
    if (ratioEl) {
      const ratio = data.totals_kb.png_lossless / data.totals_kb.weft;
      ratioEl.textContent = `${ratio.toFixed(1)}×`;
    }
  }

  /* ─────────────────────────── gallery rendering ─────────────────── */

  function renderGallery(fixtures) {
    const grid = document.getElementById("gallery-grid");
    if (!grid) return;

    // Sort: biggest positive wins vs JPEG first (most visually striking),
    // then losses at the end for the honest story. This ordering is
    // independent of the currently-selected comparison mode — we want the
    // list order stable even when the user toggles the comparison.
    const sorted = fixtures.slice().sort((a, b) => b.delta_db - a.delta_db);

    const html = sorted.map((f) => buildGalleryCard(f, DEFAULT_MODE)).join("");
    grid.innerHTML = html;
  }

  function buildGalleryCard(f, mode) {
    const modeSpec = MODES[mode];
    const other = modeSpec.isoKey ? f[modeSpec.isoKey] : null;

    const deltaBadge = buildDeltaBadge(f, modeSpec);
    const weftKb = (f.weft.bytes / 1024).toFixed(1);
    const bInfo = buildBStats(f, modeSpec);

    return `
      <article class="gallery-card" data-fixture="${escapeHtml(f.name)}">
        <div class="gallery-header">
          <div class="gallery-meta">
            <h3 class="gallery-title">${escapeHtml(f.title)}</h3>
            <span class="gallery-variant">variant: ${escapeHtml(f.variant)}</span>
          </div>
          ${deltaBadge}
        </div>
        <p class="gallery-note">${escapeHtml(f.note)}</p>

        <div class="compare-slider" data-name="${escapeHtml(f.name)}">
          <div class="compare-a">
            <img src="${escapeHtml(f.assets.weft)}" alt="Weft decoded: ${escapeHtml(f.title)}" loading="lazy">
          </div>
          <div class="compare-b">
            <img data-role="b-img"
                 src="${escapeHtml(f.assets[modeSpec.assetKey])}"
                 alt="${escapeHtml(modeSpec.altText(f))}"
                 loading="lazy">
          </div>
          <div class="compare-labels">
            <span class="compare-label-a">◀ Weft</span>
            <span class="compare-label-b" data-role="b-label">${modeSpec.sliderLabel}</span>
          </div>
          <div class="compare-handle" aria-hidden="true"></div>
        </div>

        <div class="gallery-stats">
          <div class="gallery-stat">
            <div class="gallery-stat-label">Weft</div>
            <div class="gallery-stat-num"><strong>${weftKb} KB</strong> &middot; ${f.weft.psnr_db.toFixed(2)} dB</div>
          </div>
          <div class="gallery-stat" data-role="b-stat">
            <div class="gallery-stat-label" data-role="b-stat-label">${escapeHtml(modeSpec.statLabel(f))}</div>
            <div class="gallery-stat-num" data-role="b-stat-num">${bInfo}</div>
          </div>
        </div>
      </article>
    `;
  }

  function buildDeltaBadge(f, modeSpec) {
    if (!modeSpec.hasDelta) {
      // Source mode: show Weft's absolute PSNR as the "headline" number
      return `
        <div class="gallery-delta" data-role="delta">
          ${f.weft.psnr_db.toFixed(2)} dB
        </div>
      `;
    }
    const delta = f[modeSpec.deltaKey];
    const sign = delta >= 0 ? "+" : "";
    const cls = delta >= 0 ? "positive" : "negative";
    return `
      <div class="gallery-delta ${cls}" data-role="delta">
        ${sign}${delta.toFixed(2)} dB
      </div>
    `;
  }

  function buildBStats(f, modeSpec) {
    if (modeSpec.isoKey) {
      const entry = f[modeSpec.isoKey];
      const kb = (entry.bytes / 1024).toFixed(1);
      return `<strong>${kb} KB</strong> &middot; ${entry.psnr_db.toFixed(2)} dB`;
    }
    // Source mode: show the PNG-lossless size (the honest "source bytes"
    // reference) and a "lossless" indicator instead of a PSNR number.
    const kb = (f.source_bytes / 1024).toFixed(1);
    return `<strong>${kb} KB</strong> &middot; lossless`;
  }

  /* ─────────────────────────── gallery toggle ─────────────────────── */

  function initToggle() {
    const buttons = document.querySelectorAll(".gallery-toggle .toggle-btn");
    buttons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const mode = btn.dataset.mode;
        if (!mode || mode === currentMode) return;
        setMode(mode, buttons);
      });
    });
  }

  function setMode(mode, buttons) {
    if (!MODES[mode]) return;
    currentMode = mode;
    const modeSpec = MODES[mode];

    // Update toggle button states
    if (buttons) {
      buttons.forEach((b) => {
        const active = b.dataset.mode === mode;
        b.classList.toggle("active", active);
        b.setAttribute("aria-checked", active ? "true" : "false");
      });
    }

    // Look up each card by fixture name and update the B-side
    const fixtureByName = new Map(allFixtures.map((f) => [f.name, f]));
    const cards = document.querySelectorAll(".gallery-card");

    cards.forEach((card) => {
      const name = card.dataset.fixture;
      const f = fixtureByName.get(name);
      if (!f) return;

      // B image
      const bImg = card.querySelector('[data-role="b-img"]');
      if (bImg) {
        bImg.src = f.assets[modeSpec.assetKey];
        bImg.alt = modeSpec.altText(f);
      }

      // B slider label
      const bLabel = card.querySelector('[data-role="b-label"]');
      if (bLabel) bLabel.textContent = modeSpec.sliderLabel;

      // Delta badge
      const delta = card.querySelector('[data-role="delta"]');
      if (delta) {
        delta.classList.remove("positive", "negative");
        if (modeSpec.hasDelta) {
          const d = f[modeSpec.deltaKey];
          const sign = d >= 0 ? "+" : "";
          delta.classList.add(d >= 0 ? "positive" : "negative");
          delta.textContent = `${sign}${d.toFixed(2)} dB`;
        } else {
          // Source mode: show Weft's absolute PSNR, no sign/color
          delta.textContent = `${f.weft.psnr_db.toFixed(2)} dB`;
        }
      }

      // B stat label + numbers
      const bStatLabel = card.querySelector('[data-role="b-stat-label"]');
      if (bStatLabel) bStatLabel.textContent = modeSpec.statLabel(f);
      const bStatNum = card.querySelector('[data-role="b-stat-num"]');
      if (bStatNum) bStatNum.innerHTML = buildBStats(f, modeSpec);
    });
  }

  /* ─────────────────────────── benchmark table ───────────────────── */

  function renderBenchTable(data) {
    const tbody = document.getElementById("bench-tbody");
    if (!tbody) return;

    const rows = data.fixtures.map((f) => {
      const weftKb = (f.weft.bytes / 1024).toFixed(1);
      const weftDb = f.weft.psnr_db.toFixed(2);
      const jpegDb = f.jpeg_iso.psnr_db.toFixed(2);
      const delta = f.delta_db;
      const deltaSign = delta >= 0 ? "+" : "";
      const deltaClass = delta >= 0 ? "delta-positive" : "delta-negative";
      return `
        <tr>
          <td class="col-name">${escapeHtml(f.title)}</td>
          <td class="col-right">${weftKb}</td>
          <td class="col-right">${weftDb}</td>
          <td class="col-right">${jpegDb}</td>
          <td class="col-right ${deltaClass}">${deltaSign}${delta.toFixed(2)}</td>
          <td class="col-name">${escapeHtml(f.variant)}</td>
        </tr>
      `;
    });
    tbody.innerHTML = rows.join("");

    // Aggregate row
    const totalWeftKb = data.totals_kb.weft.toFixed(1);
    document.getElementById("tfoot-bytes").textContent = totalWeftKb;
    document.getElementById("tfoot-weft-db").textContent = data.avg_weft_psnr_db.toFixed(2);
    document.getElementById("tfoot-jpeg-db").textContent = data.avg_jpeg_psnr_iso_db.toFixed(2);
    const tfootDelta = document.getElementById("tfoot-delta");
    const sign = data.avg_delta_db_vs_jpeg >= 0 ? "+" : "";
    tfootDelta.textContent = `${sign}${data.avg_delta_db_vs_jpeg.toFixed(2)}`;
    tfootDelta.className =
      "col-right " + (data.avg_delta_db_vs_jpeg >= 0 ? "delta-positive" : "delta-negative");
  }

  /* ─────────────────────────── slider interaction ────────────────── */

  function initSliders() {
    const sliders = document.querySelectorAll(".compare-slider");
    sliders.forEach(attachSlider);
  }

  function attachSlider(el) {
    const topPane = el.querySelector(".compare-b");
    const handle = el.querySelector(".compare-handle");
    if (!topPane || !handle) return;

    let dragging = false;
    let animFrame = null;
    let pendingPct = 50;

    const apply = () => {
      animFrame = null;
      topPane.style.clipPath = `inset(0 0 0 ${pendingPct}%)`;
      handle.style.left = `${pendingPct}%`;
    };

    const setFromClientX = (clientX) => {
      const rect = el.getBoundingClientRect();
      if (rect.width === 0) return;
      const x = Math.max(0, Math.min(rect.width, clientX - rect.left));
      pendingPct = (x / rect.width) * 100;
      if (animFrame === null) {
        animFrame = requestAnimationFrame(apply);
      }
    };

    // Mouse
    el.addEventListener("mousedown", (e) => {
      dragging = true;
      setFromClientX(e.clientX);
      e.preventDefault();
    });
    window.addEventListener("mousemove", (e) => {
      if (dragging) setFromClientX(e.clientX);
    });
    window.addEventListener("mouseup", () => {
      dragging = false;
    });

    // Touch
    el.addEventListener(
      "touchstart",
      (e) => {
        dragging = true;
        if (e.touches.length > 0) setFromClientX(e.touches[0].clientX);
      },
      { passive: true }
    );
    el.addEventListener(
      "touchmove",
      (e) => {
        if (dragging && e.touches.length > 0) {
          setFromClientX(e.touches[0].clientX);
        }
      },
      { passive: true }
    );
    el.addEventListener("touchend", () => {
      dragging = false;
    });
    el.addEventListener("touchcancel", () => {
      dragging = false;
    });

    // Click-to-jump (non-drag single click) works automatically because
    // the mousedown handler above sets the position from clientX.

    // Keyboard accessibility: arrow keys nudge by 2%.
    el.tabIndex = 0;
    el.setAttribute("role", "img");
    el.setAttribute("aria-label", "Image comparison slider. Drag or use arrow keys to reveal each side.");
    el.addEventListener("keydown", (e) => {
      let step = 0;
      if (e.key === "ArrowLeft") step = -2;
      else if (e.key === "ArrowRight") step = 2;
      else if (e.key === "Home") step = -100;
      else if (e.key === "End") step = 100;
      else return;
      pendingPct = Math.max(0, Math.min(100, pendingPct + step));
      if (animFrame === null) {
        animFrame = requestAnimationFrame(apply);
      }
      e.preventDefault();
    });
  }

  /* ─────────────────────────── utilities ─────────────────────────── */

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  /* ─────────────────────────── init ──────────────────────────────── */

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
