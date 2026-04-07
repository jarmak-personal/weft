// Weft site script. No framework, no build step.
//
// Responsibilities:
//  1. Fetch assets/bench_data.json produced by generate_site_assets.py
//  2. Populate hero stats + benchmark table from the JSON
//  3. Render the gallery grid (one card per fixture) with interactive
//     image-comparison sliders
//  4. Wire the slider drag + touch + keyboard interaction
//
// The JSON schema is defined by scripts/generate_site_assets.py.

(function () {
  "use strict";

  const DATA_URL = "assets/bench_data.json";

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

    populateHeroStats(data);
    renderGallery(data.fixtures);
    renderBenchTable(data);
    initSliders();
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

    // Sort: biggest positive wins first (most visually striking), then
    // losses at the end for the honest story.
    const sorted = fixtures.slice().sort((a, b) => b.delta_db - a.delta_db);

    const html = sorted.map(buildGalleryCard).join("");
    grid.innerHTML = html;
  }

  function buildGalleryCard(f) {
    const deltaSign = f.delta_db >= 0 ? "+" : "";
    const deltaClass = f.delta_db >= 0 ? "positive" : "negative";
    const weftKb = (f.weft.bytes / 1024).toFixed(1);
    const jpegKb = (f.jpeg_iso.bytes / 1024).toFixed(1);

    return `
      <article class="gallery-card">
        <div class="gallery-header">
          <div class="gallery-meta">
            <h3 class="gallery-title">${escapeHtml(f.title)}</h3>
            <span class="gallery-variant">variant: ${escapeHtml(f.variant)}</span>
          </div>
          <div class="gallery-delta ${deltaClass}">
            ${deltaSign}${f.delta_db.toFixed(2)} dB
          </div>
        </div>
        <p class="gallery-note">${escapeHtml(f.note)}</p>

        <div class="compare-slider" data-name="${escapeHtml(f.name)}">
          <div class="compare-a">
            <img src="${escapeHtml(f.assets.weft)}" alt="Weft decoded: ${escapeHtml(f.title)}" loading="lazy">
          </div>
          <div class="compare-b">
            <img src="${escapeHtml(f.assets.jpeg)}" alt="JPEG at iso-bytes: ${escapeHtml(f.title)}" loading="lazy">
          </div>
          <div class="compare-labels">
            <span class="compare-label-a">◀ Weft</span>
            <span class="compare-label-b">JPEG ▶</span>
          </div>
          <div class="compare-handle" aria-hidden="true"></div>
        </div>

        <div class="gallery-stats">
          <div class="gallery-stat">
            <div class="gallery-stat-label">Weft</div>
            <div class="gallery-stat-num"><strong>${weftKb} KB</strong> &middot; ${f.weft.psnr_db.toFixed(2)} dB</div>
          </div>
          <div class="gallery-stat">
            <div class="gallery-stat-label">JPEG (iso-bytes, q${f.jpeg_iso.quality})</div>
            <div class="gallery-stat-num"><strong>${jpegKb} KB</strong> &middot; ${f.jpeg_iso.psnr_db.toFixed(2)} dB</div>
          </div>
        </div>
      </article>
    `;
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
