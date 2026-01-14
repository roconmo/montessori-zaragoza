document.querySelectorAll('[data-carousel]').forEach(carousel => {
  const track = carousel.querySelector('[data-track]');
  const slides = Array.from(track.children);
  const prev = carousel.querySelector('.prev');
  const next = carousel.querySelector('.next');
  const dots = Array.from(carousel.querySelectorAll('.dot'));
  let index = 0;

  function update(){
    track.style.transform = `translateX(-${index * 100}%)`;
    dots.forEach((d,i)=> d.classList.toggle('is-active', i === index));
  }

  prev.addEventListener('click', () => {
    index = (index - 1 + slides.length) % slides.length;
    update();
  });

  next.addEventListener('click', () => {
    index = (index + 1) % slides.length;
    update();
  });

  dots.forEach((dot,i) => {
    dot.addEventListener('click', () => {
      index = i;
      update();
    });
  });

  update();
});

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-carousel]").forEach((carousel) => {

    const track = carousel.querySelector("[data-track]");
    const dotsWrap = carousel.querySelector("[data-dots]");
    const prev = carousel.querySelector(".prev");
    const next = carousel.querySelector(".next");

    if (!track) return;

    // 🔹 Leer configuración
    const base = carousel.dataset.base;     // ej: img/img
    const from = parseInt(carousel.dataset.from, 10);
    const to   = parseInt(carousel.dataset.to, 10);
    const ext  = carousel.dataset.ext || ".jpg";
    const alt  = carousel.dataset.alt || "";

    if (!base || isNaN(from) || isNaN(to)) {
      console.warn("Carrusel mal configurado", carousel);
      return;
    }

    // 🔹 Generar slides automáticamente
    track.innerHTML = "";
    for (let i = from; i <= to; i++) {
      const slide = document.createElement("div");
      slide.className = "carousel-slide";
      slide.innerHTML = `
        <img src="${base}${i}${ext}" alt="${alt}" loading="lazy">
      `;
      track.appendChild(slide);
    }

    const slides = Array.from(track.children);
    if (slides.length <= 1) return;

    // 🔹 Crear dots
    let dots = [];
    if (dotsWrap) {
      dotsWrap.innerHTML = "";
      slides.forEach((_, i) => {
        const d = document.createElement("button");
        d.type = "button";
        d.className = "dot";
        d.setAttribute("aria-label", `Ir a la foto ${i + 1}`);
        d.addEventListener("click", () => {
          index = i;
          update();
        });
        dotsWrap.appendChild(d);
      });
      dots = Array.from(dotsWrap.children);
    }

    let index = 0;

    function update() {
      track.style.transform = `translateX(-${index * 100}%)`;
      dots.forEach((d, i) => d.classList.toggle("is-active", i === index));
    }

    function goPrev() {
      index = (index - 1 + slides.length) % slides.length;
      update();
    }

    function goNext() {
      index = (index + 1) % slides.length;
      update();
    }

    if (prev) prev.addEventListener("click", goPrev);
    if (next) next.addEventListener("click", goNext);

    // 🔹 Swipe móvil
    let startX = 0;
    track.addEventListener("touchstart", e => startX = e.touches[0].clientX, { passive: true });
    track.addEventListener("touchend", e => {
      const diff = e.changedTouches[0].clientX - startX;
      if (Math.abs(diff) > 40) diff < 0 ? goNext() : goPrev();
    }, { passive: true });

    update();
  });
});

