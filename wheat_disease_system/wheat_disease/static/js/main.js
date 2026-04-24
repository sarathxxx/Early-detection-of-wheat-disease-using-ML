// WheatAI – main.js

// Nav toggle (mobile)
function toggleNav() {
  document.querySelector('.nav-links').classList.toggle('open');
}

// Auto-dismiss flash messages
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.flash').forEach(el => {
    setTimeout(() => el.style.opacity = '0', 4000);
    setTimeout(() => el.remove(), 4400);
  });

  // Animate confidence bars on load
  document.querySelectorAll('.conf-fill').forEach(el => {
    const w = el.style.width;
    el.style.width = '0';
    requestAnimationFrame(() => {
      requestAnimationFrame(() => { el.style.transition = 'width .5s ease'; el.style.width = w; });
    });
  });

  // Animate stat numbers (count up)
  document.querySelectorAll('.stat-num').forEach(el => {
    const raw = el.textContent.trim();
    const num = parseInt(raw);
    if (!isNaN(num) && num > 0) {
      let current = 0;
      const step = Math.ceil(num / 24);
      const timer = setInterval(() => {
        current = Math.min(current + step, num);
        el.textContent = current;
        if (current >= num) { el.textContent = raw; clearInterval(timer); }
      }, 30);
    }
  });
});
