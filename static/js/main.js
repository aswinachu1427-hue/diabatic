/**
 * main.js - Core UI Controller & Animation Orchestrator
 * Handles splash sequence, hero word stagger, KPI count-up, parallax, and toasts.
 */

document.addEventListener('DOMContentLoaded', () => {
  initSplashAnimation();
  initHeroStagger();
  initKpiCounters();
  initAmbientParallax();
});

/* 1. Splash Animation (2.2s total sequence) */
function initSplashAnimation() {
  const splash = document.getElementById('splash-screen');
  if (!splash) return;

  // Check if prefers-reduced-motion is requested
  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  if (prefersReduced) {
    splash.style.display = 'none';
    return;
  }

  // Dismiss splash after 2.1 seconds with slide-up transition
  setTimeout(() => {
    splash.classList.add('splash-hidden');
    setTimeout(() => {
      splash.style.display = 'none';
      triggerHeroEntry();
    }, 600);
  }, 2100);
}

/* 2. Hero Reveal: Staggered words & KPI pop-in */
function initHeroStagger() {
  const headline = document.getElementById('hero-headline');
  if (!headline) return;

  const words = headline.querySelectorAll('.word');
  words.forEach((word, idx) => {
    word.style.display = 'inline-block';
    word.style.opacity = '0';
    word.style.transform = 'translateY(24px)';
    word.style.transition = `all 0.5s cubic-bezier(0.16, 1, 0.3, 1) ${idx * 80}ms`;
  });
}

function triggerHeroEntry() {
  const words = document.querySelectorAll('#hero-headline .word');
  words.forEach(word => {
    word.style.opacity = '1';
    word.style.transform = 'translateY(0)';
  });

  const authCard = document.getElementById('authCard');
  if (authCard) {
    authCard.style.opacity = '0';
    authCard.style.transform = 'translateX(30px)';
    authCard.style.transition = 'all 0.7s cubic-bezier(0.16, 1, 0.3, 1) 0.2s';
    requestAnimationFrame(() => {
      authCard.style.opacity = '1';
      authCard.style.transform = 'translateX(0)';
    });
  }
}

/* 3. KPI Numbers Count-Up Animation */
function initKpiCounters() {
  const kpiElements = document.querySelectorAll('.kpi-number');
  if (kpiElements.length === 0) return;

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        animateCounter(entry.target);
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.2 });

  kpiElements.forEach(el => observer.observe(el));
}

function animateCounter(el) {
  const target = parseFloat(el.getAttribute('data-target')) || 0;
  const format = el.getAttribute('data-format') || 'number';
  const decimals = parseInt(el.getAttribute('data-decimals')) || 0;
  const duration = 1800;
  const startTime = performance.now();

  function update(now) {
    const elapsed = now - startTime;
    const progress = Math.min(elapsed / duration, 1);
    // Ease-out cubic
    const easeProgress = 1 - Math.pow(1 - progress, 3);
    const current = progress * target;

    if (format === 'percent') {
      el.textContent = current.toFixed(decimals) + '%';
    } else if (format === 'comma') {
      el.textContent = Math.round(current).toLocaleString();
    } else {
      el.textContent = Math.round(current).toString();
    }

    if (progress < 1) {
      requestAnimationFrame(update);
    } else {
      if (format === 'percent') el.textContent = target.toFixed(decimals) + '%';
      else if (format === 'comma') el.textContent = Math.round(target).toLocaleString();
      else el.textContent = Math.round(target).toString();
    }
  }
  requestAnimationFrame(update);
}

/* 4. Ambient 3D Motion & Mouse Parallax */
function initAmbientParallax() {
  const orbs = document.querySelectorAll('.ambient-orb');
  if (orbs.length === 0) return;

  let mouseX = 0, mouseY = 0;
  let currentX = 0, currentY = 0;

  window.addEventListener('mousemove', (e) => {
    mouseX = (e.clientX / window.innerWidth - 0.5) * 35;
    mouseY = (e.clientY / window.innerHeight - 0.5) * 35;
  }, { passive: true });

  function renderParallax() {
    currentX += (mouseX - currentX) * 0.05;
    currentY += (mouseY - currentY) * 0.05;

    orbs.forEach((orb, i) => {
      const depth = (i + 1) * 0.7;
      orb.style.transform = `translate(${currentX * depth}px, ${currentY * depth}px)`;
    });

    requestAnimationFrame(renderParallax);
  }
  requestAnimationFrame(renderParallax);
}

/* 5. Dynamic Toast Alert Controller */
function showToast(message, type = 'info') {
  const container = document.getElementById('toast-container');
  if (!container) return;

  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;

  const iconName = type === 'success' ? 'check-circle' :
                   type === 'danger' ? 'alert-triangle' :
                   type === 'warning' ? 'alert-circle' : 'info';

  toast.innerHTML = `
    <i data-lucide="${iconName}" style="width: 20px; height: 20px; flex-shrink: 0; color: var(--${type === 'danger' ? 'danger' : type === 'success' ? 'success' : type === 'warning' ? 'warning' : 'primary'});"></i>
    <span class="toast-message">${message}</span>
    <button type="button" onclick="this.parentElement.remove()" style="background:none; border:none; color: var(--textlight); cursor:pointer; padding:2px;">
      <i data-lucide="x" style="width: 16px; height: 16px;"></i>
    </button>
  `;

  container.appendChild(toast);
  lucide.createIcons({ root: toast });

  setTimeout(() => {
    toast.style.transition = 'all 0.3s ease';
    toast.style.opacity = '0';
    toast.style.transform = 'translateX(100%)';
    setTimeout(() => toast.remove(), 300);
  }, 4500);
}
