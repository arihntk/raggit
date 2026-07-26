(function () {
  const menuToggle = document.getElementById('menu-toggle');
  const sidebar = document.getElementById('sidebar');
  const navLinks = document.querySelectorAll('.nav-link');
  const themeToggles = document.querySelectorAll('.theme-toggle');

  // Mobile menu toggle
  if (menuToggle && sidebar) {
    menuToggle.addEventListener('click', function () {
      sidebar.classList.toggle('open');
    });

    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', function (event) {
      const isClickInside = sidebar.contains(event.target) || menuToggle.contains(event.target);
      if (!isClickInside && sidebar.classList.contains('open')) {
        sidebar.classList.remove('open');
      }
    });
  }

  // Theme toggle
  function getMermaidTheme() {
    return document.documentElement.getAttribute('data-theme') === 'light' ? 'neutral' : 'dark';
  }

  async function renderArchitectureDiagram() {
    const container = document.getElementById('architecture-diagram');
    if (!container || typeof mermaid === 'undefined') return;

    const sourceEl = container.querySelector('.mermaid-source');
    if (!sourceEl) return;

    const source = sourceEl.textContent.trim();
    const id = 'mermaid-arch-' + Math.random().toString(36).substr(2, 9);

    mermaid.initialize({
      startOnLoad: false,
      theme: getMermaidTheme(),
      flowchart: {
        useMaxWidth: true,
        htmlLabels: true,
        curve: 'basis'
      }
    });

    try {
      const result = await mermaid.render(id, source);
      container.innerHTML = result.svg;
    } catch (err) {
      console.error('Failed to render architecture diagram:', err);
    }
  }

  function toggleTheme() {
    const html = document.documentElement;
    const isLight = html.getAttribute('data-theme') === 'light';
    if (isLight) {
      html.removeAttribute('data-theme');
      localStorage.removeItem('raggit-theme');
    } else {
      html.setAttribute('data-theme', 'light');
      localStorage.setItem('raggit-theme', 'light');
    }
    renderArchitectureDiagram();
  }

  themeToggles.forEach(function (button) {
    button.addEventListener('click', toggleTheme);
  });

  window.addEventListener('load', renderArchitectureDiagram);

  // Update active nav link on scroll
  function updateActiveLink() {
    const sections = Array.from(document.querySelectorAll('section[id]'));
    const scrollPosition = window.scrollY + 120;

    let current = sections[0];
    for (const section of sections) {
      if (section.offsetTop <= scrollPosition) {
        current = section;
      }
    }

    navLinks.forEach(function (link) {
      link.classList.remove('active');
      if (link.getAttribute('href') === '#' + current.id) {
        link.classList.add('active');
      }
    });
  }

  window.addEventListener('scroll', updateActiveLink, { passive: true });
  window.addEventListener('load', updateActiveLink);
})();
