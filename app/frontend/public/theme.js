/* global document, window */

(() => {
  const root = document.documentElement;
  let storedTheme;
  try {
    const stored = window.localStorage.getItem("web-theme");
    if (stored === "light" || stored === "dark") storedTheme = stored;
  } catch {
    storedTheme = undefined;
  }
  let systemDark = false;
  try {
    systemDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  } catch {
    systemDark = false;
  }
  const theme = storedTheme ?? (systemDark ? "dark" : "light");
  root.classList.toggle("dark", theme === "dark");
  root.dataset.theme = theme;
  root.style.colorScheme = theme;
})();
