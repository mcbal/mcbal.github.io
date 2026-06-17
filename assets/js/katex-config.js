document.addEventListener("DOMContentLoaded", () => {
  const labels = new Map();
  let nextLabelNumber = 1;

  for (const match of document.body.textContent.matchAll(/\\label\{([^}]+)\}/g)) {
    const label = match[1];
    if (!labels.has(label)) {
      labels.set(label, nextLabelNumber);
      nextLabelNumber += 1;
    }
  }

  window.renderMathInElement(document.body, {
    delimiters: [
      {left: "$$", right: "$$", display: true},
      {left: "$", right: "$", display: false},
      {left: "\\(", right: "\\)", display: false},
      {left: "\\[", right: "\\]", display: true},
      {left: "\\begin{equation}", right: "\\end{equation}", display: true},
      {left: "\\begin{align}", right: "\\end{align}", display: true},
      {left: "\\begin{alignat}", right: "\\end{alignat}", display: true},
      {left: "\\begin{gather}", right: "\\end{gather}", display: true},
    ],
    preProcess: (math) => math
      .replace(/\\label\{[^}]+\}/g, "")
      .replace(/\\eqref\{([^}]+)\}/g, (_match, label) => `\\text{(${labels.get(label) || "?"})}`)
      .replace(/\\ref\{([^}]+)\}/g, (_match, label) => `\\text{${labels.get(label) || "?"}}`),
    throwOnError: false,
  });
});
