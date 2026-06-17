document.addEventListener("DOMContentLoaded", () => {
  const labels = new Map();

  const collectLabels = (root) => {
    const text = root.textContent;
    const displayMath = /\\begin\{(?:equation|align|alignat|gather)\}|\\\[|\$\$/g;
    let equationNumber = 1;
    let match;

    while ((match = displayMath.exec(text)) !== null) {
      const startToken = match[0];
      const endToken = startToken === "\\[" ? "\\]" : startToken === "$$" ? "$$" : startToken.replace("\\begin", "\\end");
      const endIndex = text.indexOf(endToken, displayMath.lastIndex);
      if (endIndex === -1) break;

      const math = text.slice(displayMath.lastIndex, endIndex);
      for (const labelMatch of math.matchAll(/\\label\{([^}]+)\}/g)) {
        const label = labelMatch[1];
        if (!labels.has(label)) {
          labels.set(label, equationNumber);
        }
      }
      equationNumber += 1;
      displayMath.lastIndex = endIndex + endToken.length;
    }
  };

  collectLabels(document.body);

  const renderReference = (label, eqref) => {
    const number = labels.get(label) || "?";
    return eqref ? `(${number})` : `${number}`;
  };

  const replaceTextReferences = (root) => {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
      acceptNode: (node) => {
        const parent = node.parentElement;
        if (!parent || parent.closest("code, pre, script, style, textarea, .katex")) {
          return NodeFilter.FILTER_REJECT;
        }
        return /\\(?:eq)?ref\{[^}]+\}/.test(node.nodeValue)
          ? NodeFilter.FILTER_ACCEPT
          : NodeFilter.FILTER_REJECT;
      },
    });

    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const node of nodes) {
      node.nodeValue = node.nodeValue
        .replace(/\\eqref\{([^}]+)\}/g, (_match, label) => renderReference(label, true))
        .replace(/\\ref\{([^}]+)\}/g, (_match, label) => renderReference(label, false));
    }
  };

  replaceTextReferences(document.body);

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
      .replace(/\\DeclareMathOperator\*?\{\\argmin\}\{arg\\,min\}/g, "")
      .replace(/\\argmin/g, "\\operatorname*{arg\\,min}")
      .replace(/\\label\{[^}]+\}/g, "")
      .replace(/\\eqref\{([^}]+)\}/g, (_match, label) => `\\text{${renderReference(label, true)}}`)
      .replace(/\\ref\{([^}]+)\}/g, (_match, label) => `\\text{${renderReference(label, false)}}`),
    throwOnError: false,
  });
});
