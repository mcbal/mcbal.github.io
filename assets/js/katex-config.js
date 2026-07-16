document.addEventListener("DOMContentLoaded", () => {
  const labels = new Map();
  const equationLabels = new Map();

  const equationId = (label) => `equation-${label}`;

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
          equationLabels.set(label, equationNumber);
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
      const fragment = document.createDocumentFragment();
      const reference = /\\(eq)?ref\{([^}]+)\}/g;
      let lastIndex = 0;
      let match;

      while ((match = reference.exec(node.nodeValue)) !== null) {
        fragment.append(node.nodeValue.slice(lastIndex, match.index));

        const [, eqref, label] = match;
        const number = renderReference(label, Boolean(eqref));
        if (labels.has(label)) {
          const link = document.createElement("a");
          link.href = `#${equationId(label)}`;
          link.className = "equation-reference";
          link.textContent = number;
          link.setAttribute("aria-label", `Equation ${labels.get(label)}`);
          fragment.append(link);
        } else {
          fragment.append(number);
        }

        lastIndex = reference.lastIndex;
      }

      fragment.append(node.nodeValue.slice(lastIndex));
      node.replaceWith(fragment);
    }
  };

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
      .replace(/\\eqref\{([^}]+)\}/g, (_match, label) => labels.has(label)
        ? `\\href{#${equationId(label)}}{\\text{${renderReference(label, true)}}}`
        : `\\text{${renderReference(label, true)}}`)
      .replace(/\\ref\{([^}]+)\}/g, (_match, label) => labels.has(label)
        ? `\\href{#${equationId(label)}}{\\text{${renderReference(label, false)}}}`
        : `\\text{${renderReference(label, false)}}`),
    trust: (context) => context.command === "\\href" && context.url.startsWith("#equation-"),
    throwOnError: false,
  });

  replaceTextReferences(document.body);

  const displayMath = document.querySelectorAll(".katex-display");
  for (const [label, equationNumber] of equationLabels) {
    const equation = displayMath[equationNumber - 1];
    if (equation) equation.id = equationId(label);
  }

  const updateMathOverflow = (element) => {
    const renderedMath = element.querySelector(".katex-html");
    if (!renderedMath) return;

    // Measure KaTeX's default centered layout. In the scrolling layout the tag
    // is made static, which would otherwise make subsequent measurements lie.
    element.classList.remove("is-overflowing");

    const containerBounds = element.getBoundingClientRect();
    const tag = renderedMath.querySelector(".tag");
    const tagBounds = tag?.getBoundingClientRect();
    const equationBounds = Array.from(renderedMath.querySelectorAll(".base"))
      .map((base) => base.getBoundingClientRect());
    const safetyMargin = 12;

    const touchesTag = tagBounds && equationBounds.some((bounds) => (
      bounds.right + safetyMargin > tagBounds.left
        && bounds.left < tagBounds.right + safetyMargin
    ));
    const approachesEdge = equationBounds.some((bounds) => (
      bounds.left < containerBounds.left + safetyMargin
        || bounds.right > containerBounds.right - safetyMargin
    ));

    element.classList.toggle(
      "is-overflowing",
      approachesEdge || touchesTag,
    );
  };

  displayMath.forEach(updateMathOverflow);

  if ("ResizeObserver" in window) {
    const mathResizeObserver = new ResizeObserver((entries) => {
      entries.forEach(({target}) => updateMathOverflow(target));
    });
    displayMath.forEach((element) => mathResizeObserver.observe(element));
  }

  document.fonts?.ready.then(() => displayMath.forEach(updateMathOverflow));
});
