/**
 * Collapse appendices visually while leaving the complete article in the
 * server-rendered HTML for crawlers, reader modes, and no-JavaScript clients.
 */
const collapseAppendices = (article) => {
  const children = Array.from(article.children);
  const appendixIndex = children.findIndex(
    (element) =>
      /^H[1-6]$/.test(element.tagName) &&
      (element.id === "appendices" || /^appendices(?:\s|:|$)/i.test(element.textContent.trim())),
  );

  let headings;
  if (appendixIndex >= 0) {
    const appendixLevel = Number(children[appendixIndex].tagName.slice(1));
    const appendixChildren = children.slice(appendixIndex + 1);
    const appendixEnd = appendixChildren.findIndex(
      (element) =>
        /^H[1-6]$/.test(element.tagName) &&
        Number(element.tagName.slice(1)) <= appendixLevel,
    );
    const section = appendixEnd < 0 ? appendixChildren : appendixChildren.slice(0, appendixEnd);
    headings = section.filter((element) => {
      if (!/^H[1-6]$/.test(element.tagName)) return false;
      return Number(element.tagName.slice(1)) === appendixLevel + 1;
    });
  } else {
    // Also support articles that use top-level "Appendix A", "Appendix B", …
    // headings without an "Appendices" container heading.
    headings = children.filter(
      (element) =>
        /^H[1-6]$/.test(element.tagName) &&
        /^appendix(?:\s|:|$)/i.test(element.textContent.trim()),
    );
  }

  headings.forEach((heading) => {
    if (heading.closest("details")) return;

    const level = Number(heading.tagName.slice(1));
    const details = document.createElement("details");
    details.className = "appendix-details";

    const summary = document.createElement("summary");
    summary.className = "appendix-summary";
    summary.textContent = heading.textContent.trim();
    details.append(summary);

    heading.before(details);
    heading.classList.add("appendix-heading");
    details.append(heading);

    // Raw TeX environments are emitted by Hugo as text nodes rather than
    // elements. Move every node so KaTeX renders display math inside the panel.
    let sibling = details.nextSibling;
    while (sibling) {
      // A Markdown horizontal rule immediately before the next top-level
      // heading belongs to that heading, not to the preceding appendix.
      if (sibling.nodeType === Node.ELEMENT_NODE && sibling.tagName === "HR") {
        let nextSignificant = sibling.nextSibling;
        while (nextSignificant?.nodeType === Node.TEXT_NODE && !nextSignificant.textContent.trim()) {
          nextSignificant = nextSignificant.nextSibling;
        }
        const nextLevel =
          nextSignificant?.nodeType === Node.ELEMENT_NODE && /^H[1-6]$/.test(nextSignificant.tagName)
            ? Number(nextSignificant.tagName.slice(1))
            : Infinity;
        if (nextLevel <= level) break;
      }

      const siblingLevel =
        sibling.nodeType === Node.ELEMENT_NODE && /^H[1-6]$/.test(sibling.tagName)
        ? Number(sibling.tagName.slice(1))
        : Infinity;
      if (siblingLevel <= level) break;

      const next = sibling.nextSibling;
      details.append(sibling);
      sibling = next;
    }
  });

  const revealFragment = () => {
    if (!location.hash) return;
    const target = document.getElementById(decodeURIComponent(location.hash.slice(1)));
    target?.closest(".appendix-details")?.setAttribute("open", "");
  };

  revealFragment();
};

const enhanceArticles = () => {
  document.querySelectorAll(".article-content").forEach(collapseAppendices);
};

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", enhanceArticles);
} else {
  enhanceArticles();
}

window.addEventListener("hashchange", enhanceArticles);

// Hugo Blox can replace page content without triggering DOMContentLoaded.
// Only react when a new article arrives, not to math rendering within it.
new MutationObserver((records) => {
  const articleAdded = records.some((record) =>
    Array.from(record.addedNodes).some(
      (node) =>
        node.nodeType === Node.ELEMENT_NODE &&
        (node.matches(".article-content") || node.querySelector(".article-content")),
    ),
  );
  if (articleAdded) enhanceArticles();
}).observe(document.documentElement, {
  childList: true,
  subtree: true,
});
