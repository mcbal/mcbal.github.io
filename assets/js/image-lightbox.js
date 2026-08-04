const createImageLightbox = () => {
  if (document.querySelector(".image-lightbox")) return;

  const dialog = document.createElement("dialog");
  dialog.className = "image-lightbox";
  dialog.setAttribute("aria-label", "Enlarged image");
  dialog.innerHTML = `
    <button class="image-lightbox-close" type="button" aria-label="Close enlarged image">
      <span aria-hidden="true">&times;</span>
    </button>
    <figure class="image-lightbox-content">
      <img class="image-lightbox-image" alt="">
      <figcaption class="image-lightbox-caption"></figcaption>
    </figure>
  `;

  const image = dialog.querySelector(".image-lightbox-image");
  const caption = dialog.querySelector(".image-lightbox-caption");
  const closeButton = dialog.querySelector(".image-lightbox-close");
  let trigger = null;

  const close = () => dialog.close();

  closeButton.addEventListener("click", close);
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) close();
  });
  dialog.addEventListener("close", () => {
    document.documentElement.classList.remove("image-lightbox-open");
    image.removeAttribute("src");
    trigger?.focus({ preventScroll: true });
    trigger = null;
  });

  document.addEventListener("click", (event) => {
    const link = event.target.closest?.("[data-lightbox]");
    if (!link || event.defaultPrevented || event.button !== 0) return;
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    if (typeof dialog.showModal !== "function") return;

    event.preventDefault();
    trigger = link;
    const preview = link.querySelector("img");
    image.src = link.href;
    image.alt = preview?.alt || "";
    caption.textContent = link.dataset.lightboxCaption || "";
    caption.hidden = !caption.textContent;
    document.documentElement.classList.add("image-lightbox-open");
    dialog.showModal();
    closeButton.focus();
  });

  document.body.append(dialog);
};

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", createImageLightbox, { once: true });
} else {
  createImageLightbox();
}
