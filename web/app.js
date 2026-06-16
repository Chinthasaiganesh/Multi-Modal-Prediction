const form = document.querySelector("#predict-form");
const clearButton = document.querySelector("#clear-button");
const productNameInput = document.querySelector("#product_name");
const descriptionInput = document.querySelector("#description");
const detailsInput = document.querySelector("#product_details");
const imageInput = document.querySelector("#image_link");
const imagePreview = document.querySelector("#image-preview");
const statusEl = document.querySelector("#status");
const priceEl = document.querySelector("#price");
const submitButton = form.querySelector("button[type='submit']");

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", isError);
}

function buildCatalogContent() {
  return [
    `Item Name: ${productNameInput.value.trim()}`,
    `Product Description: ${descriptionInput.value.trim()}`,
    detailsInput.value.trim(),
  ]
    .filter(Boolean)
    .join("\n");
}

function updatePreview() {
  const url = imageInput.value.trim();
  imagePreview.src = url || "";
  imagePreview.classList.toggle("empty", !url);
}

imageInput.addEventListener("input", updatePreview);
imagePreview.addEventListener("error", () => {
  setStatus("Image could not be loaded", true);
});
imagePreview.addEventListener("load", () => {
  if (imageInput.value.trim()) {
    setStatus("Ready");
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!imageInput.value.trim()) {
    setStatus("Product image is required", true);
    return;
  }
  if (!descriptionInput.value.trim()) {
    setStatus("Description is required", true);
    return;
  }

  submitButton.disabled = true;
  setStatus("Predicting");

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        catalog_content: buildCatalogContent(),
        description: descriptionInput.value,
        image_link: imageInput.value,
      }),
    });
    const result = await response.json();
    if (!response.ok || !result.ok) {
      throw new Error(result.error || "Prediction failed.");
    }
    priceEl.textContent = Number(result.price).toLocaleString(undefined, {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    });
    setStatus("Prediction ready");
  } catch (error) {
    priceEl.textContent = "--";
    setStatus(error.message, true);
  } finally {
    submitButton.disabled = false;
  }
});

clearButton.addEventListener("click", () => {
  productNameInput.value = "";
  descriptionInput.value = "";
  detailsInput.value = "";
  imageInput.value = "";
  updatePreview();
  priceEl.textContent = "--";
  setStatus("Ready");
  productNameInput.focus();
});

updatePreview();
