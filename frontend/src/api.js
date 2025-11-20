// src/api.js
export async function inferImage(blobOrFile) {
  const url = (import.meta.env.VITE_BACKEND_URL || "http://localhost:8000") + "/v1/infer";
  const fd = new FormData();
  // backend expects field name "image" (multipart)
  fd.append("image", blobOrFile, "capture.jpg");

  const res = await fetch(url, {
    method: "POST",
    body: fd,
  });

  if (!res.ok) {
    const text = await res.text();
    let parsed;
    try { parsed = JSON.parse(text); } catch (e) { parsed = text; }
    throw new Error(`Server returned ${res.status}: ${JSON.stringify(parsed)}`);
  }
  return res.json();
}
