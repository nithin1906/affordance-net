// src/api.js
export async function inferImage(blobOrFile) {
  // 1. Determine Backend URL dynamically
  let backendUrl = import.meta.env.VITE_BACKEND_URL;

  if (!backendUrl) {
      // If no env var (Production/Docker), point to the same host on port 8000
      const protocol = window.location.protocol;
      const hostname = window.location.hostname;
      backendUrl = `${protocol}//${hostname}:8000`;
  }

  const url = `${backendUrl}/v1/infer`;
  const fd = new FormData();
  fd.append("image", blobOrFile, "capture.jpg");

  console.log(`[API] Sending request to: ${url}`);

  try {
    const res = await fetch(url, {
      method: "POST",
      body: fd,
    });

    if (!res.ok) {
      const text = await res.text();
      console.error("[API] Server Error:", text);
      throw new Error(`Server Error ${res.status}: ${text}`);
    }
    
    const json = await res.json();
    console.log("[API] Success:", json);
    return json;

  } catch (err) {
    console.error("[API] Network/Parsing Error:", err);
    // Re-throw with a user-friendly message
    throw new Error(`Connection Failed: ${err.message}`);
  }
}