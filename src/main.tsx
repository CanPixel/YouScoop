import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import ErrorBoundary from "./components/ErrorBoundary";

// WKWebView has no console the user can reach, so surface anything that escapes
// React's boundary (async throws, rejected promises) instead of losing it.
window.addEventListener("error", (e) => {
  console.error("Uncaught error", e.error ?? e.message);
});
window.addEventListener("unhandledrejection", (e) => {
  console.error("Unhandled rejection", e.reason);
});

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>,
);
