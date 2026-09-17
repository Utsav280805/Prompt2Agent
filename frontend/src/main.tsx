// frontend/src/main.tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { App } from "./App";
import "./index.css";

const root = document.getElementById("root");
if (!root) throw new Error("#root is missing from index.html");

createRoot(root).render(
  <StrictMode>
    {/* BrowserRouter, not HashRouter: the backend serves index.html for any unmatched path, so
        real URLs work on refresh and are shareable. */}
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </StrictMode>,
);
