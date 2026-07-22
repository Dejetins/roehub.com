import { QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { createPrototypeQueryClient, PrototypeApp } from "./App";
import "./metrics";
import { UiStore } from "./state/uiStore";
import "./styles.css";

const root = document.getElementById("root");
if (!root) throw new Error("Prototype root element is missing");

createRoot(root).render(
  <StrictMode>
    <QueryClientProvider client={createPrototypeQueryClient()}>
      <PrototypeApp uiStore={new UiStore()} />
    </QueryClientProvider>
  </StrictMode>,
);
