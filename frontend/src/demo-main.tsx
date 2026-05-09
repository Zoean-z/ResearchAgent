import React from "react";
import ReactDOM from "react-dom/client";

import App from "./App";
import { installDemoApi } from "./lib/demoApi";
import "./styles.css";

installDemoApi();

function DemoBanner() {
  return (
    <div className="demo-banner">
      <strong>Static Demo</strong>
      <span>Read-only showcase using the real frontend with mocked session data.</span>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <DemoBanner />
    <App />
  </React.StrictMode>,
);
