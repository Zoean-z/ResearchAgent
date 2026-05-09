import React from "react";
import ReactDOM from "react-dom/client";

import App from "./App";
import { installDemoApi } from "./lib/demoApi";
import "./styles.css";

installDemoApi();

function DemoBanner() {
  return (
    <div className="demo-banner">
      <strong>静态演示</strong>
      <span>使用真实前端和预制会话数据的只读展示。</span>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <DemoBanner />
    <App />
  </React.StrictMode>,
);
