// common.js
// Put this file in same folder and include in pages.
// Replace FIREBASE_CONFIG and BASE_URL as needed.

const BASE_URL = localStorage.getItem('FS_BASE_URL') || "http://127.0.0.1:8000";

// FIREBASE CONFIG - replace with your project's values
const FIREBASE_CONFIG = {
  apiKey: "REPLACE_FIREBASE_APIKEY",
  authDomain: "REPLACE_AUTH_DOMAIN",
  projectId: "REPLACE_PROJECT_ID",
  appId: "REPLACE_APPID"
};

// load firebase compat libs dynamically
function loadFirebase(onLoaded){
  if(window.firebase) return onLoaded && onLoaded();
  const s = document.createElement('script'); s.src = "https://www.gstatic.com/firebasejs/9.22.2/firebase-app-compat.js";
  s.onload = () => {
    const a = document.createElement('script'); a.src = "https://www.gstatic.com/firebasejs/9.22.2/firebase-auth-compat.js";
    a.onload = () => {
      firebase.initializeApp(FIREBASE_CONFIG);
      window.fAuth = firebase.auth();
      onLoaded && onLoaded();
    };
    document.head.appendChild(a);
  };
  document.head.appendChild(s);
}

// helper to show friendly errors (no raw JSON)
function showError(container, err){
  let message = "Unknown error";
  if(!err) message = "Unknown error";
  else if(typeof err === "string") message = err;
  else if(err.detail) message = err.detail;
  else if(err.message) message = err.message;
  container.innerHTML = `<div style="color:#ffd5d5">${escapeHtml(message)}</div>`;
}

// sanitize for innerText
function escapeHtml(str){
  return String(str).replace(/[&<>"'`=\/]/g, s=>({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;","/":"&#x2F;","`":"&#96;","=":"&#61;" })[s]);
}

// short & relevant response (trim long replies - 1-3 sentences)
function shortReply(text){
  if(!text) return "";
  // split into sentences by period or newline
  let s = text.replace(/\n+/g, ' ').trim();
  const parts = s.split(/(?<=[.!?])\s+/);
  let out = parts.slice(0,2).join(' ').trim();
  if(out.length > 280) out = out.slice(0,277) + '...';
  return out;
}

// function to download a small PDF from HTML text using jsPDF
function downloadPdf(title, lines){
  const s = document.createElement('script');
  s.src = "https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js";
  s.onload = ()=>{
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    doc.setFontSize(14); doc.text(title, 14, 20);
    doc.setFontSize(11);
    let y = 34;
    lines.forEach(line=>{
      const wrap = doc.splitTextToSize(line, 180);
      doc.text(wrap, 14, y);
      y += wrap.length * 7 + 6;
      if(y > 270){ doc.addPage(); y = 20; }
    });
    const filename = `${title.replace(/\s+/g,'_')}_${Date.now()}.pdf`;
    doc.save(filename);
  };
  document.head.appendChild(s);
}
