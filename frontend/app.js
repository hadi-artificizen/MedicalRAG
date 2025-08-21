const API = "http://localhost:8000";

// DOM refs
const $ = id => document.getElementById(id);
const authForms = $("auth-forms"), userInfo = $("user-info");
const authMsg = $("auth-msg"), userEmailEl = $("user-email");
const chatHeader = $("chat-header"), chatBox = $("chat-box"), chatInputs = $("chat-inputs");

// State
let token = localStorage.getItem("token"), email = localStorage.getItem("email"), chatroomId = localStorage.getItem("chatroom");

// Helpers
const authHeaders = () => token ? { Authorization: `Bearer ${token}` } : {};
const saveSession = () => {
  localStorage.setItem("token", token || "");
  localStorage.setItem("email", email || "");
  localStorage.setItem("chatroom", chatroomId || "");
};
const ui = {
  loggedIn: () => {
    authForms.hidden = true; userInfo.hidden = false;
    $("chatroom-section").hidden = $("upload-section").hidden = false;
    userEmailEl.textContent = email;
  },
  loggedOut: () => {
    authForms.hidden = false; userInfo.hidden = true;
    $("chatroom-section").hidden = $("upload-section").hidden = true;
    chatInputs.hidden = true; chatHeader.textContent = "Please log in to start chatting";
    chatBox.innerHTML = "";
  }
};

// Fetch wrapper
async function api(url, options={}) {
  const res = await fetch(API+url, { ...options, headers:{...authHeaders(), ...options.headers}});
  if(res.status===401){ logout(); throw Error("Unauthorized"); }
  return res.ok ? res.json() : Promise.reject(await res.text());
}

// Auth
async function login(email, password) {
  const data = await api("/login",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({email,password})});
  token=data.access_token; saveSession(); ui.loggedIn(); loadChatrooms();
}
async function signup(email, password) {
  return api("/register",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({email,password})});
}
function logout(){ token=email=chatroomId=null; saveSession(); ui.loggedOut(); }

// Chatrooms
async function loadChatrooms(){
  const {chatrooms=[]}= await api(`/chatrooms/${email}`);
  const sel=$("chatroom"); sel.innerHTML="";
  chatrooms.forEach(([id])=>sel.add(new Option("Chatroom "+id,id)));
  chatroomId = chatroomId || chatrooms[0]; sel.value=chatroomId;
  chatInputs.hidden=!chatroomId; chatHeader.textContent=`Chat - Chatroom #${chatroomId}`;
  loadMessages(chatroomId);
}
async function newChatroom(){
  const {chatroom_id}= await api(`/new_chatroom/${email}`,{method:"POST"});
  chatroomId=chatroom_id; saveSession(); loadChatrooms();
}

// Messages
async function loadMessages(id){
  const {messages=[]}= await api(`/messages/${id}`);
  chatBox.innerHTML="";
  messages.forEach(([u,a])=>{ if(u) msg(u,"user"); if(a) msg(a,"assistant"); });
}
function msg(text,cls){ const d=document.createElement("div"); d.className=`message ${cls}`; d.textContent=text; chatBox.append(d); chatBox.scrollTop=chatBox.scrollHeight; }
async function send(){
  const q=$("chat-input").value.trim(); if(!q) return;
  msg(q,"user"); $("chat-input").value="";
  const {answer,email_sent}= await api("/query",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({email,chatroom_id:chatroomId,Query:q})});
  msg(answer||"No answer","assistant");
  if(email_sent){ $("upload-msg").textContent="Prescription sent to your email."; }
}

// Upload
async function uploadFiles(){
  const f=$("file-upload").files; if(!f.length) return $("upload-msg").textContent="No files selected.";
  const fd=new FormData(); [...f].forEach(x=>fd.append("files",x));
  const {message}= await api("/upload",{method:"POST",body:fd}); $("upload-msg").textContent=message;
}

// Events
$("auth-btn").onclick= async()=>{
  try{
    const e=$("email").value,p=$("password").value,act=document.querySelector("input[name=auth]:checked").value;
    if(act==="login") await login(e,p); else {await signup(e,p); authMsg.textContent="Signup successful, please login.";}
  }catch(err){authMsg.textContent=err;}
};
$("logout").onclick=logout;
$("new-chatroom").onclick=newChatroom;
$("chatroom").onchange=e=>{chatroomId=e.target.value; saveSession(); loadMessages(chatroomId);};
$("upload-btn").onclick=uploadFiles;
$("send").onclick=send;

// Init
token&&email?ui.loggedIn()&&loadChatrooms():ui.loggedOut();
